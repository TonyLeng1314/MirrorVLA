"""
Train the aug-mode discriminator (AugModePredictor) on probe data.

- Data: AugModeProbeDataset (flow + action_sign -> aug_mode 0/1/2/3).
- Loss: cross-entropy.
- Logs: train loss to wandb; after training, evaluates and logs accuracy.
"""

import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import tqdm
import wandb

from prismatic.vla.datasets import AugModeProbeDataset
from prismatic.models.aug_mode_predictor import AugModePredictor


def parse_args():
    p = argparse.ArgumentParser(description="Train aug-mode predictor (discriminator).")
    p.add_argument("--data_root", type=str, default="data/libero_augmode_probe_flow",
                   help="Root dir with index.json and per-suite subdirs (or single-suite dir with manifest.json).")
    p.add_argument("--output_dir", type=str, default="outputs/augmode_predictor",
                   help="Dir to save checkpoints and logs.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.2,
                   help="Fraction of data for validation (0 to disable).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--flow_clip", type=float, default=20.0,
                   help="Clip flow to [-flow_clip, flow_clip] in dataset.")
    p.add_argument("--flow_embed_dim", type=int, default=64)
    p.add_argument("--action_sign_embed_dim", type=int, default=32)
    # wandb (same as finetune.py: no entity arg, use default from wandb login / WANDB_ENTITY env)
    p.add_argument("--wandb_project", type=str, default="augmode_predictor")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log_interval", type=int, default=10,
                   help="Log train/step_loss to wandb every this many steps.")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device, epoch: int, log_interval: int = 10) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    step = 0
    pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        flow = batch["flow"].to(device, dtype=torch.float32).permute(0, 3, 1, 2)  # (B, H, W, 2) -> (B, 2, H, W)
        action_sign = batch["action_sign"].to(device, dtype=torch.float32)
        labels = batch["aug_mode"].to(device, dtype=torch.long)

        logits = model(flow, action_sign)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        step += 1
        pbar.set_postfix(loss=loss.item())
        if wandb.run is not None and step % log_interval == 0:
            wandb.log({"train/step_loss": loss.item(), "epoch": epoch})
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm.tqdm(loader, desc="Eval", leave=False):
        flow = batch["flow"].to(device, dtype=torch.float32).permute(0, 3, 1, 2)
        action_sign = batch["action_sign"].to(device, dtype=torch.float32)
        labels = batch["aug_mode"].to(device, dtype=torch.long)

        logits = model(flow, action_sign)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset: all suites (or single suite) with optional flow_clip
    dataset = AugModeProbeDataset(
        args.data_root,
        include_images=False,
        flow_clip=args.flow_clip,
    )
    n_total = len(dataset)

    if args.val_ratio > 0 and args.val_ratio < 1:
        n_val = int(n_total * args.val_ratio)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_ds = dataset
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_ds else None

    model = AugModePredictor(
        flow_embed_dim=args.flow_embed_dim,
        action_sign_embed_dim=args.action_sign_embed_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_name = args.wandb_run_name or f"augmode_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"
    if args.wandb_mode != "disabled":
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            mode=args.wandb_mode,
        )

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.log_interval)
        if wandb.run is not None:
            wandb.log({"train/epoch_loss": train_loss, "epoch": epoch})
        print(f"Epoch {epoch} train loss: {train_loss:.4f}")

        if val_loader is not None:
            acc = evaluate(model, val_loader, device)
            if wandb.run is not None:
                wandb.log({"eval/accuracy": acc, "epoch": epoch})
            print(f"Epoch {epoch} val accuracy: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                ckpt_path = os.path.join(args.output_dir, "best.pt")
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_accuracy": acc}, ckpt_path)
                print(f"  -> saved best checkpoint to {ckpt_path}")

    # Final checkpoint
    torch.save(
        {"model": model.state_dict(), "epoch": args.epochs, "config": vars(args)},
        os.path.join(args.output_dir, "last.pt"),
    )
    print(f"Saved last checkpoint to {args.output_dir}/last.pt")

    # Final evaluation: report accuracy (on val if available, else on train for reference)
    if val_loader is not None:
        final_acc = evaluate(model, val_loader, device)
        print(f"Final validation accuracy: {final_acc:.4f}")
        if wandb.run is not None:
            wandb.log({"eval/final_accuracy": final_acc})
    else:
        final_acc = evaluate(model, train_loader, device)
        print(f"Final train accuracy (no val split): {final_acc:.4f}")
        if wandb.run is not None:
            wandb.log({"eval/train_accuracy": final_acc})

    if wandb.run is not None:
        wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
