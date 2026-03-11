torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 20000 \
  --max_steps 20005 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity 1d75f18b4cabe74eeb95f3f7cde473708d3d0ce1 \
  --wandb_project libero_spatial_no_noops \
  --run_id_note VLA-Adapter--spatial--manual-run \
  --wandb_mode offline \
  --mask_cam None \
  --enable_four_quadrant_input True \
  --history_frames 2




  PYTHONPATH=/home/zlwanggroup/hanpg/projects/vla_adapter_project/envs/LIBERO:$PYTHONPATH python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 1 \
  --use_film False \
  --pretrained_checkpoint outputs/configs+libero_spatial_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--spatial--20260119_163346--20000_chkpt \
  --use_pro_version True \
  --flip_ratio 0.0 \
  --cat_wrist True \
  --mask_cam None \
  --enable_view_fusion False \
  --view_fusion_invert False




  quick use libero dummy test:
  PYTHONPATH=/home/zlwanggroup/hanpg/projects/vla_adapter_project/envs/LIBERO:$PYTHONPATH python experiments/robot/libero/eval_libero_simplified.py



---

## Generate libero aug-mode probe dataset (optical flow)

From repo root. Requires: `pip install opencv-python` and LIBERO env (e.g. `PYTHONPATH` with libero).

Output layout: each task suite is under `output_dir/<task_suite_name>/` (e.g. `data/libero_augmode_probe_flow/libero_spatial/`, `.../libero_object/`). A top-level `index.json` lists all suites and sample counts.

**All task suites (default):**
```bash
python scripts/generate_augmode_probe_data.py \
  --output_dir data/libero_augmode_probe_flow
# or explicitly:
python scripts/generate_augmode_probe_data.py \
  --task_suites all \
  --output_dir data/libero_augmode_probe_flow
```

**Specific task suite(s), comma-separated:**
```bash
python scripts/generate_augmode_probe_data.py \
  --task_suites libero_spatial,libero_object,libero_goal \
  --output_dir data/libero_augmode_probe_flow
```

**Custom probe params and inits:**
```bash
python scripts/generate_augmode_probe_data.py \
  --task_suites all \
  --output_dir data/libero_augmode_probe_flow \
  --probe_steps 5 \
  --probe_delta 0.2 \
  --num_initial_states_per_task 20 \
  --image_res 256 \
  --seed 42
```

**With LIBERO on PYTHONPATH (e.g. cluster):**
```bash
PYTHONPATH=/path/to/envs/LIBERO:$PYTHONPATH python scripts/generate_augmode_probe_data.py \
  --task_suites all \
  --output_dir data/libero_augmode_probe_flow \
  --num_initial_states_per_task 15
```

**Load all task suites (one dataset, random read across suites):**

Pass the root dir that contains `index.json`; the dataset merges all suites into one flat list. Use `DataLoader(shuffle=True)` to sample randomly from all suites.

```python
from torch.utils.data import DataLoader
from prismatic.vla.datasets import AugModeProbeDataset

ds = AugModeProbeDataset(
    "data/libero_augmode_probe_flow",  # root with index.json + libero_spatial/, libero_object/, ...
    include_images=False,
    flow_clip=20.0,
)
# len(ds) = total samples across all suites
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
batch = next(iter(loader))
# batch["flow"], batch["action_sign"], batch["aug_mode"]; batch["task_suite"] gives suite name per sample
```

**Load a single suite only:**
```python
ds = AugModeProbeDataset(
    "data/libero_augmode_probe_flow/libero_spatial",
    include_images=False,
    flow_clip=20.0,
)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
```

---

## Train aug-mode discriminator

Cross-entropy loss, wandb logging, train/val split, accuracy evaluation at the end.

```bash
python vla-scripts/train_augmode_predictor.py \
  --data_root data/libero_augmode_probe_flow \
  --output_dir outputs/augmode_predictor \
  --batch_size 64 \
  --epochs 30 \
  --lr 1e-3 \
  --val_ratio 0.2 \
  --wandb_project augmode_predictor \
  --wandb_entity your-entity
```

Checkpoints: `outputs/augmode_predictor/best.pt` (best val accuracy), `last.pt` (final). Wandb: `train/step_loss`, `train/epoch_loss`, `eval/accuracy`, `eval/final_accuracy`.