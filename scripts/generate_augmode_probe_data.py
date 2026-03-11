"""
Aug-mode probe data generation (optical flow version).

For each Libero subtask: reset env to the task initial state, run a fixed number of
probe steps (y-axis only, ±probe_delta per step), record before/after images,
compute optical flow, action sign, and label aug_mode; save as .npz for
AugModeProbeDataset. After each sample we reset again so the probe is always
executed from the initial state.

aug_mode vs augmentation (aligned with train/eval quadrants):
  0: Normal (no flip, action +y)
  1: Horizontally flip images (flip before/after then compute flow), action sign still +y
  2: Action invert (probe with -y, i.e. -probe_delta), no image flip
  3: Both: horizontal flip + action invert (probe -y, then flip images)

Requires: opencv-python (pip install opencv-python) for optical flow.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tqdm

# Allow imports from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import cv2
except ImportError:
    raise ImportError("Please install opencv-python: pip install opencv-python")

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
)


# Probe action: y-axis only ±delta, rest 0, gripper -1 (same as dummy)
def get_probe_action(delta: float = 0.2):
    """Return 7D action [x, y, z, roll, pitch, yaw, gripper]; only y-axis is delta (signed)."""
    return [0.0, float(delta), 0.0, 0.0, 0.0, 0.0, -1.0]


def get_probe_action_sign(y_sign: int):
    """Action sign for the probe: y_sign in {+1, -1} for y-axis direction."""
    return np.array([0, y_sign, 0, 0, 0, 0, 0], dtype=np.float32)


def flip_image_horizontal(img: np.ndarray) -> np.ndarray:
    """Horizontally flip image (same as eval flip_type='horizontal')."""
    return np.flip(img, axis=1).copy()


def compute_optical_flow(before_bgr: np.ndarray, after_bgr: np.ndarray) -> np.ndarray:
    """
    Compute optical flow (H, W, 2) from before/after frames; format matches training.
    Uses OpenCV Farneback; output is (dx, dy) per pixel.
    """
    gray_before = cv2.cvtColor(before_bgr, cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(after_bgr, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_before, gray_after, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    return flow.astype(np.float32)


def run_probe_and_capture(env, initial_obs, probe_steps: int, probe_delta: float):
    """
    From initial_obs (right after set_init_state), run probe_steps of probe (y-axis probe_delta per step).
    Returns before/after main and wrist images. probe_delta can be positive or negative (+y or -y).
    """
    probe_action = get_probe_action(probe_delta)

    # Before probe (initial state)
    before_main = get_libero_image(initial_obs)
    before_wrist = get_libero_wrist_image(initial_obs)

    # Run probe
    obs = initial_obs
    for _ in range(probe_steps):
        obs, _, _, _ = env.step(probe_action)

    # After probe
    after_main = get_libero_image(obs)
    after_wrist = get_libero_wrist_image(obs)

    return before_main, after_main, before_wrist, after_wrist


def main():
    parser = argparse.ArgumentParser(description="Generate aug-mode probe data (optical flow version).")
    parser.add_argument("--task_suites", type=str, default="all",
                        help="Task suite(s): 'all' or comma-separated (e.g. libero_spatial,libero_object).")
    parser.add_argument("--output_dir", type=str, default="data/libero_augmode_probe_flow",
                        help="Output root; each suite is saved under output_dir/<task_suite_name>/.")
    parser.add_argument("--probe_steps", type=int, default=5,
                        help="Number of fixed steps for probe (y-axis only).")
    parser.add_argument("--probe_delta", type=float, default=0.2,
                        help="Delta per step on y-axis.")
    parser.add_argument("--num_initial_states_per_task", type=int, default=10,
                        help="Max number of initial states to sample per task (uses first N from suite).")
    parser.add_argument("--image_res", type=int, default=256,
                        help="Environment image resolution (camera height/width).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suites.strip().lower() == "all":
        suite_names = sorted(benchmark_dict.keys())
    else:
        suite_names = [s.strip() for s in args.task_suites.split(",") if s.strip()]
    for name in suite_names:
        if name not in benchmark_dict:
            raise ValueError(f"Unknown task suite: {name}. Available: {list(benchmark_dict.keys())}")

    num_samples_per_suite = {}
    for task_suite_name in suite_names:
        suite_output_dir = os.path.join(args.output_dir, task_suite_name)
        os.makedirs(suite_output_dir, exist_ok=True)

        task_suite = benchmark_dict[task_suite_name]()
        num_tasks = task_suite.n_tasks
        sample_paths = []
        total = 0

        for task_id in range(num_tasks):
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, "openvla", resolution=args.image_res)
            env.seed(args.seed)

            initial_states = task_suite.get_task_init_states(task_id)
            n_inits = min(args.num_initial_states_per_task, len(initial_states))

            for init_idx in tqdm.tqdm(
                range(n_inits),
                desc=f"{task_suite_name} task {task_id}/{num_tasks}",
                leave=False,
            ):
                # Run +y probe first (for aug_mode 0 and 1)
                env.reset()
                obs = env.set_init_state(initial_states[init_idx])
                before_plus, after_plus, before_wrist_plus, after_wrist_plus = run_probe_and_capture(
                    env, obs, args.probe_steps, args.probe_delta
                )
                # Run -y probe (for aug_mode 2 and 3); reset to same initial state again
                env.reset()
                obs = env.set_init_state(initial_states[init_idx])
                before_minus, after_minus, before_wrist_minus, after_wrist_minus = run_probe_and_capture(
                    env, obs, args.probe_steps, -args.probe_delta
                )

                # Four aug_modes = four quadrant augmentations; each (task, init) yields 4 samples
                # 0: normal (+y, no flip)  1: flip image (+y, flip before/after)
                # 2: invert action (-y, no flip)  3: both (-y, flip before/after)
                samples_to_save = [
                    (before_plus, after_plus, before_wrist_plus, after_wrist_plus, +1, 0),
                    (flip_image_horizontal(before_plus), flip_image_horizontal(after_plus),
                     flip_image_horizontal(before_wrist_plus), flip_image_horizontal(after_wrist_plus), +1, 1),
                    (before_minus, after_minus, before_wrist_minus, after_wrist_minus, -1, 2),
                    (flip_image_horizontal(before_minus), flip_image_horizontal(after_minus),
                     flip_image_horizontal(before_wrist_minus), flip_image_horizontal(after_wrist_minus), -1, 3),
                ]
                for before_main, after_main, before_wrist, after_wrist, y_sign, aug_mode in samples_to_save:
                    flow = compute_optical_flow(before_main, after_main)
                    action_sign = get_probe_action_sign(y_sign)

                    fname = f"sample_{total:06d}.npz"
                    out_path = os.path.join(suite_output_dir, fname)
                    np.savez_compressed(
                        out_path,
                        before_main=before_main,
                        after_main=after_main,
                        before_wrist=before_wrist,
                        after_wrist=after_wrist,
                        flow=flow,
                        action_sign=action_sign,
                        aug_mode=np.int64(aug_mode),
                        task_id=np.int64(task_id),
                        init_state_idx=np.int64(init_idx),
                        task_description=np.array(task_description, dtype=object),
                    )
                    sample_paths.append(fname)
                    total += 1

            env.close()
            del env

        manifest = {
            "task_suite": task_suite_name,
            "probe_steps": args.probe_steps,
            "probe_delta": args.probe_delta,
            "num_tasks": num_tasks,
            "num_initial_states_per_task": args.num_initial_states_per_task,
            "image_res": args.image_res,
            "seed": args.seed,
            "num_samples": total,
            "samples": sample_paths,
            "aug_mode_variants": 4,
        }
        with open(os.path.join(suite_output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        num_samples_per_suite[task_suite_name] = total
        print(f"Suite {task_suite_name}: saved {total} samples to {suite_output_dir}")

    # Top-level index: list of suites and sample counts
    index = {
        "task_suites": suite_names,
        "num_samples_per_suite": num_samples_per_suite,
        "total_samples": sum(num_samples_per_suite.values()),
    }
    with open(os.path.join(args.output_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Done. Total {index['total_samples']} samples across {len(suite_names)} suite(s). Index: {args.output_dir}/index.json")


if __name__ == "__main__":
    main()
