"""
Simplified LIBERO evaluation script for quick testing.

What it does:
- Open a LIBERO env (you pick the task suite and task id)
- Execute a predefined action trajectory (from a .npy/.npz file) or a short default dummy trajectory
- Save the rollout as an MP4 in the chosen output directory

Notes:
- This script reuses your project's helper functions from
  `experiments.robot.libero.libero_utils` when available (recommended).
- If those helpers are not importable the script will exit with a helpful
  message (so you know to run it from the repository root or add the
  helpers to PYTHONPATH).

Usage example:
    python eval_libero_simplified.py --task_suite libero_spatial --task_id 0 \
        --action_file ./actions.npy --save_dir ./rollouts

If no --action_file is provided the script will run a short default dummy
trajectory using `get_libero_dummy_action`.


"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import imageio
import tqdm

# Add repo root (optional) so local helper imports work when running from scripts/
# Adjust this if you run from a different location.
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        get_libero_wrist_image,
        save_rollout_video as helper_save_rollout_video,
    )
except Exception as e:
    raise ImportError(
        "Failed to import project helpers. Make sure you run this script from the project root "
        "and that `experiments.robot.libero.libero_utils` is importable. Original error: {}".format(e)
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def load_actions_from_file(path: str):
    """Load actions from a .npy or .npz file. Expect an array of shape (T, action_dim)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Action file not found: {path}")

    if path.suffix == ".npz":
        data = np.load(path)
        # prefer an array named 'actions' if present
        if "actions" in data.files:
            actions = data["actions"]
        else:
            # take the first array
            actions = data[data.files[0]]
    else:
        actions = np.load(path)

    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape (T, action_dim), got {actions.shape}")
    return actions


def save_video_fallback(frames, out_path, fps=20):
    """Save a list/array of uint8 frames to out_path (mp4) using imageio."""
    ensure_dir(Path(out_path).parent.as_posix())
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()


def build_default_actions(num_steps: int, model_family: str = "openvla"):
    """Create a short dummy action trajectory using project helpers.
    Returns an ndarray shape (num_steps, action_dim).
    """
    # get a single dummy action and tile it
    dummy = get_libero_dummy_action(model_family)
    # dummy might be a list or np.array; convert to np.array
    dummy = np.array(dummy)
    actions = np.tile(dummy[np.newaxis, :], (num_steps, 1))
    return actions

def generate_smooth_random_trajectory(length=200):
    """
    生成一段长度为 length 的平滑随机连续轨迹。
    使用不同频率和相位的正弦波叠加，模拟复杂但连续的机械臂运动。
    """
    t = np.arange(length)
    actions = np.zeros((length, 7))
    
    # 为 6 个自由度随机生成频率 (决定运动快慢) 和相位 (决定起始方向)
    np.random.seed(42) # 固定随机种子，保证每次生成的随机轨迹一样
    freqs = np.random.uniform(0.02, 0.08, size=6)
    phases = np.random.uniform(0, 2 * np.pi, size=6)
    
    amps = np.array([0.3]*6) 
    
    # 生成连续的 delta actions
    for i in range(6):
        actions[:, i] = amps[i] * np.sin(freqs[i] * t + phases[i])
    
    noise = 0.1 * np.random.randn(length, 6)
    actions[:, :6] += noise

    # 夹爪 (索引 6)：每 50 步开合一次
    actions[:, 6] = np.where((t // 50) % 2 == 0, 1.0, -1.0)
    
    return actions

def mirror_trajectory(actions):
    """
    你的核心镜像公式！
    假设水平翻转图像，左变右：Y(1), Roll(3), Yaw(5) 取反
    """
    mirrored = np.copy(actions)
    mirrored[:, 1] = -actions[:, 1]  # 反转 Y
    mirrored[:, 3] = -actions[:, 3]  # 反转 Roll
    mirrored[:, 5] = -actions[:, 5]  # 反转 Yaw
    return mirrored

def run_trajectory(env, actions, max_steps=None):
    """Run actions in env and collect frames. Returns (success_flag, frames).

    This function is intentionally minimal: it executes the provided actions
    sequentially. If env returns done=True it stops early and marks success.
    """
    frames = []
    success = False

    # Reset env and get initial observation
    try:
        obs = env.reset()
    except TypeError:
        # some envs accept reset(return_info=True)
        obs = env.reset()

    for step_idx, action in tqdm.tqdm(enumerate(actions), total=len(actions), desc="Running trajectory"):
        # collect an image for the replay
        try:
            frame = get_libero_image(obs)
        except Exception:
            # fallback: try to read a common RGB key
            frame = obs.get("rgb_third", None) or obs.get("rgb", None) or obs.get("image", None)
            if frame is None:
                raise RuntimeError("Could not extract an image from the observation; make sure get_libero_image is available.")

        frames.append(frame)

        # step the env
        obs, reward, done, info = env.step(action.tolist())

        if done:
            success = True
            break

        # optional hard limit
        if max_steps is not None and (step_idx + 1) >= max_steps:
            break

    return success, frames


def main():
    parser = argparse.ArgumentParser(description="Simplified LIBERO eval: run a fixed action trajectory and save rollout.")
    parser.add_argument("--task_suite", type=str, default="libero_spatial", help="Which LIBERO task suite (e.g. libero_spatial)")
    parser.add_argument("--task_id", type=int, default=0, help="Task id / subtask index (int)")
    parser.add_argument("--action_file", type=str, default=None, help="Path to .npy or .npz file with actions (shape T x action_dim)")
    parser.add_argument("--num_default_steps", type=int, default=1000, help="If no action_file given, number of dummy steps to run")
    parser.add_argument("--save_dir", type=str, default="./rollouts/tmp", help="Directory to save the rollout video")
    parser.add_argument("--model_family", type=str, default="openvla", help="Model family string used by get_libero_dummy_action/get_libero_env")
    parser.add_argument("--env_img_res", type=int, default=256, help="Environment image resolution to request from get_libero_env")
    parser.add_argument("--max_steps", type=int, default=None, help="Hard cap on executed steps (overrides action length if smaller)")
    parser.add_argument("--fps", type=int, default=20, help="FPS for saved video")
    args = parser.parse_args()

    save_dir = ensure_dir(args.save_dir)

    # Build task suite and task
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite not in benchmark_dict:
        raise ValueError(f"Unknown task suite: {args.task_suite}. Available: {list(benchmark_dict.keys())}")

    task_suite = benchmark_dict[args.task_suite]()
    if args.task_id < 0 or args.task_id >= task_suite.n_tasks:
        raise ValueError(f"Invalid task_id {args.task_id} for suite {args.task_suite} (n_tasks={task_suite.n_tasks})")

    task = task_suite.get_task(args.task_id)

    # Create environment via helper
    env, task_description = get_libero_env(task, args.model_family, resolution=args.env_img_res)
    print(f"Created env for task: {task_description}")

    # ===== Build manual trajectory (length = 600) =====

    # actions = []

    # action_dim = 7
    # gripper_value = -1  # 不管夹爪，固定

    # for dim in range(6):  # 只动前6个维度
    #     # +0.1 * 50
    #     for _ in range(50):
    #         a = np.zeros(action_dim)
    #         a[dim] = 0.1
    #         a[6] = gripper_value
    #         actions.append(a)

    #     # -0.1 * 50
    #     for _ in range(50):
    #         a = np.zeros(action_dim)
    #         a[dim] = -0.1
    #         a[6] = gripper_value
    #         actions.append(a)
    actions = generate_smooth_random_trajectory(length=200)
    actions_mirr = mirror_trajectory(actions)

    # Run trajectory
    success, frames = run_trajectory(env, actions, max_steps=args.max_steps)

    # Run mirrored trajectory
    success_mirr, frames_mirr = run_trajectory(env, actions_mirr, max_steps=args.max_steps)
    
    # Combine frames (side by side)
    combined_frames = []
    for f1, f2 in zip(frames, frames_mirr):
        combined = np.concatenate([f1, f2], axis=1)  # 水平拼接
        combined_frames.append(combined)
    
    # Save rollout
    out_filename = f"rollout_{args.task_suite}_task{args.task_id}.mp4"
    out_path = Path(save_dir) / out_filename


    print("Project helper save_rollout_video failed or not available — falling back to imageio writer.")
    save_video_fallback(combined_frames, out_path, fps=args.fps)
    print(f"Saved video to: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
