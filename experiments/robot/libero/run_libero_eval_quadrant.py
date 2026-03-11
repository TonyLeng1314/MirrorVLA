"""
run_libero_eval_quadrant.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark
from collections import Counter

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.augmode_probe_utils import (
    compute_optical_flow,
    get_probe_action,
    get_probe_action_sign,
)
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.models.aug_mode_predictor import AugModePredictor
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, GRIPPER_RANGE


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}

# Four quadrants definition
QUADRANTS = [
    {"name": "Normal",       "flip_ratio": 0.0, "invert_action": False},
    {"name": "Image Flip",   "flip_ratio": 1.0, "invert_action": False},
    {"name": "Action Flip",  "flip_ratio": 0.0, "invert_action": True},
    {"name": "Both Flip",    "flip_ratio": 1.0, "invert_action": True},
]


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "vla-adapter"                # version of 
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    phase: str = "Inference"
    
    #################################################################################################################
    # Mirror related
    #################################################################################################################
    flip_ratio: float= 0.0                           # Ratio of flipping the image horizontally during evaluation
    flip_type: str = 'horizontal'                    # 'horizontal', 'vertical', or 'both', the type of flipping to apply if flip_ratio > 0
    cat_wrist: bool = True                           # Whether to concatenate wrist camera image to the side of the third-person image in the replay video
    mask_cam: str = 'None'                           # 'None', 'wrist', 'third' , which camera to mask during evaluation
    invert_action: bool = False                      # Whether to invert horizontal action dimension
    
    enable_view_fusion: bool = False                 # Enable view fusion during evaluation
    view_fusion_invert: bool = False                 # Invert the view fusion weights during evaluation

    #################################################################################################################
    # History (disabled; kept for compatibility)
    #################################################################################################################
    history_frames: int = 0                           # Deprecated
    history_compression_rate: int = 1                 # Deprecated
    normalize_history_actions_sign_only: bool = False  # Deprecated

    #################################################################################################################
    # Quadrant eval
    #################################################################################################################
    enable_quadrant_eval: bool = False               # If True, run 4-quadrant evaluation

    #################################################################################################################
    # Aug-mode conditioning (use GT mode when discriminator not ready)
    #################################################################################################################
    use_aug_mode_condition: bool = False             # If True, action head receives aug_mode; set True for aug-mode-trained checkpoints

    #################################################################################################################
    # Aug-mode predictor (discriminator): probe at episode start, predict aug_mode, then restore env
    #################################################################################################################
    use_aug_mode_predictor: bool = False             # If True, load discriminator and run probe before each episode to predict aug_mode
    aug_mode_predictor_checkpoint: str = ""          # Path to predictor checkpoint (e.g. outputs/augmode_predictor/best.pt)
    probe_steps: int = 5                             # Must match data generation; number of y-axis probe steps
    probe_delta: float = 0.2                         # Must match data generation; delta per step on y-axis
    flow_clip: float = 20.0                          # Clip flow for predictor input (match training)


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    if getattr(cfg, "use_aug_mode_predictor", False):
        assert cfg.aug_mode_predictor_checkpoint, "aug_mode_predictor_checkpoint required when use_aug_mode_predictor=True"
        assert cfg.use_aug_mode_condition, "use_aug_mode_condition must be True when using aug_mode_predictor"


def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)
    model.set_version(cfg.save_version)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    aug_mode_predictor = None
    if getattr(cfg, "use_aug_mode_predictor", False) and getattr(cfg, "aug_mode_predictor_checkpoint", ""):
        ckpt = torch.load(cfg.aug_mode_predictor_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        aug_mode_predictor = AugModePredictor()
        aug_mode_predictor.load_state_dict(state, strict=True)
        aug_mode_predictor.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        aug_mode_predictor = aug_mode_predictor.to(device)
        logger.info(f"Loaded aug_mode_predictor from {cfg.aug_mode_predictor_checkpoint}")

    return model, action_head, proprio_projector, noisy_action_projector, processor, aug_mode_predictor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    initial_states = task_suite.get_task_init_states(task_id)
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size, flip=False, cfg: GenerateConfig = None):
    img = get_libero_image(obs)
    if flip:
        if cfg.flip_type == 'horizontal':
            img = np.flip(img, axis=1).copy()
        elif cfg.flip_type == 'vertical':
            img = np.flip(img, axis=0).copy()
        elif cfg.flip_type == 'both':
            img = np.flip(img, axis=(0, 1)).copy()
    wrist_img = get_libero_wrist_image(obs)

    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img


def process_action(action, model_family, invert_action=False):
    """Process action before sending to environment."""
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    # Invert horizontal action dimension for quadrant eval
    # action[:3] are [x, y, z] translations; action[0] is left/right
    if invert_action:
        action[1] *= -1   # invert y (left/right)
        action[3] *= -1   # invert roll (rotation around x-axis)
        action[5] *= -1   # invert yaw (rotation around z-axis)
    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    invert_action=False,
    aug_mode: Optional[int] = None,
    aug_mode_predictor=None,
    probe_steps: int = 5,
    probe_delta: float = 0.2,
    flow_clip: float = 20.0,
):
    """Run a single episode. If aug_mode_predictor is set: forward probe -> predict aug_mode -> reverse probe (same steps, -delta); all probe frames included in rollout."""
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    flip = np.random.rand() < cfg.flip_ratio
    probe_replay: list = []

    # Forward probe + discriminator, then reverse probe (same length, -delta); no env reset, all frames in rollout
    if aug_mode_predictor is not None and initial_state is not None:
        before_img = get_libero_image(obs)
        probe_action_fwd = get_probe_action(probe_delta)
        for _ in range(probe_steps):
            obs, _, _, _ = env.step(probe_action_fwd)
            observation, img = prepare_observation(obs, resize_size, flip=flip, cfg=cfg)
            if cfg.cat_wrist:
                frame = np.concatenate([observation["full_image"], observation["wrist_image"]], axis=1)
            else:
                frame = img
            probe_replay.append(frame)
        after_img = get_libero_image(obs)
        flow = compute_optical_flow(before_img, after_img)
        action_sign = get_probe_action_sign(1)
        device = next(aug_mode_predictor.parameters()).device
        flow_t = torch.from_numpy(flow).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
        if flow_clip > 0:
            flow_t = flow_t.clamp(-flow_clip, flow_clip)
        action_sign_t = torch.from_numpy(action_sign).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = aug_mode_predictor(flow_t, action_sign_t)
            aug_mode = int(logits.argmax(dim=1).item())
        # Reverse probe: same number of steps with -probe_delta to move back; include in rollout
        probe_action_rev = get_probe_action(-probe_delta)
        for _ in range(probe_steps):
            obs, _, _, _ = env.step(probe_action_rev)
            observation, img = prepare_observation(obs, resize_size, flip=flip, cfg=cfg)
            if cfg.cat_wrist:
                frame = np.concatenate([observation["full_image"], observation["wrist_image"]], axis=1)
            else:
                frame = img
            probe_replay.append(frame)

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match NUM_ACTIONS_CHUNK "
              f"{NUM_ACTIONS_CHUNK}. For best performance, execute the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # History conditioning is disabled
    history_frames = 0
    history_compression_rate = 1
    history_observations = None
    history_actions_deque = None

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size, flip=flip, cfg=cfg)

            if cfg.mask_cam != 'None':
                if cfg.mask_cam == 'wrist':
                    observation['wrist_image'] = np.zeros_like(observation['wrist_image'])
                elif cfg.mask_cam == 'third':
                    observation['full_image'] = np.zeros_like(observation['full_image'])
                else:
                    raise ValueError(f"Unsupported mask_cam option: {cfg.mask_cam}")

            if cfg.enable_view_fusion:
                alpha = float(observation["state"][-1].copy())
                alpha = (alpha - GRIPPER_RANGE[0]) / (GRIPPER_RANGE[1] - GRIPPER_RANGE[0])
                if cfg.view_fusion_invert:
                    alpha = 1.0 - alpha
                full_image_float = observation["full_image"].astype(np.float32)
                wrist_image_float = observation["wrist_image"].astype(np.float32)
                full_image_float *= (1.0 - alpha)
                wrist_image_float *= alpha
                observation["full_image"] = full_image_float.astype(np.uint8)
                observation["wrist_image"] = wrist_image_float.astype(np.uint8)

            if cfg.cat_wrist:
                replay_image = np.concatenate([observation['full_image'], observation['wrist_image']], axis=1)
                replay_images.append(replay_image)
            else:
                replay_images.append(img)

            if len(action_queue) == 0:
                history_images = None
                history_actions = None
                actions = get_action(
                    cfg, model, observation, task_description,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film, use_minivlm=cfg.use_minivlm,
                    history_images=history_images,
                    history_actions=history_actions,
                    aug_mode=aug_mode,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family, invert_action=invert_action)

            # History buffer disabled; do not store past observations/actions

            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    if probe_replay:
        replay_images = probe_replay + replay_images
    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    save_version=None,
    task_start_time=None,
    per_task_success_counter: Counter = None,
    q_name: str = "Normal",
    aug_mode: Optional[int] = None,
    aug_mode_predictor=None,
):
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    per_task_success_counter[task_description] = 0

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)
        success, replay_images = run_episode(
            cfg, env, task_description, model, resize_size,
            processor, action_head, proprio_projector, noisy_action_projector,
            initial_state, log_file,
            invert_action=cfg.invert_action,
            aug_mode=aug_mode,
            aug_mode_predictor=aug_mode_predictor,
            probe_steps=getattr(cfg, "probe_steps", 5),
            probe_delta=getattr(cfg, "probe_delta", 0.2),
            flow_clip=getattr(cfg, "flow_clip", 20.0),
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1
            per_task_success_counter[task_description] += 1

        if cfg.enable_quadrant_eval:
            prefix = q_name
        else:
            prefix = None
            
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description,
            log_file=log_file, save_version=save_version, task_start_time=task_start_time,
            prefix=prefix
        )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    env.close()
    del env

    if cfg.use_wandb:
        wandb.log({
            f"success_rate/{task_description}": task_success_rate,
            f"num_episodes/{task_description}": task_episodes,
        })

    return total_episodes, total_successes


# ──────────────────────────────────────────────────────────────────────────────
# Markdown table helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_quadrant_summary(quadrant_name: str, counter: Counter, total_episodes: int,
                           total_successes: int, num_trials_per_task: int):
    """Print a per-quadrant markdown table."""
    rows = [
        (task, count, num_trials_per_task, count / num_trials_per_task * 100)
        for task, count in counter.items()
    ]
    success_rate = total_successes / total_episodes * 100 if total_episodes > 0 else 0.0

    c1 = max(len("Task"),        max(len(r[0]) for r in rows), len("**Overall**"))
    c2 = max(len("Successes"),   max(len(f"{r[1]}/{r[2]}") for r in rows), len(f"{total_successes}/{total_episodes}"))
    c3 = max(len("Success Rate"),max(len(f"{r[3]:.1f}%") for r in rows), len(f"{success_rate:.1f}%"))

    sep = f"| {'-'*c1} | {'-'*c2}:| {'-'*c3}:|"
    print(f"\n### Quadrant: {quadrant_name}")
    print(f"| {'Task':<{c1}} | {'Successes':>{c2}} | {'Success Rate':>{c3}} |")
    print(sep)
    for task, count, total, rate in rows:
        print(f"| {task:<{c1}} | {f'{count}/{total}':>{c2}} | {f'{rate:.1f}%':>{c3}} |")
    print(sep)
    print(f"| {'**Overall**':<{c1}} | {f'{total_successes}/{total_episodes}':>{c2}} | {f'{success_rate:.1f}%':>{c3}} |")
    print()


def print_final_summary(all_results: dict, num_trials_per_task: int):
    """Print a combined markdown table with one column per quadrant."""
    quadrant_names = list(all_results.keys())

    # Collect all task names (preserve order from first quadrant)
    all_tasks = list(next(iter(all_results.values()))["counter"].keys())

    # Column widths
    c_task = max(len("Task"), max(len(t) for t in all_tasks), len("**Overall**"))
    # Each quadrant gets a column of format "successes/total (rate%)"
    q_col_w = max(
        max(len(q) for q in quadrant_names),
        len(f"{num_trials_per_task}/{num_trials_per_task} (100.0%)")
    )

    def q_cell(count, total):
        rate = count / total * 100 if total > 0 else 0.0
        return f"{count}/{total} ({rate:.1f}%)"

    # Header
    header = f"| {'Task':<{c_task}} |"
    for q in quadrant_names:
        header += f" {q:^{q_col_w}} |"
    sep = f"| {'-'*c_task} |" + f" {'-'*q_col_w} |" * len(quadrant_names)

    print("\n## Final Summary (All Quadrants)\n")
    print(header)
    print(sep)

    # Per-task rows
    for task in all_tasks:
        row = f"| {task:<{c_task}} |"
        for q in quadrant_names:
            count = all_results[q]["counter"].get(task, 0)
            row += f" {q_cell(count, num_trials_per_task):^{q_col_w}} |"
        print(row)

    print(sep)

    # Overall row
    overall_row = f"| {'**Overall**':<{c_task}} |"
    grand_total_ep, grand_total_suc = 0, 0
    for q in quadrant_names:
        ep  = all_results[q]["episodes"]
        suc = all_results[q]["successes"]
        grand_total_ep  += ep
        grand_total_suc += suc
        overall_row += f" {q_cell(suc, ep):^{q_col_w}} |"
    print(overall_row)

    # Grand overall across all quadrants
    grand_rate = grand_total_suc / grand_total_ep * 100 if grand_total_ep > 0 else 0.0
    print(sep)
    print(f"\n**Grand Overall**: {grand_total_suc}/{grand_total_ep} ({grand_rate:.1f}%)\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    from datetime import datetime
    task_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor, aug_mode_predictor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    if getattr(cfg, "use_aug_mode_predictor", False):
        log_message("Using aug_mode_predictor: probe at episode start, then restore env.", log_file)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Decide which quadrants to run
    quadrants_to_run = QUADRANTS if cfg.enable_quadrant_eval else [QUADRANTS[0]]

    all_results = {}  # {quadrant_name: {counter, episodes, successes}}

    for quadrant_idx, quadrant in enumerate(quadrants_to_run):
        q_name        = quadrant["name"]
        flip_ratio    = quadrant["flip_ratio"]
        invert_action = quadrant["invert_action"]
        # GT aug_mode for this quadrant (0=Normal, 1=Image Flip, 2=Action Flip, 3=Both)
        aug_mode_gt   = quadrant_idx if getattr(cfg, "use_aug_mode_condition", False) else None

        log_message(f"\n{'='*60}", log_file)
        log_message(f"Starting quadrant: {q_name}  (flip_ratio={flip_ratio}, invert_action={invert_action})", log_file)
        if aug_mode_predictor is not None:
            log_message("Using aug_mode_predictor (probe per episode).", log_file)
        elif aug_mode_gt is not None:
            log_message(f"Using GT aug_mode={aug_mode_gt} for action head.", log_file)
        log_message(f"{'='*60}", log_file)

        # Override cfg for this quadrant
        cfg.flip_ratio = flip_ratio
        cfg.invert_action = invert_action

        per_task_success_counter = Counter()
        total_episodes, total_successes = 0, 0

        for task_id in tqdm.tqdm(range(num_tasks)):
            total_episodes, total_successes = run_task(
                cfg, task_suite, task_id, model, resize_size,
                processor, action_head, proprio_projector, noisy_action_projector,
                total_episodes, total_successes, log_file,
                cfg.save_version, task_start_time, per_task_success_counter,
                q_name=q_name,
                aug_mode=None if aug_mode_predictor is not None else aug_mode_gt,
                aug_mode_predictor=aug_mode_predictor,
            )

        all_results[q_name] = {
            "counter":   per_task_success_counter,
            "episodes":  total_episodes,
            "successes": total_successes,
        }

        # Per-quadrant summary table
        print_quadrant_summary(q_name, per_task_success_counter, total_episodes,
                               total_successes, cfg.num_trials_per_task)

    # Final cross-quadrant summary (only meaningful if quadrant eval enabled)
    if cfg.enable_quadrant_eval:
        print_final_summary(all_results, cfg.num_trials_per_task)
    else:
        # Single quadrant run: use aug_mode=0 (Normal) when use_aug_mode_condition
        q = list(all_results.values())[0]
        print_quadrant_summary(
            "Normal", q["counter"], q["episodes"], q["successes"], cfg.num_trials_per_task
        )

    # wandb
    final_success_rate = sum(v["successes"] for v in all_results.values()) / \
                         max(sum(v["episodes"] for v in all_results.values()), 1)
    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_success_rate})
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()