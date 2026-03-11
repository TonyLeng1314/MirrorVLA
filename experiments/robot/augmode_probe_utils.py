"""
Shared probe utils for aug-mode: aligned with scripts/generate_augmode_probe_data.py.
Used by eval to run the same y-axis probe and compute flow for the discriminator.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def get_probe_action(delta: float = 0.2):
    """Return 7D action [x, y, z, roll, pitch, yaw, gripper]; only y-axis is delta (signed)."""
    return [0.0, float(delta), 0.0, 0.0, 0.0, 0.0, -1.0]


def get_probe_action_sign(y_sign: int = 1):
    """Action sign for the probe: y_sign in {+1, -1} for y-axis direction."""
    return np.array([0, y_sign, 0, 0, 0, 0, 0], dtype=np.float32)


def compute_optical_flow(before_rgb: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
    """
    Compute optical flow (H, W, 2) from before/after RGB frames; same as data generation.
    Requires opencv-python.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for compute_optical_flow. pip install opencv-python")
    gray_before = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_before, gray_after, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    return flow.astype(np.float32)
