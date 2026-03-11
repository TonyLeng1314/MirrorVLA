"""
AugModeProbeDataset: loads .npz samples produced by generate_augmode_probe_data.py (optical flow version).

Each sample contains: flow (H,W,2), action_sign (7,), aug_mode (0/1/2/3),
and optionally before/after main and wrist images (for debugging or image-based input).

If root_dir contains index.json (multi-suite), all task suites are merged into one flat list
so that DataLoader(shuffle=True) yields samples from all suites in random order.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class AugModeProbeDataset(Dataset):
    """
    Loads optical-flow aug-mode probe data. Single-suite: root_dir has manifest.json + sample_*.npz.
    Multi-suite: root_dir has index.json and subdirs <task_suite_name>/ with manifest + samples;
    all suites are merged into one flat index for random reading across suites.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        include_images: bool = False,
        flow_clip: Optional[float] = None,
    ):
        """
        Args:
            root_dir: Single-suite: dir with manifest.json and sample_*.npz.
                      Multi-suite: dir with index.json and subdirs per task_suite_name.
            include_images: If True, return before_main, after_main, etc. (for image-based input or viz).
            flow_clip: If set, clip flow to [-flow_clip, flow_clip] for stable training.
        """
        self.root_dir = Path(root_dir)
        self.include_images = include_images
        self.flow_clip = flow_clip

        index_path = self.root_dir / "index.json"
        if index_path.exists():
            # Multi-suite: flatten all (suite_dir, sample_fname) into one list
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            self._samples: List[Tuple[Path, str]] = []
            for suite_name in index["task_suites"]:
                suite_dir = self.root_dir / suite_name
                manifest_path = suite_dir / "manifest.json"
                if not manifest_path.exists():
                    raise FileNotFoundError(f"manifest.json not found in {suite_dir}")
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for fname in manifest["samples"]:
                    self._samples.append((suite_dir, fname))
            self.manifest = {"num_suites": len(index["task_suites"]), "total_samples": len(self._samples)}
        else:
            # Single-suite
            manifest_path = self.root_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"manifest.json or index.json not found in {self.root_dir}")
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
            self._samples = [(self.root_dir, fname) for fname in self.manifest["samples"]]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        suite_dir, fname = self._samples[idx]
        path = suite_dir / fname
        data = np.load(path, allow_pickle=True)

        flow = data["flow"].astype(np.float32)
        if self.flow_clip is not None:
            flow = np.clip(flow, -self.flow_clip, self.flow_clip)

        action_sign = data["action_sign"].astype(np.float32)
        aug_mode = int(data["aug_mode"])

        out: Dict[str, Any] = {
            "flow": torch.from_numpy(flow),
            "action_sign": torch.from_numpy(action_sign),
            "aug_mode": torch.tensor(aug_mode, dtype=torch.long),
        }
        if self.include_images:
            out["before_main"] = torch.from_numpy(data["before_main"])
            out["after_main"] = torch.from_numpy(data["after_main"])
            out["before_wrist"] = torch.from_numpy(data["before_wrist"])
            out["after_wrist"] = torch.from_numpy(data["after_wrist"])
        # Optional: for debugging or grouping
        out["task_id"] = int(data["task_id"])
        out["init_state_idx"] = int(data["init_state_idx"])
        out["task_suite"] = self._samples[idx][0].name  # subdir name (suite name when multi-suite)

        return out
