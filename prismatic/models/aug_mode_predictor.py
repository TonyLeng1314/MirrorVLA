"""
Aug-mode predictor (discriminator): predicts aug_mode 0/1/2/3 from probe data.

Inputs (aligned with AugModeProbeDataset):
  - flow: optical flow (B, 2, H, W); dataset gives (B, H, W, 2), permute to (B, 2, H, W) before forward.
  - action_sign: (B, 7), sign of the probe action per dimension (e.g. [0, ±1, 0, 0, 0, 0, 0] for y-only).

Output:
  - logits: (B, 4), unnormalized scores for aug_mode in {0, 1, 2, 3}.
    (0=normal, 1=image flip, 2=action invert, 3=both)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class ResidualBlock2d(nn.Module):
    """Two convs with residual: out = relu(conv2(relu(conv1(x))) + shortcut(x)). Shortcut is 1x1 when in_ch != out_ch."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(h + self.shortcut(x))


class FlowEncoder(nn.Module):
    """(B, 2, H, W) -> (B, flow_embed_dim). Stem conv + two residual blocks + pool + linear."""

    def __init__(self, flow_channels: int = 2, flow_embed_dim: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(flow_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block2 = ResidualBlock2d(32, 64)
        self.block3 = ResidualBlock2d(64, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(128, flow_embed_dim)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        x = self.stem(flow)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class AugModePredictor(nn.Module):
    """
    Skeleton aug-mode predictor. Input: flow (B, 2, H, W), action_sign (B, 7). Output: logits (B, 4).
    """

    def __init__(
        self,
        flow_channels: int = 2,
        flow_embed_dim: int = 64,
        action_sign_dim: int = 7,
        action_sign_embed_dim: int = 32,
        num_classes: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        # -------------------------------------------------------------------------
        # Flow encoder: (B, 2, H, W) -> (B, flow_embed_dim)
        # Replace with your CNN (e.g. Conv2d + pool + flatten).
        # -------------------------------------------------------------------------
        self.flow_encoder = FlowEncoder(flow_channels, flow_embed_dim)

        # -------------------------------------------------------------------------
        # Action sign embedding: (B, 7) -> (B, action_sign_embed_dim)
        # Replace with your embedding (e.g. deeper MLP or quantized embedding).
        # -------------------------------------------------------------------------
        self.action_sign_embed = nn.Linear(action_sign_dim, action_sign_embed_dim)

        # -------------------------------------------------------------------------
        # Head: (B, flow_embed_dim + action_sign_embed_dim) -> (B, num_classes)
        # Replace with your classifier (e.g. MLP with hidden layer).
        # -------------------------------------------------------------------------
        fused_dim = flow_embed_dim + action_sign_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        flow: torch.Tensor,
        action_sign: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            flow: (B, 2, H, W), optical flow. If your batch has (B, H, W, 2), do
                  flow = flow.permute(0, 3, 1, 2).
            action_sign: (B, 7), float, e.g. in [-1, 0, 1] per dimension.

        Returns:
            logits: (B, 4), unnormalized scores for aug_mode in {0, 1, 2, 3}.
        """
        # Flow path: (B, 2, H, W) -> (B, flow_embed_dim)
        flow_feat = self.flow_encoder(flow)

        # Action sign path: (B, 7) -> (B, action_sign_embed_dim)
        action_feat = self.action_sign_embed(action_sign)

        # Fuse: (B, flow_embed_dim + action_sign_embed_dim)
        fused = torch.cat([flow_feat, action_feat], dim=1)

        # Logits: (B, num_classes)
        logits = self.head(fused)
        return logits


def get_dataloader_batch_shapes():
    """
    For reference: shapes from AugModeProbeDataset + default collate.
    """
    # flow: (B, H, W, 2)  -> for model, permute to (B, 2, H, W)
    # action_sign: (B, 7)
    # aug_mode: (B,) long, label in {0,1,2,3}
    pass