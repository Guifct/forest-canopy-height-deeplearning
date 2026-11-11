"""
Deep Learning Architectures for Forest Canopy Height (FCH / RH98) Estimation.

This module defines all model tiers evaluated in the Master's Dissertation:
1. DummyRegressor: Non-spatial statistical baseline (predicts training set mean).
2. MinimalistCNN: 3-layer shallow convolutional baseline (Early Fusion, ~65K params).
3. SpatialCNN: 4-block deep CNN with Dilated Convolutions & Coarse 3x3 Grid Pooling (~610K params).
4. LateFusionSpatialCNN_Base: Multi-sensor branch fusion without attention (~850K params).
5. LateFusionSpatialCNN_SE: Late Fusion with Squeeze-and-Excitation (SE) Channel Attention per branch (~890K params).
6. LateFusionSpatialCNN_CenterAware: Late Fusion with SE Channel Attention AND Gaussian Center-Aware Spatial Pooling (~890K params).
7. LateFusionEnsemble: Meta-architecture aggregating multiple Late Fusion models.
"""

import math
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# =============================================================================
# Helper Sensor Grouping Utilities
# =============================================================================

def infer_sensor_group_from_name(name: str) -> str:
    """Heuristic grouping of satellite channel names into physical sensor branches."""
    u = str(name).upper()
    if any(k in u for k in ["SENTINEL1", "SENTINEL-1", "S1", "_VV", "_VH", "VV_", "VH_", "SAR"]):
        return "S1_SAR"
    s2_tokens = [f"B{i:02d}" for i in range(1, 13)] + ["B8A", "NDVI", "EVI", "NDWI", "NDMI", "NBR"]
    if any(k in u for k in ["SENTINEL2", "SENTINEL-2", "S2"] + s2_tokens):
        return "S2_OPTICAL"
    if any(k in u for k in ["ALOS", "PALSAR", "HH", "HV"]):
        return "ALOS_SAR"
    if any(k in u for k in ["DEM", "DTM", "DSM", "ELEV", "SLOPE", "ASPECT", "TPI", "TRI"]):
        return "TOPOGRAPHY"
    if any(k in u for k in ["MASK", "VALID", "AVAIL", "NODATA"]):
        return "MASKS"
    return "OTHER"


def infer_late_fusion_channel_groups(band_names: List[str]) -> OrderedDict:
    """Constructs ordered channel indices mapped to distinct sensor branches."""
    groups = OrderedDict()
    for idx, name in enumerate(band_names):
        groups.setdefault(infer_sensor_group_from_name(name), []).append(idx)
    
    preferred_order = ["S1_SAR", "S2_OPTICAL", "ALOS_SAR", "TOPOGRAPHY", "MASKS", "OTHER"]
    ordered = OrderedDict()
    for grp in preferred_order:
        if grp in groups:
            ordered[grp] = groups[grp]
    for grp in groups:
        if grp not in ordered:
            ordered[grp] = groups[grp]
    return ordered


# =============================================================================
# 1. Baseline Model: Dummy Regressor (Non-Spatial Baseline)
# =============================================================================

class DummyRegressor(nn.Module):
    """
    Non-spatial statistical baseline.
    Predicts the constant mean of the training target distribution, ignoring all inputs.
    Used to establish an empirical lower bound on RMSE and R2 (Table 4.1).
    """
    def __init__(self, constant_value: float = 0.0):
        super().__init__()
        self.register_buffer("mean_value", torch.tensor(float(constant_value), dtype=torch.float32))

    def set_mean(self, value: float):
        self.mean_value.fill_(float(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return self.mean_value.expand(batch_size)


# =============================================================================
# 2. Baseline Model: Minimalist CNN Regressor (Early Fusion, ~65K Params)
# =============================================================================

class MinimalistCNN(nn.Module):
    """
    Shallow 3-block convolutional baseline as described in Dissertation Section 4.1.2.
    - Block 1: Conv2d(in_channels, 32, 3) + ReLU + MaxPool2d(2, 2)
    - Block 2: Conv2d(32, 64, 3) + ReLU + MaxPool2d(2, 2)
    - Block 3: Conv2d(64, 64, 3) + ReLU + AdaptiveAvgPool2d(1, 1)
    - MLP Regressor: Linear(64, 32) + ReLU + Linear(32, 1)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.regressor(features).squeeze(1)


# =============================================================================
# 3. Early Fusion Architecture: SpatialCNN (~610K Params)
# =============================================================================

class SpatialCNN(nn.Module):
    """
    Deeper Early Fusion CNN with Dilated Convolutions & Coarse 3x3 Grid Pooling.
    Section 4.3 of the Dissertation.
    - 4 Conv Blocks with BatchNorm2d and GELU
    - Block 3 incorporates dilation=2 (9x9 receptive field without spatial downsampling)
    - Coarse AdaptiveAvgPool2d((3, 3)) preserving spatial layout before flattening
    - 3-layer MLP Regressor with Dropout(0.3) and Dropout(0.2)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: local feature extraction
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Block 2: channel expansion
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Block 3: Dilated convolution (dilation=2, padding=2)
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Block 4: high-level representation
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((3, 3))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        pooled = self.pool(feat)
        return self.regressor(pooled).squeeze(1)


# =============================================================================
# 4. Building Blocks: Squeeze-and-Excitation & Center-Aware Pooling
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel Attention Module (Hu et al., 2018).
    Dynamically recalibrates channel-wise feature responses by modeling channel interdependencies.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class GaussianCenterWeightedPooling(nn.Module):
    """
    Gaussian Center-Weighted Spatial Pooling (Section 4.4.2 of Dissertation).
    Assigns exponentially decaying weights from the patch center:
        w(i, j) = exp( - ((i - ic)^2 + (j - jc)^2) / (2 * sigma^2) )
    Extracts high-resolution representation precisely at the central LiDAR footprint.
    """
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        device = x.device
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        
        y = torch.arange(h, device=device).float() - cy
        x_grid = torch.arange(w, device=device).float() - cx
        yy, xx = torch.meshgrid(y, x_grid, indexing='ij')
        dist_sq = yy**2 + xx**2
        weights = torch.exp(-dist_sq / (2.0 * (self.sigma ** 2)))
        weights = weights / weights.sum()  # Normalize to sum to 1
        
        # Weighted spatial sum over (H, W) -> (B, C)
        out = (x * weights.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3))
        return out


# =============================================================================
# 5. Late Fusion Architecture: Base, +SE, and +SE+CenterAware
# =============================================================================

class SensorSpatialBranch(nn.Module):
    """
    Configurable sensor-specific spatial convolutional branch.
    Supports:
    - Standard coarse 2x2 grid pooling
    - Optional Squeeze-and-Excitation (SE) channel attention
    - Optional Center-Aware Gaussian pooling
    """
    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 32,
        use_se: bool = False,
        use_center_aware: bool = False,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.use_se = use_se
        self.use_center_aware = use_center_aware

        # Conv Block 1: 1x1 projection
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.se1 = SEBlock(branch_channels) if use_se else nn.Identity()

        # Conv Block 2: 3x3 local context
        self.conv2 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.se2 = SEBlock(branch_channels) if use_se else nn.Identity()

        # Conv Block 3: 3x3 dilated convolution (dilation=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.se3 = SEBlock(branch_channels) if use_se else nn.Identity()

        # Stream 1: Coarse 2x2 spatial grid pooling (preserves landscape context)
        self.coarse_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        
        # Stream 2: Gaussian Center-Aware footprint feature extraction
        if self.use_center_aware:
            self.center_pool = GaussianCenterWeightedPooling(sigma=sigma)
            self.out_features = branch_channels * 2 * 2 + branch_channels  # 4 * C + C
        else:
            self.center_pool = None
            self.out_features = branch_channels * 2 * 2  # 4 * C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se1(self.conv1(x))
        x = self.se2(self.conv2(x))
        x = self.se3(self.conv3(x))

        coarse = self.flatten(self.coarse_pool(x))
        if self.use_center_aware:
            center = self.center_pool(x)
            return torch.cat([coarse, center], dim=1)
        return coarse


class LateFusionSpatialCNN(nn.Module):
    """
    Unified Late Fusion Architecture covering all dissertation proposed variants:
    1. Base Late Fusion: use_se=False, use_center_aware=False (~850K params)
    2. Late Fusion + SE: use_se=True, use_center_aware=False (~890K params)
    3. Late Fusion + SE + Center-Aware (Proposed): use_se=True, use_center_aware=True (~890K params)
    """
    def __init__(
        self,
        in_channels: int,
        channel_groups: Dict[str, List[int]],
        branch_channels: int = 32,
        use_se: bool = True,
        use_center_aware: bool = True,
        center_sigma: float = 1.0,
    ):
        super().__init__()
        self.group_names = list(channel_groups.keys())
        self.group_indices = [list(channel_groups[g]) for g in self.group_names]
        self.module_keys = [re.sub(r"[^A-Za-z0-9_]", "_", g) for g in self.group_names]

        self.branches = nn.ModuleDict()
        total_features = 0

        for group_name, idxs in zip(self.group_names, self.group_indices):
            mkey = re.sub(r"[^A-Za-z0-9_]", "_", group_name)
            branch = SensorSpatialBranch(
                in_channels=len(idxs),
                branch_channels=branch_channels,
                use_se=use_se,
                use_center_aware=use_center_aware,
                sigma=center_sigma,
            )
            self.branches[mkey] = branch
            total_features += branch.out_features

        self.regressor = nn.Sequential(
            nn.Linear(total_features, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for mkey, idxs in zip(self.module_keys, self.group_indices):
            features.append(self.branches[mkey](x[:, idxs, :, :]))
        fused = torch.cat(features, dim=1)
        return self.regressor(fused).squeeze(1)


# Aliases for explicit dissertation models
class LateFusionSpatialCNN_Base(LateFusionSpatialCNN):
    """Late Fusion without Attention or Center Pooling."""
    def __init__(self, in_channels: int, channel_groups: Dict[str, List[int]], branch_channels: int = 32):
        super().__init__(in_channels, channel_groups, branch_channels=branch_channels, use_se=False, use_center_aware=False)


class LateFusionSpatialCNN_SE(LateFusionSpatialCNN):
    """Late Fusion with Squeeze-and-Excitation (SE) Channel Attention."""
    def __init__(self, in_channels: int, channel_groups: Dict[str, List[int]], branch_channels: int = 32):
        super().__init__(in_channels, channel_groups, branch_channels=branch_channels, use_se=True, use_center_aware=False)


class LateFusionSpatialCNN_CenterAware(LateFusionSpatialCNN):
    """Proposed Model: Late Fusion with SE Channel Attention and Center-Aware Pooling."""
    def __init__(self, in_channels: int, channel_groups: Dict[str, List[int]], branch_channels: int = 32, center_sigma: float = 1.0):
        super().__init__(in_channels, channel_groups, branch_channels=branch_channels, use_se=True, use_center_aware=True, center_sigma=center_sigma)


# =============================================================================
# 6. Model Tier: Late Fusion Ensemble
# =============================================================================

class LateFusionEnsemble(nn.Module):
    """
    Ensemble Architecture combining N trained Late Fusion models across folds or random seeds.
    Averages predictions during regional sliding-window inference to minimize variance.
    """
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = [model(x) for model in self.models]
        return torch.stack(predictions, dim=0).mean(dim=0)
