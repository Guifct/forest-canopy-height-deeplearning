import re
from collections import OrderedDict

import torch
import torch.nn as nn

def infer_sensor_group_from_name(name):
    u = str(name).upper()
    if any(k in u for k in ["SENTINEL1", "SENTINEL-1", "S1", "_VV", "_VH", "VV_", "VH_", "SAR"]): return "S1_SAR"
    s2_band_tokens = [f"B{i:02d}" for i in range(1, 13)] + ["B8A", "NDVI", "EVI", "NDWI", "NDMI", "NBR"]
    if any(k in u for k in ["SENTINEL2", "SENTINEL-2", "S2"] + s2_band_tokens): return "S2_OPTICAL"
    if any(k in u for k in ["DEM", "DTM", "DSM", "ELEV", "SLOPE", "ASPECT", "TPI", "TRI"]): return "TOPOGRAPHY"
    if any(k in u for k in ["MASK", "VALID", "AVAIL", "NODATA"]): return "MASKS"
    return "OTHER"

def infer_late_fusion_channel_groups(band_names):
    groups = OrderedDict()
    for idx, name in enumerate(band_names):
        groups.setdefault(infer_sensor_group_from_name(name), []).append(idx)
    
    preferred_order = ["S1_SAR", "S2_OPTICAL", "TOPOGRAPHY", "MASKS", "OTHER"]
    ordered = OrderedDict()
    for group in preferred_order:
        if group in groups: ordered[group] = groups[group]
    for group in groups:
        if group not in ordered: ordered[group] = groups[group]
    return ordered

class SensorSpatialBranchCenterAware(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int = 32, center_radius: int = 1, global_pool_size: int = 2):
        super().__init__()
        self.center_radius = int(center_radius)
        self.global_pool_size = int(global_pool_size)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((global_pool_size, global_pool_size))
        self.flatten = nn.Flatten()
        self.out_features = branch_channels * global_pool_size * global_pool_size + branch_channels

    def extract_center_features(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        cy, cx = h // 2, w // 2
        y0, y1 = max(cy - self.center_radius, 0), min(cy + self.center_radius + 1, h)
        x0, x1 = max(cx - self.center_radius, 0), min(cx + self.center_radius + 1, w)
        return x[:, :, y0:y1, x0:x1].mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.cat([self.flatten(self.global_pool(x)), self.extract_center_features(x)], dim=1)

class LateFusionSpatialCNN_CenterAware(nn.Module):
    def __init__(self, in_channels: int, channel_groups, branch_channels: int = 32):
        super().__init__()
        self.group_names = list(channel_groups.keys())
        self.group_indices = [list(channel_groups[g]) for g in self.group_names]
        
        self.branches = nn.ModuleDict()
        total_features = 0
        for group_name, idxs in zip(self.group_names, self.group_indices):
            module_key = re.sub(r"[^A-Za-z0-9_]", "_", group_name)
            branch = SensorSpatialBranchCenterAware(in_channels=len(idxs), branch_channels=branch_channels)
            self.branches[module_key] = branch
            total_features += branch.out_features
            
        self.module_keys = [re.sub(r"[^A-Za-z0-9_]", "_", g) for g in self.group_names]
        
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
        for module_key, indices in zip(self.module_keys, self.group_indices):
            features.append(self.branches[module_key](x[:, indices, :, :]))
        return self.regressor(torch.cat(features, dim=1)).squeeze(1)
