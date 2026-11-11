import os
import sys

# Corrigir os avisos do GDAL no ambiente conda do Windows
os.environ["GDAL_DATA"] = os.path.join(sys.prefix, "Library", "share", "gdal")
os.environ["PROJ_LIB"]  = os.path.join(sys.prefix, "Library", "share", "proj")

import json
import re
import random
from pathlib import Path
from collections import OrderedDict
import copy

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

PATCH_SIZE = 11
PATCH_BASE_DIR = Path(r"D:\Tese_data_refinada\_example_patches_by_size")
RAW_BASE_DIR = PATCH_BASE_DIR / f"patch_{PATCH_SIZE}x{PATCH_SIZE}"
VALIDATION_DIR = Path(r"D:\Tese_data_refinada\Areas_gpkg")

# Pasta para salvar os resultados do RFE
OUTPUT_BASE_DIR = Path(__file__).parent / "RFE_results"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "rh_opt_098"

BANDS_TO_REMOVE_PER_STEP = 10
STOP_AT_N_BANDS = 9

SEED = 42
N_SPLITS = 5
N_HEIGHT_BINS = 5

BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 70
PATIENCE = 15
WEIGHT_DECAY = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============================================================
# 2. CARREGAMENTO E DATASET
# ============================================================

def extract_point_id(filename):
    m = re.search(r"point_(\d+)_", filename)
    if m is None:
        raise ValueError(f"Não consegui extrair point_id de: {filename}")
    return int(m.group(1))

def build_target_dict(gdf, target_column):
    possible_id_cols = ["point_id", "POINT_ID", "id", "ID", "fid", "FID"]
    id_col = next((c for c in possible_id_cols if c in gdf.columns), None)
    ids = gdf[id_col].astype(int).values if id_col else gdf.index.astype(int).values
    targets = gdf[target_column].values
    return {int(pid): float(t) for pid, t in zip(ids, targets) if not pd.isna(t)}

def canonical_band_name(name):
    s = str(name)
    m = re.search(r"20\d{6}_.+", s)
    if m: s = m.group(0)
    s = re.sub(r"_(A|MOSAIC)$", "", s, flags=re.IGNORECASE)
    return s

def get_valid_band_indices(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    raster_names = meta.get("raster_names", [])
    valid_indices = [
        i for i, name in enumerate(raster_names)
        if "CHM" not in name.upper() and "LIDAR" not in name.upper() and "OTHER" not in name.upper()
    ]
    return valid_indices, raster_names

def load_all_regions_tifs(raw_base_dir, validation_dir, target_column="rh_opt_098"):
    raw_base_dir = Path(raw_base_dir)
    validation_dir = Path(validation_dir)
    
    X_list, y_list, region_list, pid_list = [], [], [], []
    channels_ref, band_names_ref, canonical_band_names_ref = None, None, None

    for area_dir in sorted(raw_base_dir.iterdir()):
        if not area_dir.is_dir(): continue
        area_name = area_dir.name
        gpkg_path = validation_dir / f"{area_name}_HEIGHT_Dataset.gpkg"
        
        if not gpkg_path.exists(): continue
        metas = list(area_dir.glob("*.json"))
        if not metas: continue

        gdf = gpd.read_file(gpkg_path)
        target_dict = build_target_dict(gdf, target_column)
        valid_idx, raster_names = get_valid_band_indices(metas[0])
        valid_band_names = [raster_names[i] for i in valid_idx]
        canonical_band_names = [canonical_band_name(n) for n in valid_band_names]

        if channels_ref is None:
            channels_ref = len(valid_idx)
            band_names_ref = valid_band_names
            canonical_band_names_ref = canonical_band_names
        else:
            if canonical_band_names != canonical_band_names_ref:
                key_to_raster_index = {canonical_band_name(n): i for i, n in enumerate(raster_names)}
                valid_idx = [key_to_raster_index[key] for key in canonical_band_names_ref]

        for tif in sorted(area_dir.rglob("*.tif")):
            pid = extract_point_id(tif.name)
            if pid not in target_dict: continue
            
            with rasterio.open(tif) as src:
                patch = src.read().astype(np.float32)[valid_idx]
            
            X_list.append(patch)
            y_list.append(target_dict[pid])
            region_list.append(area_name)
            pid_list.append(pid)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    regions = np.array(region_list)
    return X, y, regions, band_names_ref

def normalize_from_train(X_train_raw, X_val_raw):
    mean = np.nanmean(X_train_raw, axis=(0, 2, 3), keepdims=True)
    std  = np.nanstd(X_train_raw,  axis=(0, 2, 3), keepdims=True)
    std  = np.where(std == 0, 1e-6, std)
    
    X_train = np.nan_to_num((X_train_raw - mean) / std, nan=0., posinf=0., neginf=0.).astype(np.float32)
    X_val   = np.nan_to_num((X_val_raw   - mean) / std, nan=0., posinf=0., neginf=0.).astype(np.float32)
    return X_train, X_val, mean, std

def base_augmentation(x):
    k = np.random.randint(0, 4)
    if k > 0: x = np.rot90(x, k, axes=(1, 2)).copy()
    if np.random.rand() > 0.5: x = np.flip(x, axis=2).copy()
    if np.random.rand() > 0.5: x = np.flip(x, axis=1).copy()
    return x

class FCHDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.transform = transform

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.transform is not None: x = self.transform(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# ============================================================
# 3. MODELO LATE FUSION CENTER-AWARE
# ============================================================

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

# ============================================================
# 4. LÓGICA DE TREINO E MÉTRICAS
# ============================================================

def regression_metrics(y_true, y_pred):
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))
    r2 = float(r2_score(y_true_np, y_pred_np)) if len(y_true_np) > 1 else float("nan")
    return {"RMSE": rmse, "R2": r2}

def _eval_loader(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bX, by in loader:
            all_preds.append(model(bX.to(device)).cpu().numpy())
            all_targets.append(by.numpy())
    return np.concatenate(all_targets), np.concatenate(all_preds)

# ============================================================
# 5. RFE LOOP
# ============================================================

def get_band_importances(model, group_indices):
    """Calcula a importância L1-norm de cada banda de input com base na 1ª layer de cada branch."""
    importances = {}
    for module_key, indices in zip(model.module_keys, group_indices):
        branch = model.branches[module_key]
        # conv1[0] is the nn.Conv2d(in_channels, branch_channels, 1)
        weights = branch.conv1[0].weight.detach().cpu().numpy() # shape (branch_channels, in_channels, 1, 1)
        # Sum of absolute weights across output channels
        l1_norm = np.sum(np.abs(weights), axis=(0, 2, 3))
        for local_idx, global_idx in enumerate(indices):
            importances[global_idx] = l1_norm[local_idx]
    return importances

def main():
    print("A carregar dados originais...")
    X_full, y_full, regions, band_names_ref = load_all_regions_tifs(RAW_BASE_DIR, VALIDATION_DIR, TARGET_COLUMN)
    
    # Criar estratificação
    # Bin heights to create classes
    y_bins = pd.qcut(y_full, q=N_HEIGHT_BINS, labels=False, duplicates='drop')
    stratify_labels = np.array([f"{r}_{b}" for r, b in zip(regions, y_bins)])
    
    # Remover classes com apenas 1 membro (KFold não lida bem com elas)
    val_counts = pd.Series(stratify_labels).value_counts()
    singletons = val_counts[val_counts < N_SPLITS].index
    for i, label in enumerate(stratify_labels):
        if label in singletons:
            stratify_labels[i] = "UNKNOWN" # Classe fallback genérica para minoritários

    active_band_indices = list(range(len(band_names_ref)))
    
    rfe_history = []
    iteration = 1
    
    summary_csv_path = OUTPUT_BASE_DIR / "rfe_summary.csv"
    
    while len(active_band_indices) >= STOP_AT_N_BANDS:
        print(f"\n{'='*50}\nIteração {iteration} | Bandas ativas: {len(active_band_indices)}\n{'='*50}")
        
        current_band_names = [band_names_ref[i] for i in active_band_indices]
        channel_groups = infer_late_fusion_channel_groups(current_band_names)
        
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        
        fold_rmses = []
        fold_r2s = []
        importances_per_fold = {i: [] for i in range(len(active_band_indices))}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, stratify_labels)):
            print(f"  Fold {fold+1}/{N_SPLITS}...")
            
            # Slice only active bands
            X_train_raw = X_full[train_idx][:, active_band_indices, :, :]
            X_val_raw = X_full[val_idx][:, active_band_indices, :, :]
            y_train = y_full[train_idx]
            y_val = y_full[val_idx]
            
            X_train, X_val, _, _ = normalize_from_train(X_train_raw, X_val_raw)
            
            train_loader = DataLoader(FCHDataset(X_train, y_train, base_augmentation), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(FCHDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
            
            model = LateFusionSpatialCNN_CenterAware(in_channels=len(active_band_indices), channel_groups=channel_groups).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            criterion = nn.MSELoss()
            
            best_val_rmse = float('inf')
            best_model_state = None
            patience_counter = 0
            
            for epoch in range(EPOCHS):
                model.train()
                for bX, by in train_loader:
                    bX, by = bX.to(device), by.to(device)
                    optimizer.zero_grad()
                    preds = model(bX)
                    loss = criterion(preds, by)
                    loss.backward()
                    optimizer.step()
                    
                y_val_true, y_val_pred = _eval_loader(model, val_loader)
                val_metrics = regression_metrics(y_val_true, y_val_pred)
                
                if val_metrics["RMSE"] < best_val_rmse:
                    best_val_rmse = val_metrics["RMSE"]
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= PATIENCE:
                    break
                    
            # Avaliar melhor modelo e guardar importância
            model.load_state_dict(best_model_state)
            y_val_true, y_val_pred = _eval_loader(model, val_loader)
            final_metrics = regression_metrics(y_val_true, y_val_pred)
            
            fold_rmses.append(final_metrics["RMSE"])
            fold_r2s.append(final_metrics["R2"])
            
            fold_importances = get_band_importances(model, model.group_indices)
            for local_idx in range(len(active_band_indices)):
                importances_per_fold[local_idx].append(fold_importances[local_idx])
                
        # Aggregate across folds
        mean_rmse = np.mean(fold_rmses)
        mean_r2 = np.mean(fold_r2s)
        
        print(f"-> Iteração {iteration} terminada. Mean Val RMSE: {mean_rmse:.2f} | Mean Val R2: {mean_r2:.3f}")
        
        avg_importances = [np.mean(importances_per_fold[i]) for i in range(len(active_band_indices))]
        
        # Guardar info
        rfe_history.append({
            "iteration": iteration,
            "n_bands": len(active_band_indices),
            "mean_rmse": mean_rmse,
            "mean_r2": mean_r2,
        })
        pd.DataFrame(rfe_history).to_csv(summary_csv_path, index=False)
        
        # Guardar as bandas desta iteração para registo
        bands_csv_path = OUTPUT_BASE_DIR / f"iter_{iteration}_bands.csv"
        pd.DataFrame({
            "band_name": current_band_names,
            "importance": avg_importances
        }).to_csv(bands_csv_path, index=False)
        
        # Eliminar piores bandas
        if len(active_band_indices) <= STOP_AT_N_BANDS:
            print("Alvo mínimo de bandas atingido. Fim do RFE.")
            break
            
        n_to_remove = min(BANDS_TO_REMOVE_PER_STEP, len(active_band_indices) - STOP_AT_N_BANDS + 1)
        if len(active_band_indices) - n_to_remove < STOP_AT_N_BANDS:
            n_to_remove = len(active_band_indices) - STOP_AT_N_BANDS
            
        if n_to_remove <= 0:
            break
            
        # Ordenar pelos índices que têm menor importância
        sorted_local_indices = np.argsort(avg_importances)
        worst_local_indices = sorted_local_indices[:n_to_remove]
        
        worst_global_indices = [active_band_indices[i] for i in worst_local_indices]
        worst_band_names = [current_band_names[i] for i in worst_local_indices]
        
        print(f"Bandas a remover: {worst_band_names}")
        
        # Remover de active_band_indices mantendo a ordem original
        active_band_indices = [idx for idx in active_band_indices if idx not in worst_global_indices]
        iteration += 1

if __name__ == "__main__":
    main()
