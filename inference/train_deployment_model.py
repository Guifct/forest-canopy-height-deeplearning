import os
import sys
import json
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import our extracted model module
from late_fusion import LateFusionSpatialCNN_CenterAware, infer_late_fusion_channel_groups

# Configure environments for GDAL if needed
os.environ["GDAL_DATA"] = os.path.join(sys.prefix, "Library", "share", "gdal")
os.environ["PROJ_LIB"]  = os.path.join(sys.prefix, "Library", "share", "proj")

# ============================================================
# CONFIGURATIONS
# ============================================================
PATCH_SIZE = 11
RAW_BASE_DIR = Path(r"D:\Tese_data_refinada\_example_patches_by_size\patch_11x11")
VALIDATION_DIR = Path(r"D:\Tese_data_refinada\Areas_gpkg")
BANDS_CSV_PATH = Path(r"C:\Users\gui\Documents\Tese\trabalho tese - Cópia\organizado\RFE\RFE_results\iter_27_bands.csv")

TARGET_COLUMN = "rh_opt_098"

SEED = 42
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 60 # Fixed number of epochs for full deployment retraining
WEIGHT_DECAY = 0.0

SCRIPT_DIR = Path(os.getcwd())
OUTPUT_DIR = SCRIPT_DIR

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
print(f"Device: {device}")

# ============================================================
# DATA LOADING
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

def load_target_canonical_bands():
    if not BANDS_CSV_PATH.exists():
        raise FileNotFoundError(f"Ficheiro de bandas não encontrado: {BANDS_CSV_PATH}")
    df = pd.read_csv(BANDS_CSV_PATH)
    canonical_targets = set([canonical_band_name(name) for name in df['band_name']])
    print(f"--> Lidas {len(canonical_targets)} bandas canónicas alvo.")
    return canonical_targets

def get_valid_band_indices(metadata_path, canonical_targets):
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    raster_names = meta.get("raster_names", [])
    valid_indices = []
    for i, name in enumerate(raster_names):
        if canonical_band_name(name) in canonical_targets:
            valid_indices.append(i)
    return valid_indices, raster_names

def load_all_regions_tifs(raw_base_dir, validation_dir, canonical_targets, target_column="rh_opt_098"):
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
        valid_idx, raster_names = get_valid_band_indices(metas[0], canonical_targets)
        valid_band_names = [raster_names[i] for i in valid_idx]
        canonical_band_names = [canonical_band_name(n) for n in valid_band_names]

        if channels_ref is None:
            channels_ref = len(valid_idx)
            band_names_ref = valid_band_names
            canonical_band_names_ref = canonical_band_names
        else:
            if canonical_band_names != canonical_band_names_ref:
                key_to_raster_index = {canonical_band_name(n): i for i, n in enumerate(raster_names)}
                valid_idx = [key_to_raster_index[key] for key in canonical_band_names_ref if key in key_to_raster_index]

        for tif in sorted(area_dir.rglob("*.tif")):
            pid = extract_point_id(tif.name)
            if pid not in target_dict: continue
            
            with rasterio.open(tif) as src:
                patch = src.read().astype(np.float32)[valid_idx]
            
            # Apply SAR dB conversion
            for c, bname in enumerate(valid_band_names):
                if 'S1' in bname or 'ALOS2' in bname:
                    patch[c] = 10 * np.log10(np.clip(patch[c], 1e-6, None))
            
            X_list.append(patch)
            y_list.append(target_dict[pid])
            region_list.append(area_name)
            pid_list.append(pid)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    regions = np.array(region_list)
    return X, y, regions, pid_list, band_names_ref

canonical_targets = load_target_canonical_bands()
print("A carregar dados...")
X_full, y_full, regions, pids, band_names = load_all_regions_tifs(RAW_BASE_DIR, VALIDATION_DIR, canonical_targets, TARGET_COLUMN)
print(f"X_full shape: {X_full.shape} | y_full shape: {y_full.shape}")

# Calculate Normalization over 100% data
print("A calcular estatísticas globais...")
mean = np.nanmean(X_full, axis=(0, 2, 3), keepdims=True)
std  = np.nanstd(X_full,  axis=(0, 2, 3), keepdims=True)
std  = np.where(std == 0, 1e-6, std)

# Save normalization tensors to disk
torch.save(torch.tensor(mean), OUTPUT_DIR / "mean.pt")
torch.save(torch.tensor(std), OUTPUT_DIR / "std.pt")
print("Estatísticas globais guardadas em mean.pt e std.pt")

# Apply Normalization
X_train = np.nan_to_num((X_full - mean) / std, nan=0., posinf=0., neginf=0.).astype(np.float32)

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

train_loader = DataLoader(FCHDataset(X_train, y_full, base_augmentation), batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# TRAINING 100% DATA
# ============================================================
print("A iniciar o treino no dataset completo (100%)...")
channel_groups = infer_late_fusion_channel_groups(band_names)
model = LateFusionSpatialCNN_CenterAware(in_channels=len(band_names), channel_groups=channel_groups).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for bX, by in train_loader:
        bX, by = bX.to(device), by.to(device)
        optimizer.zero_grad()
        preds = model(bX)
        loss = criterion(preds, by)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * bX.size(0)
    
    epoch_loss /= len(train_loader.dataset)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss (MSE): {epoch_loss:.4f}")

model_path = OUTPUT_DIR / "final_model_deploy.pt"
torch.save(model.state_dict(), model_path)
print(f"Treino concluído. Modelo guardado em {model_path}")
