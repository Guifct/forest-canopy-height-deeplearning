import os
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import re
from late_fusion import LateFusionSpatialCNN_CenterAware, infer_late_fusion_channel_groups

import sys

# ==========================================================
# CONFIGURATIONS
# ==========================================================
REGION_NAME = sys.argv[1] if len(sys.argv) > 1 else "SerraLousa_2020"
ALIGNED_TIFF = Path(rf"C:\Users\gui\Documents\Tese\trabalho tese - Cópia\Deploy_Regional_Model\aligned_rasters\{REGION_NAME}_aligned_42bands.tif")
BANDS_CSV_PATH = Path(r"C:\Users\gui\Documents\Tese\trabalho tese - Cópia\organizado\RFE\RFE_results\iter_27_bands.csv")
MODEL_WEIGHTS = Path(r"C:\Users\gui\Documents\Tese\trabalho tese - Cópia\Deploy_Regional_Model\model_weights\final_model_deploy.pt")
MEAN_TENSOR = Path(r"C:\Users\gui\Documents\Tese\trabalho tese - Cópia\Deploy_Regional_Model\model_weights\mean.pt")
STD_TENSOR = Path(r"C:\Users\gui\Documents\Tese\trabalho tese - Cópia\Deploy_Regional_Model\model_weights\std.pt")
OUTPUT_TIFF = Path(rf"C:\Users\gui\Documents\Tese\trabalho tese - Cópia\Deploy_Regional_Model\predicted_outputs\{REGION_NAME}_predicted_height.tif")

PATCH_SIZE = 11
PAD = PATCH_SIZE // 2  # 5
BLOCK_SIZE = 256
BATCH_SIZE = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================================
# HELPERS
# ==========================================================
def canonical_band_name(name):
    s = str(name)
    m = re.search(r"20\d{6}_.+", s)
    if m: s = m.group(0)
    s = re.sub(r"_(A|MOSAIC)$", "", s, flags=re.IGNORECASE)
    return s

# 1. Load canonical band targets
df_bands = pd.read_csv(BANDS_CSV_PATH)
canonical_targets = [canonical_band_name(name) for name in df_bands['band_name']]

# 2. Get band names from the aligned TIFF
with rasterio.open(ALIGNED_TIFF) as src:
    ordered_band_names = list(src.descriptions)

# 3. Load Normalization Stats
mean_val = torch.load(MEAN_TENSOR, map_location='cpu').numpy()
std_val = torch.load(STD_TENSOR, map_location='cpu').numpy()

# 4. Load Model
channel_groups = infer_late_fusion_channel_groups(ordered_band_names)
model = LateFusionSpatialCNN_CenterAware(in_channels=len(ordered_band_names), channel_groups=channel_groups)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location='cpu'))
model.to(device)
model.eval()

# 5. Process Image in Blocks
with rasterio.open(ALIGNED_TIFF) as ref_src:
    meta = ref_src.meta.copy()
    full_width = ref_src.width
    full_height = ref_src.height
    nodata_val = ref_src.nodata

meta.update(count=1, dtype='float32', nodata=-9999.0, compress='deflate', predictor=2, bigtiff='yes')

print(f"Region dimensions: {full_width}x{full_height}")

with rasterio.open(OUTPUT_TIFF, 'w', **meta) as dst:
    for row_start in range(0, full_height, BLOCK_SIZE):
        for col_start in range(0, full_width, BLOCK_SIZE):
            
            # Unpadded block bounds
            row_end = min(row_start + BLOCK_SIZE, full_height)
            col_end = min(col_start + BLOCK_SIZE, full_width)
            
            out_h = row_end - row_start
            out_w = col_end - col_start
            
            # Padded bounds for reading
            read_row_start = max(0, row_start - PAD)
            read_row_end = min(full_height, row_end + PAD)
            read_col_start = max(0, col_start - PAD)
            read_col_end = min(full_width, col_end + PAD)
            
            read_h = read_row_end - read_row_start
            read_w = read_col_end - read_col_start
            
            # If the padded window is smaller than 11x11, we can't extract a patch.
            if read_h < PATCH_SIZE or read_w < PATCH_SIZE:
                dst.write(np.full((1, out_h, out_w), -9999.0, dtype=np.float32), window=Window(col_start, row_start, out_w, out_h))
                continue
                
            # Read padded block for all bands from the single file
            read_window = Window(read_col_start, read_row_start, read_w, read_h)
            with rasterio.open(ALIGNED_TIFF) as src:
                block_data = src.read(window=read_window) # Shape [42, read_h, read_w]
            
            nodata_mask_pad = (block_data[0] == nodata_val)
            
            # Apply SAR dB conversion only on valid pixels
            for c, bname in enumerate(ordered_band_names):
                if 'S1' in bname or 'ALOS2' in bname:
                    valid_mask = ~nodata_mask_pad
                    block_data[c][valid_mask] = 10 * np.log10(np.clip(block_data[c][valid_mask], 1e-6, None))
            
            # Extract the precise nodata mask for the UNPADDED block center
            local_row_offset = row_start - read_row_start
            local_col_offset = col_start - read_col_start
            center_nodata_mask = nodata_mask_pad[local_row_offset:local_row_offset+out_h, local_col_offset:local_col_offset+out_w]
            
            # Normalize
            block_data = np.where(block_data == nodata_val, np.nan, block_data)
            mean_val_3d = mean_val.reshape(42, 1, 1)
            std_val_3d = std_val.reshape(42, 1, 1)
            block_data = np.nan_to_num((block_data - mean_val_3d) / std_val_3d, nan=0., posinf=0., neginf=0.).astype(np.float32)
            
            # NOTE: handle borders by padding the tensor if necessary
            padded_block_data = np.zeros((42, out_h + 2*PAD, out_w + 2*PAD), dtype=np.float32)
            paste_row = PAD - (row_start - read_row_start)
            paste_col = PAD - (col_start - read_col_start)
            padded_block_data[:, paste_row:paste_row+read_h, paste_col:paste_col+read_w] = block_data
            
            block_tensor = torch.tensor(padded_block_data).unsqueeze(0)
            patches = torch.nn.functional.unfold(block_tensor, kernel_size=PATCH_SIZE)
            patches = patches.squeeze(0).transpose(0, 1) # [out_h * out_w, C*11*11]
            patches = patches.view(-1, 42, PATCH_SIZE, PATCH_SIZE) # [L, C, 11, 11]
            
            # Predict in batches
            preds = np.zeros((patches.shape[0],), dtype=np.float32)
            
            with torch.no_grad():
                for i in range(0, patches.shape[0], BATCH_SIZE):
                    batch = patches[i:i+BATCH_SIZE].to(device)
                    preds[i:i+BATCH_SIZE] = model(batch).cpu().numpy()
            
            preds_2d = preds.reshape(out_h, out_w)
            
            # Apply NoData Mask
            preds_2d[center_nodata_mask] = -9999.0
            
            # Write to output TIFF
            dst.write(preds_2d.reshape(1, out_h, out_w), window=Window(col_start, row_start, out_w, out_h))
            
        print(f"Processed row {row_start}/{full_height}")

print("Inference completed successfully!")
