import os
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
import pandas as pd
import re
from rasterio.enums import Resampling

import sys

# ==========================================================
# CONFIGURATIONS
# ==========================================================
REGION_NAME = sys.argv[1] if len(sys.argv) > 1 else "SerraLousa_2020"
DATA_DIR = Path(rf"D:\Tese_data_refinada\DATA\{REGION_NAME}")
BANDS_CSV_PATH = Path(r"C:\Users\gui\Documents\Tese\trabalho tese - Cópia\organizado\RFE\RFE_results\iter_27_bands.csv")
OUTPUT_TIFF = Path(rf"C:\Users\gui\Documents\Tese\trabalho tese - Cópia\Deploy_Regional_Model\aligned_rasters\{REGION_NAME}_aligned_42bands.tif")

def canonical_band_name(name):
    s = str(name)
    m = re.search(r"20\d{6}_.+", s)
    if m: s = m.group(0)
    s = re.sub(r"_(A|MOSAIC)$", "", s, flags=re.IGNORECASE)
    return s

# 1. Load canonical band targets
df_bands = pd.read_csv(BANDS_CSV_PATH)
canonical_targets = [canonical_band_name(name) for name in df_bands['band_name']]
print(f"Loaded {len(canonical_targets)} target bands.")

# 2. Match with actual .tif files
all_tifs = list(DATA_DIR.glob("*.tif"))
tif_map = {}
for tif in all_tifs:
    c_name = canonical_band_name(tif.stem)
    tif_map[c_name] = tif

ordered_tifs = []
for c_target in canonical_targets:
    if c_target in tif_map:
        ordered_tifs.append(tif_map[c_target])

assert len(ordered_tifs) == len(canonical_targets), "Missing bands!"

# 3. Find Reference Raster (We'll use an S2 band as reference)
ref_tif = None
for t in ordered_tifs:
    if "S2_" in t.stem:
        ref_tif = t
        break
if ref_tif is None:
    ref_tif = ordered_tifs[0]

print(f"Using {ref_tif.name} as spatial reference...")

with rasterio.open(ref_tif) as src:
    ref_meta = src.meta.copy()
    ref_transform = src.transform
    ref_crs = src.crs
    ref_width = src.width
    ref_height = src.height

# Update meta for the 42-band stacked output
ref_meta.update(
    count=len(ordered_tifs),
    dtype='float32',
    nodata=-9999.0,
    compress='deflate',
    predictor=2,
    bigtiff='yes'
)

print(f"Target Grid: {ref_width}x{ref_height}, CRS: {ref_crs}")

import numpy as np

# 4. Reproject and Stack
with rasterio.open(OUTPUT_TIFF, 'w', **ref_meta) as dst:
    for i, tif_path in enumerate(ordered_tifs):
        print(f"Aligning band {i+1}/{len(ordered_tifs)}: {tif_path.name}")
        with rasterio.open(tif_path) as src:
            source_array = src.read(1)
            # Create a destination array
            dest_array = np.full((ref_height, ref_width), -9999.0, dtype=np.float32)
            
            reproject(
                source=source_array,
                destination=dest_array,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_nodata=-9999.0,
                resampling=Resampling.bilinear
            )
            
            # Mask out nodata and any surviving NaNs
            dest_array[np.isnan(dest_array)] = -9999.0
            dest_array[dest_array == -9999.0] = -9999.0
            dst.write(dest_array, i + 1)
            dst.set_band_description(i + 1, tif_path.stem)

print(f"Successfully generated {OUTPUT_TIFF}")
