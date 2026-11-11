import subprocess
import sys
from pathlib import Path

# List of regions to process
regions = [
    "Monsanto_2020",
    "Pombal_2020",
    "SintraCascais_2020",
    "VPA_2020"
]

python_exe = r"C:\Users\gui\Documents\Tese\.conda\python.exe"

for region in regions:
    print(f"\n==============================================")
    print(f"PROCESSING REGION: {region}")
    print(f"==============================================\n")
    
    # 1. Align Rasters
    print(f"--- Step 1: Aligning Rasters for {region} ---")
    ret_align = subprocess.run([python_exe, str(Path(__file__).parent / "align_rasters.py"), region])
    if ret_align.returncode != 0:
        print(f"ERROR: Alignment failed for {region}. Skipping to next...")
        continue
        
    # 2. Estimate Region
    print(f"\n--- Step 2: Estimating Biomass for {region} ---")
    ret_est = subprocess.run([python_exe, str(Path(__file__).parent / "estimate_region.py"), region])
    if ret_est.returncode != 0:
        print(f"ERROR: Inference failed for {region}. Skipping to next...")
        continue
        
    print(f"\nSUCCESS: Finished processing {region}!")

print("\nALL REGIONS PROCESSED SUCCESSFULLY!")
