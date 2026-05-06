# -*- coding: utf-8 -*-
"""
Batch process only sessions that already have non-empty plots
"""

import sys
from pathlib import Path

# Add your helper path
sys.path.append("C:/Users/Freitag/Documents/GitHub/Phd_Letzkus/pre_processing/align_data")
from newnewhelper import process_neural_data_pipeline


def get_folders_with_nonempty_plots_no_bpod(base_path):
    """
    Returns subfolders where:
    - raw/subfolder_x/plots exists AND is not empty
    - AND BPOD.csv does NOT exist in subfolder_x
    """
    base_path = Path(base_path)
    valid_folders = []

    for subfolder in base_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        plots_path = subfolder / "plots"
        bpod_file = subfolder / "BPOD.csv"
        
        if (
            plots_path.exists()
            and plots_path.is_dir()
            and any(p.is_file() for p in plots_path.iterdir())  # non-empty plots
            and not bpod_file.exists()  # BPOD.csv must NOT exist
        ):
            valid_folders.append(subfolder)
    print(len(valid_folders))

    return valid_folders


# ---- MAIN ----

base_root = r"H:\Data\raw"
folders = get_folders_with_nonempty_plots_no_bpod(base_root)

print(f"Found {len(folders)} folders with non-empty plots and no BPOD\n")

for folder in folders:
    print(f"\nProcessing: {folder}")
    
    try:
        process_neural_data_pipeline(
            folder,
            output_filename='bined.csv',
            bin_size=0.01
        )
    except Exception as e:
        print(f"Skipped {folder.name}: {e}")