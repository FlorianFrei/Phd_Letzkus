# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:18:35 2025

@author: Freitag
"""
import os
import numpy as np
import tifffile as tiff
from aicspylibczi import CziFile

def czi_to_tiff(czi_path, tiff_path):
    """Convert tiled .czi into a stitched multi-page .tiff (one page per channel)."""
    czi = CziFile(czi_path)

    # Stitch tiles into full mosaic (shape: (T, Z, C, Y, X, 1))
    mosaic = czi.read_mosaic(C=0)  # read all channels
    image = np.squeeze(mosaic)     # drop singleton dims

    # If channels are in axis 0, make them multi-page TIFF
    if image.ndim == 3:  
        # Shape: (C, Y, X)
        tiff.imwrite(tiff_path, image, imagej=True, bigtiff=False)
    elif image.ndim == 4:
        # Shape: (Z, C, Y, X) or (T, C, Y, X) -> flatten into (pages, Y, X)
        # Move channels to first axis
        pages = np.moveaxis(image, 1, 0).reshape(-1, image.shape[-2], image.shape[-1])
        tiff.imwrite(tiff_path, pages, imagej=True, bigtiff=False)
    else:
        # Simple 2D case
        tiff.imwrite(tiff_path, image, imagej=True, bigtiff=False)

def convert_all_czi(root_folder, out_folder):
    """Convert all .czi files in folder of folders to stitched TIFFs."""
    os.makedirs(out_folder, exist_ok=True)

    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith(".czi"):
                foldername = os.path.basename(dirpath)
                base_name = os.path.splitext(fname)[0]
                new_name = f"{foldername}_{base_name}.tif"

                src = os.path.join(dirpath, fname)
                dst = os.path.join(out_folder, new_name)

                # Handle name conflicts
                if os.path.exists(dst):
                    base, ext = os.path.splitext(new_name)
                    counter = 1
                    while os.path.exists(os.path.join(out_folder, f"{base}_{counter}{ext}")):
                        counter += 1
                    dst = os.path.join(out_folder, f"{base}_{counter}{ext}")

                print(f"Converting: {src} -> {dst}")
                czi_to_tiff(src, dst)

# Example usage:
# convert_all_czi("path/to/root/folder", "path/to/output/folder")


# Example usage:
convert_all_czi(r"D:\3556-17\histology", r"D:\3556-17\histology\tiffs")
