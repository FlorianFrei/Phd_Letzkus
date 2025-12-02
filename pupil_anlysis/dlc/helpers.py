# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 17:13:29 2025

@author: Freitag
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, SmoothBivariateSpline
from scipy.signal import medfilt
import cv2


def process_raw_data(file_path, MIN_CERTAINTY):
    """
    Processes raw data from a DLC CSV output file to calculate pupil width, height, and center coordinates.
    """
    df = pd.read_csv(file_path, header=[1, 2])
    del df["bodyparts"]
    
    width = df[("rightpupil", "x")] - df[("leftpupil", "x")]
    height = df[("ventralpupil", "y")] - df[("dorsalpupil", "y")]
    center_x = df[("leftpupil", "x")] + 0.5 * width
    valid = (df.loc[:, (slice(None), "likelihood")] > MIN_CERTAINTY).all(axis=1)
    center_y_eyelid = (df[("righteyelid"), "y"] + df[("lefteyelid"), "y"]) / 2
    
    # Check if height values are valid
    height_valid = (df[("ventralpupil", "likelihood")] > MIN_CERTAINTY) & (df[("dorsalpupil", "likelihood")] > MIN_CERTAINTY)
    
    return df, width, height, center_x, valid, center_y_eyelid

def estimate_height_from_width_pos_all(width, height, center_x, valid):
    """
    Performs grid-based linear interpolation and smooth bivariate spline fitting.
    """
    if not np.any(valid) or np.sum(valid) < 10:  # Need minimum points for interpolation
        print("Warning: Insufficient valid data points for interpolation")
        return None
        
    points = np.column_stack((center_x[valid], width[valid]))
    values = height[valid]
    
    x_grid, y_grid = np.meshgrid(
        np.linspace(center_x[valid].min(), center_x[valid].max(), 100),
        np.linspace(width[valid].min(), width[valid].max(), 100)
    )
    z_grid = griddata(points, values, (x_grid, y_grid), method="linear")
    
    # Remove NaN values for spline fitting
    ind = ~np.isnan(z_grid)
    x_notnan, y_notnan, z_notnan = x_grid[ind], y_grid[ind], z_grid[ind]
    
    if len(x_notnan) < 10:  # Need minimum points for spline
        print("Warning: Insufficient interpolated points for spline fitting")
        return None
        
    F = SmoothBivariateSpline(x_notnan, y_notnan, z_notnan, kx=3, ky=3)
    return F

def adjust_center_height(df, F, width, height, center_x, MIN_CERTAINTY):
    """
    Adjusts height using interpolation for invalid points and calculates center_y.
    """
    if F is None:
        center_y = df[("ventralpupil", "y")] - 0.5 * height
        isInterpolated_indices = np.zeros(len(df), dtype=bool)
        return center_y, height, isInterpolated_indices
    
    MAX_DIST_PUPIL_LID = 5  # in pixels
    
    only_bottom_valid = (df[("dorsalpupil", "likelihood")] < MIN_CERTAINTY) & (df[("ventralpupil", "likelihood")] > MIN_CERTAINTY)
    only_top_valid = (df[("dorsalpupil", "likelihood")] > MIN_CERTAINTY) & (df[("ventralpupil", "likelihood")] < MIN_CERTAINTY)
    lid_near_pupil_top = (df[("dorsalpupil", "y")] - df[("dorsaleyelid", "y")]) < MAX_DIST_PUPIL_LID
    lid_near_pupil_bottom = df[("ventraleyelid", "y")] - df[("ventralpupil", "y")] < MAX_DIST_PUPIL_LID
    width_valid = (df[("leftpupil", "likelihood")] > MIN_CERTAINTY) & (df[("rightpupil", "likelihood")] > MIN_CERTAINTY)
    
    ind = width_valid & (only_bottom_valid | only_top_valid | lid_near_pupil_top | lid_near_pupil_bottom)
    height[ind] = F.ev(center_x[ind], width[ind])  # Adjust height at invalid points
    
    center_y = df[("ventralpupil", "y")] - 0.5 * height
    center_y[width_valid & only_top_valid] = df.loc[width_valid & only_top_valid, ("dorsalpupil", "y")] + 0.5 * height[width_valid & only_top_valid]
    
    isInterpolated_indices = ind
    
    return center_y, height, isInterpolated_indices

def detect_blinks(df, width, center_x, center_y, height, MIN_CERTAINTY, print_out=True):
    """
    Detects blinks based on various conditions from the original measure_pupil.py
    """
    LID_MIN_STD = 7
    MIN_DIST_LID_CENTER = 0.5  # in number of pupil heights   
    SURROUNDING_BLINKS = 5  # in frames
    
    # Initialize blinks array
    blinks = np.logical_or(df[("leftpupil", "likelihood")] < MIN_CERTAINTY, 
                          df[("rightpupil", "likelihood")] < MIN_CERTAINTY)
    
    # Distance between upper and lower eye lids is very small -> add to blinks 
    lid_distance = df[("ventraleyelid", "y")] - df[("dorsaleyelid", "y")]
    lid_valid = (df[("ventraleyelid", "likelihood")] > MIN_CERTAINTY) & (df[("dorsaleyelid", "likelihood")] > MIN_CERTAINTY)
    
    if np.any(lid_valid):
        lid_mean = np.mean(lid_distance[lid_valid])
        lid_std = np.std(lid_distance[lid_valid])
        blinks[lid_valid] = np.logical_or(blinks[lid_valid], 
                                         lid_distance[lid_valid] < (lid_mean - LID_MIN_STD * lid_std))
    
    # If top and bottom of pupil uncertain -> add to blinks
    blinks = np.logical_or(blinks, 
                          np.logical_and(df[("dorsalpupil", "likelihood")] < MIN_CERTAINTY, 
                                       df[("ventralpupil", "likelihood")] < MIN_CERTAINTY))
    
    # Get minimum distance of center to lid (top or bottom); if distance too small -> add to blinks
    dist_top_lid_to_center = center_y - df[("dorsaleyelid", "y")]
    dist_top_lid_to_center[df[("dorsaleyelid", "likelihood")] < MIN_CERTAINTY] = np.nan
    dist_bottom_lid_to_center = df[("ventraleyelid", "y")] - center_y
    dist_bottom_lid_to_center[df[("ventraleyelid", "likelihood")] < MIN_CERTAINTY] = np.nan
    
    dist_lid_center = np.nanmin(np.column_stack((dist_top_lid_to_center, dist_bottom_lid_to_center)), axis=1)
    blinks = np.logical_or(blinks, dist_lid_center < (MIN_DIST_LID_CENTER * height))
    
    # Include n frames before and after detected blinks
    tmp = blinks.copy()
    for t in range(1, SURROUNDING_BLINKS + 1):
        blinks = np.logical_or(blinks, np.concatenate((np.zeros(t, dtype=bool), tmp[:-t])))
        blinks = np.logical_or(blinks, np.concatenate((tmp[t:], np.zeros(t, dtype=bool))))
    
    # Convert to numpy array if it's a pandas Series
    if hasattr(blinks, 'to_numpy'):
        blinks_array = blinks.to_numpy()
    else:
        blinks_array = blinks
    
    if print_out:
        d = np.diff(blinks_array.astype(int))
        starts = np.where(d == 1)[0] + 1
        stops = np.where(d == -1)[0]
        
        # Handle edge cases
        if blinks_array[0]:
            starts = np.concatenate(([0], starts))
        if blinks_array[-1]:
            stops = np.concatenate((stops, [len(blinks_array)-1]))
            
        print(f"Detected {len(starts)} blink episodes:")
        for i, (start, stop) in enumerate(zip(starts, stops)):
            print(f"  Episode {i+1}: frames {start} - {stop}")
    
    return blinks_array, starts, stops

def adjust_for_blinks(center_x, center_y_adj, height_adj, width, blinks):
    """
    Adjusts data for blinks by setting values to NaN during blink periods.
    """
    center_x_adj = np.where(blinks, np.nan, center_x)
    center_y_adj = np.where(blinks, np.nan, center_y_adj)
    diameter = np.where(blinks, np.nan, height_adj)
    
    data = np.column_stack((center_x_adj, center_y_adj, diameter))
    return data, [center_x_adj, center_y_adj], diameter

def apply_medfilt(center_x, center_y, height, span):
    """
    Applies median filtering for smoothing.
    """
    return medfilt(center_x, span), medfilt(center_y, span), medfilt(height, span)

def save_outputs_with_blinks(out_folder, data, blinks, bl_starts, bl_stops):
    """
    Enhanced save function with blink highlighting in plots.
    """
    os.makedirs(out_folder, exist_ok=True)
    
    # Save data files
    np.save(os.path.join(out_folder, "eye.xyPos.npy"), data[:, :2])
    np.save(os.path.join(out_folder, "eye.diameter.npy"), data[:, 2])
    np.save(os.path.join(out_folder, "eye.blinks.npy"), blinks)
    
    # Create enhanced plots with blink highlighting
    names = ['Center X', 'Center Y', 'Diameter']
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    for i, ax in enumerate(axs):
        # Skip plotting if all values are NaN
        if np.all(np.isnan(data[:, i])):
            ax.text(0.5, 0.5, f'{names[i]} data not available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes)
            ax.set_ylabel(names[i])
            continue
        
        # Calculate limits to avoid NaN issues
        valid_data = data[~np.isnan(data[:, i]), i]
        if len(valid_data) > 0:
            min_val, max_val = np.nanmin(valid_data), np.nanmax(valid_data)
            range_val = max_val - min_val
            y_limits = [min_val - 0.1 * range_val, max_val + 0.1 * range_val]
            ax.set_ylim(y_limits)
            
            # Plot data
            ax.plot(data[:, i], 'k-', linewidth=1, label=names[i])
            ax.set_ylabel(names[i])
            
            # Highlight blinks with gray overlay
            if len(bl_starts) > 0 and len(bl_stops) > 0:
                for start, stop in zip(bl_starts, bl_stops):
                    ax.fill_betweenx(y_limits, start, stop, color='gray', alpha=0.3, label='Blinks' if i == 0 else "")
    
    # Add legend to first subplot
    if len(bl_starts) > 0:
        axs[0].legend()
    
    axs[-1].set_xlabel('Frames')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "xyPos_diameter_blinks.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Outputs saved to: {out_folder}")


def choose_roi(video_path):
    """
    Interactive ROI selection for a video.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Jump to the middle frame
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {middle_frame} from {video_path}")

    roi = cv2.selectROI(f"ROI for {os.path.basename(video_path)}", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):  # nothing selected
        return None

    x, y, w, h = roi
    return [int(x), int(x + w), int(y), int(y + h)]  # DLC expects [xmin, xmax, ymin, ymax]
