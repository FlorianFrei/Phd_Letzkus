# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 16:50:47 2025

@author: Freitag
"""

# Change working directory to the folder where this script is saved
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import glob
import shutil
import deeplabcut
import helpers 






#%% Settings
VIDEO_FOLDER = r"C:\Users\Freitag\Desktop\videos"
MIN_CERTAINTY = 0.6
MAKE_LABELED_VIDEO = True
CONFIG_PATH = r"C:\Users\Freitag\Documents\GitHub\EyeVideoAnalysis\dlc\config.yaml"


#%% Main pipeline

def main():
    # Find all videos
    videos = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
    print(f"Found {len(videos)} videos")
    
    if not videos:
        print("No videos found in the specified folder.")
        return
    
    # Step 1: Collect ROIs for all videos first
    print("\n=== ROI Selection Phase ===")
    video_rois = {}
    for vid in videos:
        print(f"Select ROI for {os.path.basename(vid)}")
        print("Instructions: Draw rectangle around pupil area, press ENTER when done, ESC to skip")
        roi = helpers.choose_roi(vid)
        if roi is None:
            print(f"No ROI selected for {os.path.basename(vid)}, will use full frame.")
        video_rois[vid] = roi
        print(f"ROI saved: {roi}\n")
    
    print("All ROIs collected. Starting video analysis...\n")
    
    # Step 2: Process each video with its ROI
    print("=== Video Analysis Phase ===")
    for vid in videos:
        base = os.path.splitext(os.path.basename(vid))[0]
        out_folder = os.path.join(VIDEO_FOLDER, base)
        os.makedirs(out_folder, exist_ok=True)

        # Move video to output folder
        dest_video = os.path.join(out_folder, os.path.basename(vid))
        if not os.path.exists(dest_video):
            shutil.move(vid, dest_video)

        print(f"Processing {base}...")
        roi = video_rois[vid]

        try:
            # Run DLC analysis
            deeplabcut.analyze_videos(CONFIG_PATH, [dest_video], 
                                    destfolder=out_folder,
                                    save_as_csv=True, 
                                    cropping=roi)

            # Optionally make labeled video
            if MAKE_LABELED_VIDEO:
                deeplabcut.create_labeled_video(CONFIG_PATH, [dest_video],
                                              destfolder=out_folder,
                                              displaycropped=True)

            # Find DLC output CSV
            csvs = glob.glob(os.path.join(out_folder, "*DLC*.csv"))
            if not csvs:
                print(f"Warning: No DLC csv found for {base}")
                continue
            csv_path = csvs[0]

            # Run enhanced pupil analysis
            df, width, height, center_x, valid, center_y_eyelid = helpers.process_raw_data(csv_path, MIN_CERTAINTY)
            
            # Interpolation and height adjustment
            F = helpers.estimate_height_from_width_pos_all(width, height, center_x, valid)
            center_y, height_adj, isInterpolated = helpers.adjust_center_height(df, F, width, height, center_x, MIN_CERTAINTY)
            
            # Detect blinks
            blinks, bl_starts, bl_stops = helpers.detect_blinks(df, width, center_x, center_y, height_adj, MIN_CERTAINTY)
            
            # Apply smoothing
            cx, cy, h = helpers.apply_medfilt(center_x, center_y, height_adj,5)
            
            # Adjust for blinks
            data, centers, diameter = helpers.adjust_for_blinks(cx, cy, h, width, blinks)
            
            # Save outputs with enhanced plotting
            helpers.save_outputs_with_blinks(out_folder, data, blinks, bl_starts, bl_stops)
            
            print(f"Completed analysis for {base}")
            
        except Exception as e:
            print(f"Error processing {base}: {str(e)}")
            continue

    print("\nAll videos processed. Check individual folders for results.")

if __name__ == "__main__":
    main()