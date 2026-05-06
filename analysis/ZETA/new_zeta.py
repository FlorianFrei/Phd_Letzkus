# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:39:21 2026

@author: Freitag
"""

import numpy as np
import pandas as pd
import zetapy as zp

#%% -------------------- LOAD SPIKES --------------------

spikes = np.load(
    r"H:\Data\raw\8186_naive\spike_times\spike_times.npy",
    allow_pickle=True
)

spikes_df = pd.DataFrame(spikes)

# keep only good + mua units
spikes_df = spikes_df[spikes_df["label"].isin(["mua", "good"])]

#%% -------------------- LOAD BPOD --------------------

BPOD = pd.read_csv(r"H:\Data\raw\8186_naive\BPOD.csv")

# keep only relevant columns (if they exist)
BPOD = BPOD[["trial_number", "trial_type"]].drop_duplicates()

# opto trials
opto_trials = BPOD[BPOD["trial_type"].isin([3, 4])]

#%% -------------------- LOAD TIMESTAMPS --------------------

timestamps = pd.read_csv(r"H:\Data\raw\8186_naive\Meta\Audio.csv")

# keep only rising edges
timestamps = timestamps[timestamps["edge_type"] == "rising"]

# ensure alignment with BPOD (ASSUMES same order / length)
timestamps = timestamps.reset_index(drop=True)
BPOD = BPOD.reset_index(drop=True)

#%% -------------------- ZETA ALL TRIALS --------------------

u_list = []
p_list = []

event_times = timestamps["timestamps"].values

for unit in spikes_df["unit_index"].unique():
    
    spikes_unit = spikes_df[spikes_df["unit_index"] == unit]
    spike_times = spikes_unit["time_s"].values
    
    dblZetaP = zp.zetatest(
        spike_times,
        event_times,
        dblUseMaxDur=1,
        boolPlot=False,
        intResampNum=1000
    )[0]
    
    u_list.append(unit)
    p_list.append(dblZetaP)

zeta = pd.DataFrame({
    "unit": u_list,
    "p_val": p_list
})

#%% -------------------- ZETA OPTO ONLY --------------------

def main():
# select timestamps where BPOD trial is opto
    opto_mask = BPOD["trial_type"].isin([3, 4])
    
    timestamps_opto = timestamps.loc[opto_mask.values].copy()
    
    # subtract 100 ms
    timestamps_opto["timestamps"] = timestamps_opto["timestamps"] - 0.1
    
    event_times_opto = timestamps_opto["timestamps"].values
    
    u_list = []
    p_list = []
    
    for unit in spikes_df["unit_index"].unique():
        print(unit)
        spikes_unit = spikes_df[spikes_df["unit_index"] == unit]
        spike_times = spikes_unit["time_s"].values
        
        dblZetaP = zp.zetatest(
            spike_times,
            event_times_opto,
            dblUseMaxDur=0.1,
            boolPlot=False,
            intResampNum=500,
        )[0]
        
        u_list.append(unit)
        p_list.append(dblZetaP)
    
    zeta_opto = pd.DataFrame({
        "unit": u_list,
        "p_val": p_list
    })
    

if __name__ == "__main__":
    main()
    
    
    
#%%

import numpy as np
import pandas as pd
import zetapy as zp
from pathlib import Path


# --------------------------------------------------
# Core ZETA function for a single folder
# --------------------------------------------------

def run_zeta_for_folder(folder_path):

    folder_path = Path(folder_path)

    spike_file = folder_path / "spike_times" / "spike_times.npy"
    bpod_file = folder_path / "BPOD.csv"
    audio_file = folder_path / "Meta" / "Audio.csv"
    meta_folder = folder_path / "Meta"
    output_file = meta_folder / "Zeta_audio.csv"

    # --- safety checks
    if not spike_file.exists():
        return
    if not bpod_file.exists():
        print(f"Missing BPOD: {folder_path}")
        return
    if not audio_file.exists():
        print(f"Missing Audio: {folder_path}")
        return
    if output_file.exists():
        print('csv exists, skipping')
        return

    print(f"Processing: {folder_path}")

    # -------------------- LOAD --------------------

    spikes = np.load(spike_file, allow_pickle=True)
    spikes_df = pd.DataFrame(spikes)
    spikes_df = spikes_df[spikes_df["label"].isin(["mua", "good"])]

    BPOD = pd.read_csv(bpod_file)
    BPOD = BPOD[["trial_number", "trial_type"]].drop_duplicates()

    timestamps = pd.read_csv(audio_file)
    timestamps = timestamps[timestamps["edge_type"] == "rising"]

    # -------------------- ALIGNMENT CHECK --------------------

    timestamps = timestamps.reset_index(drop=True)
    BPOD = BPOD.reset_index(drop=True)

    if len(timestamps) != len(BPOD):
        raise ValueError(
            f"Alignment mismatch in {folder_path}:\n"
            f"timestamps = {len(timestamps)}, BPOD = {len(BPOD)}"
        )
        

    # -------------------- ZETA ALL --------------------

    event_times = timestamps["timestamps"].values

    zeta = compute_zeta(
        spikes_df,
        event_times,
        max_dur=1,
        resamp=1000
    )

    # -------------------- ZETA OPTO --------------------

    opto_mask = BPOD["trial_type"].isin([3, 4])
    timestamps_opto = timestamps.loc[opto_mask.values].copy()

    timestamps_opto["timestamps"] -= 0.1
    event_times_opto = timestamps_opto["timestamps"].values

    zeta_opto = compute_zeta(
        spikes_df,
        event_times_opto,
        max_dur=0.1,
        resamp=500
    )

    # -------------------- SAVE --------------------

    zeta.to_csv(meta_folder / "Zeta_audio.csv", index=False)
    zeta_opto.to_csv(meta_folder / "Zeta_opto.csv", index=False)


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def compute_zeta(spikes_df, event_times, max_dur, resamp):

    u_list = []
    p_list = []

    for unit in spikes_df["unit_index"].unique():

        spike_times = spikes_df.loc[
            spikes_df["unit_index"] == unit, "time_s"
        ].values

        p = zp.zetatest(
            spike_times,
            event_times,
            dblUseMaxDur=max_dur,
            boolPlot=False,
            intResampNum=resamp
        )[0]

        u_list.append(unit)
        p_list.append(p)

    return pd.DataFrame({
        "unit": u_list,
        "p_val": p_list
    })


# --------------------------------------------------
# Traverse all folders
# --------------------------------------------------

def run_zeta_for_root(root_dir):

    root_dir = Path(root_dir)


    for path in root_dir.rglob("spike_times"):
        try:
            run_zeta_for_folder(path.parent)
        except:
            continue


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    root = r"I:\Data\raw"
    run_zeta_for_root(root)