# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 14:55:24 2025

@author: Freitag
"""


#%%

import sys
sys.path.append(r"C:\Users\Freitag\Documents\GitHub\Phd_Letzkus\align_data")
import os
import numpy as np
import pandas as pd
import new_helper as nh
basefolder ="D:/3556-17/3556-17_naive_g0"
mat_name = next(file for file in os.listdir(basefolder) if file.endswith('.mat'))

last_part = basefolder.split('/')[-1]
meta_path = f"{basefolder}/{last_part}_imec0/{last_part}_t0.imec0.ap.meta"


#%% load data


cluster_info= pd.read_csv(basefolder + str('/sorted/phy/cluster_info.tsv'),sep = '\t')
clust = np.array (np.load(basefolder + str('/sorted/phy/spike_clusters.npy')))
times = np.array(np.load(basefolder + str('/sorted/phy/spike_times.npy')))[:,0]
ITI = pd.read_csv(basefolder + str('/Meta/ttl_edge_times.csv'))
raw_BPOD = nh.load_mat(basefolder + '/' + mat_name)
ttlsound =  pd.read_csv(basefolder + str('/Meta/soundttl.csv'))
sampling_freq = float([line.split('=')[1] for line in open(meta_path) if line.startswith('imSampRate=')][0])



#%% format ITI

ITI = ITI.iloc[:,0]

#%%
#ITI =ITI[1:].reset_index(drop=True)

proceed = nh.check_state_alignment(ITI, raw_BPOD)
#%% wrangle BPOD


BPOD =nh.BPOD_wrangle_claude(raw_BPOD, ITI, proceed)



BPOD = nh.add_sound_delays(BPOD, ttlsound)



#%%

Ephys_good = nh.Ephys_wrangle(cluster_info,clust,times,sampling_freq)
#%%


def annotate_spikes_interval_join_binless(ephys_df, bpod_df, spike_time_col='seconds',
                                 start_col='continuous_start', end_col='continuous_time'):
    """
    More efficient approach using pandas interval-based logic.
    Better performance for large datasets.
    """
    # Sort both dataframes by time for efficiency
    ephys_sorted = ephys_df.sort_values(spike_time_col).reset_index(drop=True)
    bpod_sorted = bpod_df.sort_values(start_col).reset_index(drop=True)
    
    # Use merge_asof twice to find the boundaries
    # First, find the state that starts before or at each spike time
    merged = pd.merge_asof(
        ephys_sorted,
        bpod_sorted,
        left_on=spike_time_col,
        right_on=start_col,
        direction='backward'
    )
    
    # Keep only spikes that fall within the interval
    events_with_behv = merged[
        (merged[spike_time_col] >= merged[start_col]) & 
        (merged[spike_time_col] <= merged[end_col])
    ]
    
    print(f"Matched {len(events_with_behv)} out of {len(ephys_df)} spikes")
    return events_with_behv
#%%

#events_with_behv = annotate_spikes_interval_join(Ephys_binned, BPOD,spike_time_col='time_bin')
events_with_behv2 = nh.annotate_spikes_interval_join(Ephys_good, BPOD,spike_time_col='seconds')
#%%

events_with_behv = events_with_behv[['cluster_id','seconds','state_name','trial_number','trial_type']]
events_with_behv2.to_csv(basefolder + str('/nobin.csv'))

#events_with_behv = events_with_behv2[['cluster_id','seconds','state_name','trial_number','trial_type']]
#events_with_behv.to_csv(basefolder + str('/nobin.csv'))