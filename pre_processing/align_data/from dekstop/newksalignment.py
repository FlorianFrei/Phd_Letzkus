# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 19:00:22 2025

@author: Freitag
"""

#%%
import sys
sys.path.append(r"C:\Users\Freitag\Documents\GitHub\Phd_Letzkus\pre_processing\align_data")
from new_helper import process_neural_data_pipeline
basefolder = "D:/Data/3198-52/3198-52_recall_g0"
results = process_neural_data_pipeline(basefolder, 'M4_recall.csv',bin_size=0.01)
#%%


import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
import pandas as pd
from new_helper import*
basefolder ="D:/Data/3198-52/3198-52_g0"
mat_name = next(file for file in os.listdir(basefolder) if file.endswith('.mat'))

last_part = basefolder.split('/')[-1]
meta_path = f"{basefolder}/{last_part}_imec0/{last_part}_t0.imec0.ap.meta"


#%% load data


cluster_info= pd.read_csv(basefolder + str('/sorted/phy/cluster_info.tsv'),sep = '\t')
clust = np.array (np.load(basefolder + str('/sorted/phy/spike_clusters.npy')))
times = np.array(np.load(basefolder + str('/sorted/phy/spike_times.npy')))[:,0]
ITI = pd.read_csv(basefolder + str('/Meta/ttl_edge_times.csv'))
raw_BPOD = load_mat(basefolder + '/' + mat_name)
ttlsound =  pd.read_csv(basefolder + str('/Meta/soundttl.csv'))
sampling_freq = float([line.split('=')[1] for line in open(meta_path) if line.startswith('imSampRate=')][0])



#%% format ITI

ITI = ITI.iloc[:,0]

#%%
#ITI =ITI[1:].reset_index(drop=True)

proceed = check_state_alignment(ITI, raw_BPOD)
#%% wrangle BPOD


BPOD =BPOD_wrangle_claude(raw_BPOD, ITI, proceed)



BPOD = add_sound_delays(BPOD, ttlsound)



#%%

Ephys_good = Ephys_wrangle(cluster_info,clust,times,sampling_freq)
#%%

Ephys_binned = bin_Ephys(Ephys_good,bin_size=0.01)
#%%

events_with_behv = annotate_spikes_interval_join(Ephys_binned, BPOD,spike_time_col='time_bin')
#events_with_behv2 = annotate_spikes_interval_join(Ephys_good, BPOD,spike_time_col='seconds')
#%%

events_with_behv = events_with_behv[['cluster_id','time_bin','event_count','state_name','trial_number','trial_type']]
events_with_behv.to_csv(basefolder + str('/M2_naive2.csv'),index=False)

#events_with_behv = events_with_behv2[['cluster_id','seconds','state_name','trial_number','trial_type']]
#events_with_behv.to_csv(basefolder + str('/nobin.csv'))

#%% plot
trial_types = (
    BPOD.groupby("trial_number")["trial_type"]
    .unique()              # Get unique trial_type per trial_number
    .explode()             # If you want a flat Series of all values
    .reset_index(drop=True)  # Optional: remove index
)
trial_types.values
types =trial_types.values

times2 = times/sampling_freq
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def plot_psth(spike_times, stim_times, window=(-0.1, 0.5), bin_size=0.01, sigma=2):
    """
    Plot a peri-stimulus time histogram (PSTH) as smoothed spike rate.
    
    Parameters:
        spike_times (np.ndarray): 1D array of spike timestamps (in seconds).
        stim_times (np.ndarray): 1D array of stimulus timestamps (in seconds).
        window (tuple): Time window around stimulus (start, end) in seconds.
        bin_size (float): Width of histogram bins (in seconds).
        sigma (float): Gaussian smoothing kernel width (in bins).
    """
    spike_times = np.asarray(spike_times)
    stim_times = np.asarray(stim_times)
    
    bins = np.arange(window[0], window[1], bin_size)
    all_aligned_spikes = []

    for t in stim_times:
        aligned_spikes = spike_times - t
        mask = (aligned_spikes >= window[0]) & (aligned_spikes < window[1])
        all_aligned_spikes.extend(aligned_spikes[mask])

    counts, edges = np.histogram(all_aligned_spikes, bins=bins)
    rate = counts / (len(stim_times) * bin_size)  # spike rate in Hz
    rate_smoothed = gaussian_filter1d(rate, sigma=sigma)

    plt.figure(figsize=(8, 4))
    centers = (edges[:-1] + edges[1:]) / 2
    plt.plot(centers, rate_smoothed, color='black')
    plt.axvline(0, color='red', linestyle='--', label='Stimulus onset')
    plt.xlabel('Time (s) from stimulus')
    plt.ylabel('Spike rate (Hz)')
    plt.title('PSTH')
    plt.legend()
    plt.tight_layout()
    plt.show()


from scipy.ndimage import gaussian_filter1d

def plot_psth_by_type(
    spike_times,
    stim_times,
    types,
    window=(-0.1, 0.5),
    bin_size=0.01,
    sigma=2,
    subplots=False,
    zscore=False,
):
    """
    Plot peri-stimulus time histograms (PSTHs) grouped by stimulus type.

    Parameters:
        spike_times (np.ndarray): 1D array of spike timestamps (in seconds).
        stim_times (np.ndarray): 1D array of stimulus timestamps (in seconds).
        types (array-like): 1D array of stimulus types (same length as stim_times).
        window (tuple): Time window around stimulus (start, end) in seconds.
        bin_size (float): Width of histogram bins (in seconds).
        sigma (float): Gaussian smoothing kernel width (in bins).
        subplots (bool): If True, plot separate subplots per type; otherwise overlay in one plot.
        zscore (bool): If True, Z-score each PSTH trace before plotting.
    """
    spike_times = np.asarray(spike_times)
    stim_times = np.asarray(stim_times)
    types = np.asarray(types)

    bins = np.arange(window[0], window[1], bin_size)
    centers = (bins[:-1] + bins[1:]) / 2
    unique_types = np.unique(types)
    n_types = len(unique_types)
    colors = plt.cm.get_cmap('tab10', n_types)

    if subplots:
        fig, axs = plt.subplots(n_types, 1, figsize=(8, 2.5 * n_types), sharex=True)
        if n_types == 1:
            axs = [axs]
    else:
        plt.figure(figsize=(10, 5))

    for i, ttype in enumerate(unique_types):
        type_mask = types == ttype
        relevant_stim_times = stim_times[type_mask]

        all_aligned_spikes = []
        for t in relevant_stim_times:
            aligned_spikes = spike_times - t
            mask = (aligned_spikes >= window[0]) & (aligned_spikes < window[1])
            all_aligned_spikes.extend(aligned_spikes[mask])

        counts, _ = np.histogram(all_aligned_spikes, bins=bins)
        rate = counts / (len(relevant_stim_times) * bin_size)
        rate_smoothed = gaussian_filter1d(rate, sigma=sigma)

        # --- Z-score option ---
        if zscore:
            mean_val = np.mean(rate_smoothed)
            std_val = np.std(rate_smoothed)
            if std_val > 0:
                rate_smoothed = (rate_smoothed - mean_val) / std_val
            else:
                rate_smoothed = rate_smoothed * 0  # avoid NaN if no variance

        if subplots:
            ax = axs[i]
            ax.plot(centers, rate_smoothed, color=colors(i))
            ax.axvline(0, color='black', linestyle='--')
            ylabel = 'Z-scored Rate' if zscore else 'Rate (Hz)'
            ax.set_ylabel(ylabel)
            ax.set_title(f'Type {ttype}')
            if i == n_types - 1:
                ax.set_xlabel('Time (s) from stimulus')
        else:
            plt.plot(centers, rate_smoothed, label=f'Type {ttype}', color=colors(i))

    if subplots:
        plt.tight_layout()
        plt.show()
    else:
        #plt.axvline(0, color='black', linestyle='--', label='Stimulus onset')
        plt.xlabel('Time (s) from stimulus')
        ylabel = 'Z-scored Rate' if zscore else 'Spike rate (Hz)'
        plt.ylabel(ylabel)
        plt.title('PSTH by Stimulus Type')
        plt.legend()
        plt.tight_layout()
        plt.show()



    


plot_psth(times2, ttlsound,sigma=2,  window=(-0.5, 6),bin_size=0.005)
plot_psth_by_type(times2, ttlsound, types, sigma=5, window=(-1, 6),subplots=False,bin_size=0.01,zscore=True)


#%%
trial_types = (
    BPOD.groupby("trial_number")["trial_type"]
    .unique()              # Get unique trial_type per trial_number
    .explode()             # If you want a flat Series of all values
    .reset_index(drop=True)  # Optional: remove index
)
trial_types.values
types =trial_types.values

times2 = times/sampling_freq
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
def plot_psth_by_type2(
    spike_times,
    stim_times,
    types,
    clust,
    window=(-0.1, 0.5),
    bin_size=0.01,
    sigma=2,
    subplots=False,
    zscore=False,
):
    """
    Plot peri-stimulus time histograms (PSTHs) grouped by stimulus type,
    normalizing units before averaging if `zscore=True`.

    Parameters:
        spike_times (np.ndarray): 1D array of spike timestamps (in seconds).
        stim_times (np.ndarray): 1D array of stimulus timestamps (in seconds).
        types (array-like): 1D array of stimulus types (same length as stim_times).
        clust (np.ndarray): 1D array of cluster/unit IDs, same length as spike_times.
        window (tuple): Time window around stimulus (start, end) in seconds.
        bin_size (float): Width of histogram bins (in seconds).
        sigma (float): Gaussian smoothing kernel width (in bins).
        subplots (bool): If True, plot separate subplots per type; otherwise overlay in one plot.
        zscore (bool): If True, Z-score each unit's PSTH before averaging.
    """
    spike_times = np.asarray(spike_times)
    stim_times = np.asarray(stim_times)
    types = np.asarray(types)
    clust = np.asarray(clust)

    bins = np.arange(window[0], window[1], bin_size)
    centers = (bins[:-1] + bins[1:]) / 2
    unique_types = np.unique(types)
    n_types = len(unique_types)
    unique_units = np.unique(clust)

    colors = plt.cm.get_cmap('tab10', n_types)

    if subplots:
        fig, axs = plt.subplots(n_types, 1, figsize=(8, 2.5 * n_types), sharex=True)
        if n_types == 1:
            axs = [axs]
    else:
        plt.figure(figsize=(10, 5))

    for i, ttype in enumerate(unique_types):
        type_mask = types == ttype
        relevant_stim_times = stim_times[type_mask]

        unit_traces = []
        for unit in unique_units:
            unit_spike_times = spike_times[clust == unit]

            all_aligned_spikes = []
            for t in relevant_stim_times:
                aligned_spikes = unit_spike_times - t
                mask = (aligned_spikes >= window[0]) & (aligned_spikes < window[1])
                all_aligned_spikes.extend(aligned_spikes[mask])

            counts, _ = np.histogram(all_aligned_spikes, bins=bins)
            rate = counts / (len(relevant_stim_times) * bin_size)
            rate_smoothed = gaussian_filter1d(rate, sigma=sigma)

            if zscore:
                mean_val = np.mean(rate_smoothed)
                std_val = np.std(rate_smoothed)
                if std_val > 0:
                    rate_smoothed = (rate_smoothed - mean_val) / std_val
                else:
                    rate_smoothed = np.zeros_like(rate_smoothed)

            unit_traces.append(rate_smoothed)

        # Average across units
        if unit_traces:
            mean_trace = np.mean(unit_traces, axis=0)
        else:
            mean_trace = np.zeros(len(centers))

        if subplots:
            ax = axs[i]
            ax.plot(centers, mean_trace, color=colors(i))
            ax.axvline(0, color='black', linestyle='--')
            ylabel = 'Z-scored Rate' if zscore else 'Rate (Hz)'
            ax.set_ylabel(ylabel)
            ax.set_title(f'Type {ttype}')
            if i == n_types - 1:
                ax.set_xlabel('Time (s) from stimulus')
        else:
            plt.plot(centers, mean_trace, label=f'Type {ttype}', color=colors(i))

    if subplots:
        plt.tight_layout()
        plt.show()
    else:
        plt.axvline(0, color='black', linestyle='--', label='Stimulus onset')
        plt.xlabel('Time (s) from stimulus')
        ylabel = 'Z-scored Rate' if zscore else 'Spike rate (Hz)'
        plt.ylabel(ylabel)
        plt.title('PSTH by Stimulus Type (Unit-normalized)')
        plt.legend()
        plt.tight_layout()
        plt.show()

plot_psth_by_type2(times2, ttlsound, types, clust, window=(-2, 6),bin_size=0.01,sigma=4,zscore=True)
plot_psth_by_type(times2, ttlsound, types, window=(-2, 6),bin_size=0.01,sigma=4,zscore=True)


