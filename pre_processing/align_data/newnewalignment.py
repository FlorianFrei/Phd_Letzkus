
#%%
import sys
from pathlib import Path
sys.path.append("C:/Users/Freitag/Documents/GitHub/Phd_Letzkus/pre_processing/align_data")
from newnewhelper import process_neural_data_pipeline
basefolder = r"I:\Data\raw\8186\8186_naive"
basepath = Path(basefolder)
results = process_neural_data_pipeline(basepath, 'bined.csv',bin_size=0.01,time_window=None)
#%%


import os
from pathlib import Path

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
import pandas as pd
from newnewhelper import *
basefolder =r"I:\Data\raw\8186\8186_naive"
basepath = Path(basefolder)
mat_name = next(file for file in os.listdir(basepath) if file.endswith('.mat'))


#%% load data
spike_path = basepath / 'spike_times' / 'spike_times.npy'
spike_times = pd.DataFrame(np.load(spike_path,allow_pickle=True))



ITI = pd.read_csv(basepath / 'Meta' / 'State_changes.csv')
raw_BPOD = load_mat(basepath / mat_name)
ttlsound =  pd.read_csv(basepath /  'Meta' / 'Audio.csv')
ttlsound = ttlsound[ttlsound['edge_type']=='rising'].iloc[:,0]




#%% format ITI

ITI = ITI.iloc[:,0]

#%%
#ITI =ITI[1:].reset_index(drop=True)

proceed = check_state_alignment(ITI, raw_BPOD)
#%% wrangle BPOD


BPOD =BPOD_wrangle_claude(raw_BPOD, ITI, proceed)



BPOD = add_sound_delays(BPOD, ttlsound)

#BPOD.to_csv(basefolder + str('/BPOD.csv') )

#%%

Ephys_good = Ephys_wrangle(spike_times)
#%%

Ephys_binned = bin_Ephys(Ephys_good,bin_size=0.01)
#%%

events_with_behv = annotate_spikes_interval_join(Ephys_binned, BPOD,spike_time_col='time_bin')
#events_with_behv2 = annotate_spikes_interval_join(Ephys_good, BPOD,spike_time_col='seconds')
#%%

events_with_behv = events_with_behv[['cluster_id','time_bin','event_count','state_name','trial_number','trial_type']]
events_with_behv.to_csv(basefolder + str('/17_naive_allUnits.csv'))

#events_with_behv = events_with_behv2[['cluster_id','seconds','state_name','trial_number','trial_type']]
#events_with_behv.to_csv(basefolder + str('/nobin.csv'))

len(events_with_behv['cluster_id'].unique())

#%% plot
trial_types = (
    BPOD.groupby("trial_number")["trial_type"]
    .unique()              # Get unique trial_type per trial_number
    .explode()             # If you want a flat Series of all values
    .reset_index(drop=True)  # Optional: remove index
)
mapping = {
    '1': 'Upsweep',
    '2': 'Downsweep',
    '3': 'Opto_Upsweep', # Fixed typo: 'Upwsweep' -> 'Upsweep'
    '4': 'Opto_Downsweep'
}

# 3. Apply the mapping
# .map() looks up the value in the dictionary; if not found, it keeps the original
trial_types_mapped = trial_types.astype(str).map(mapping).fillna(trial_types)

# 4. Convert to a numpy array (if you specifically need .values)
types = trial_types_mapped.values

times2 = spike_times[['time_s']]
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



import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def plot_psth_by_type(spike_times, stim_times, types, window=(-0.1, 0.5), bin_size=0.01, sigma=2, subplots=False):
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

        if subplots:
            ax = axs[i]
            ax.plot(centers, rate_smoothed, color=colors(i))
            ax.axvline(0, color='black', linestyle='--')
            ax.set_ylabel('Rate (Hz)')
            ax.set_title(f'Type {ttype}')
            if i == n_types - 1:
                ax.set_xlabel('Time (s) from stimulus')
        else:
            plt.plot(centers, rate_smoothed, label=f'Type {ttype}', color=colors(i))

    if subplots:
        plt.tight_layout()
        plt.show()
    else:
        plt.axvline(0, color='black', linestyle='--', label='Stimulus onset')
        plt.xlabel('Time (s) from stimulus')
        plt.ylabel('Spike rate (Hz)')
        plt.title('PSTH by Stimulus Type')
        plt.legend()
        plt.tight_layout()
        plt.show()


    

plot_psth(times2, ttlsound,sigma=8,  window=(-2, 6),bin_size=0.01)
plot_psth_by_type(times2, ttlsound, types, sigma=8, window=(-2, 2),subplots=False,bin_size=0.01)
