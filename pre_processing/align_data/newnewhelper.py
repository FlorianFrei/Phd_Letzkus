''' 
variety of helper functions to transform raw BPOD and raw KS data to an aligned Dataframe
@author: FlorianFreitag
'''
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import matlab

import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"Function '{func.__name__}' ran in {duration:.3f} seconds.")
        return result
    return wrapper

def load_mat(filename): #TODO NOT MY CODE NORA GAVE TO ME
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d
    
    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list, dtype='object')
        else:
            return ndarray

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)

def check_state_alignment_old(ITI, raw_BPOD):
    """
    Validates timing alignment between ITI state change times and BPOD trial data
    by comparing intervals between consecutive state changes.
    
    Args:
        ITI: List of all state change times from external recording
        raw_BPOD: Raw BPOD data structure containing trial information
    
    Returns:
        bool: True if timing alignment is acceptable, False otherwise
    """
    print(f"ITI state changes: {len(ITI)}")
    print(f"BPOD trials: {len(raw_BPOD['SessionData']['TrialStartTimestamp'])}")
    
    # Get first trial's state change times from BPOD
    first_trial_states = raw_BPOD['SessionData']['RawEvents']['Trial'][0]['States']
    
    # Extract and sort all state timestamps from first trial
    bpod_state_times = []
    for state_name, times in first_trial_states.items():
        if times is not None:
            if len(times) == 2:  # [start, end] format
                bpod_state_times.extend([times[0], times[1]])
            else:  # Single timestamp
                bpod_state_times.append(times[0])
    
    bpod_state_times.sort()
    
    # Compare first few state intervals between ITI and BPOD
    tolerance = 0.01  # 10ms tolerance
    alignment_checks = min(3, len(bpod_state_times) - 1, len(ITI) - 1)
    
    for i in range(alignment_checks):
        iti_interval = ITI[i + 1] - ITI[i]
        bpod_interval = bpod_state_times[i + 1] - bpod_state_times[i]
        time_diff = abs(iti_interval - bpod_interval)
        
        print(f'State interval {i + 1}: ITI={iti_interval:.4f}s, BPOD={bpod_interval:.4f}s, diff={time_diff:.4f}s')
        
        if time_diff < tolerance:
            print(f'✓ State interval {i + 1} alignment acceptable')
            return True
    
    print('✗ No acceptable timing alignment found')
    return False


def check_state_alignment(ITI, raw_BPOD, n_checks=10, tolerance=0.02):
    """
    Validates alignment by comparing ITI pulse intervals against BPOD state durations.

    One ITI pulse fires at each state start; dead-time gaps generate no pulse.
    For states that are last in their trial, the following ITI interval includes
    dead-time and will exceed the BPOD state duration — this is expected.
    Only intra-trial mismatches count as alignment failures.
    """
    session_data = raw_BPOD['SessionData']
    iti_arr = np.asarray(ITI.values if hasattr(ITI, 'values') else ITI, dtype=float)

    print(f"ITI state changes: {len(iti_arr)}")
    print(f"BPOD trials:       {len(session_data['TrialStartTimestamp'])}")

    # Flatten valid state durations across all trials in recording order.
    # Track which states are last in their trial (cross-trial boundary follows).
    bpod_durations = []
    is_last_in_trial = []

    for trial_data in session_data['RawEvents']['Trial']:
        trial_states = []
        for times in trial_data['States'].values():
            if times is None:
                continue
            times_arr = np.atleast_1d(times).astype(float)
            if len(times_arr) == 2 and not np.any(np.isnan(times_arr)):
                trial_states.append((float(times_arr[0]), float(times_arr[1]) - float(times_arr[0])))

        trial_states.sort(key=lambda x: x[0])
        durations = [d for _, d in trial_states]
        bpod_durations.extend(durations)
        if durations:
            is_last_in_trial.extend([False] * (len(durations) - 1) + [True])

    iti_intervals = np.diff(iti_arr)
    n = min(n_checks, len(bpod_durations) - 1, len(iti_intervals))

    print(f"\nFirst {n} intervals (* = cross-trial boundary, dead-time is expected):")
    print(f"{'#':>3}  {'':5}  {'ITI interval':>12}  {'BPOD duration':>13}  {'diff':>8}")
    print("-" * 55)

    passed_intra = 0
    total_intra  = 0

    for i in range(n):
        at_boundary = is_last_in_trial[i]
        diff = iti_intervals[i] - bpod_durations[i]  # positive = dead-time absorbed

        if at_boundary:
            ok    = diff >= -tolerance  # only fail if ITI is shorter than BPOD duration
            label = f"{'✓' if ok else '✗'}*"
        else:
            ok    = abs(diff) < tolerance
            label = f"{'✓' if ok else '✗'} "
            total_intra += 1
            if ok:
                passed_intra += 1

        print(f"{i+1:>3}  {label:<5}  {iti_intervals[i]:>12.4f}  {bpod_durations[i]:>13.4f}  {diff:>+8.4f}s")

    print("-" * 55)
    if total_intra > 0:
        print(f"Intra-trial pairs passing: {passed_intra}/{total_intra}")

    result = (total_intra == 0) or (passed_intra / total_intra >= 0.5)
    print("✓ Alignment acceptable\n" if result else "✗ Alignment failed — review output above\n")
    return result

def BPOD_wrangle_claude_old(raw_BPOD, ITI, proceed):
    """
    Takes raw MATLAB BPOD data and transforms it into a DataFrame of all trials.
    
    Args:
        raw_BPOD: Raw MATLAB BPOD data structure
        ITI: Inter-trial interval data
        proceed: Boolean flag to proceed with processing
    
    Returns:
        pandas.DataFrame: Processed BPOD data with continuous timestamps
    """
    if not proceed:
        print("Stop, this is not gonna work")
        return None
    
    session_data = raw_BPOD['SessionData']
    
    # Calculate aligned start and end times
    time_offset = ITI[0] - session_data['TrialStartTimestamp'][0]
    trial_start_times = session_data['TrialStartTimestamp'] + time_offset
    trial_end_times = session_data['TrialEndTimestamp'] + time_offset
    
    # Calculate dead time between trials
    dead_times = []
    for trial_idx in range(len(trial_start_times) - 1):
        current_trial_end = trial_end_times[trial_idx]
        next_trial_start = trial_start_times[trial_idx + 1]
        dead_time = next_trial_start - current_trial_end
        dead_times.append(dead_time)
    
    # Last trial has no dead time
    dead_times.append(0)
    
    # Process each trial's state data
    trial_dataframes = []
    
    for trial_idx, trial_data in enumerate(session_data['RawEvents']['Trial']):
        # Convert states dictionary to DataFrame
        states_df = pd.DataFrame.from_dict(trial_data['States']).transpose()
        states_df['state_name'] = states_df.index
        states_df['trial_number'] = trial_idx
        
        # Add dead time as an additional state
        last_state_end = states_df[1].max()
        dead_time_end = last_state_end + dead_times[trial_idx]
        dead_time_row = pd.DataFrame({
            0: [last_state_end],
            1: [dead_time_end], 
            'state_name': ['dead_time'],
            'trial_number': [trial_idx]
        })
        
        states_df = pd.concat([states_df, dead_time_row], ignore_index=True)
        trial_dataframes.append(states_df)
    
    # Combine all trials into single DataFrame
    combined_df = pd.concat(trial_dataframes, ignore_index=True)
    combined_df = combined_df.dropna().reset_index(drop=True)
    
    # Remove the complex continuous time calculation since we'll use ITI directly
    
    # Check ITI length and trim if needed
    expected_iti_length = len(combined_df) + 1  # states + 1 (extra timestamp at end)
    print(f"Expected ITI length: {expected_iti_length}, Actual ITI length: {len(ITI)}")
    
    if len(ITI) != expected_iti_length:
        print(f"Warning: ITI length mismatch. Expected {expected_iti_length}, got {len(ITI)}")
        if len(ITI) > len(combined_df):
            print("Trimming ITI to match number of states")
            ITI_trimmed = ITI[:len(combined_df)]
        elif len(ITI) == len(combined_df):
            ITI_trimmed = ITI[:len(combined_df)]
        else:
            print("Error: ITI too short for number of states")
            return None
    else:
        ITI_trimmed = ITI[:len(combined_df)]
    
    # Use ITI timestamps directly for continuous timing
    combined_df['state_duration'] = combined_df[1] - combined_df[0]
    combined_df['continuous_start'] = ITI_trimmed
    combined_df['continuous_time'] = combined_df['continuous_start'] + combined_df['state_duration']
    
    # Add trial types
    trial_types = []
    for trial_num in combined_df['trial_number']:
        trial_type = session_data['TrialTypes'][trial_num]
        trial_types.append(trial_type)
    combined_df['trial_type'] = trial_types
    
    return combined_df

def BPOD_wrangle_claude(raw_BPOD, ITI, proceed):
    """
    Transforms raw BPOD data into a time-aligned DataFrame.

    ITI has one pulse per real state start; dead-time intervals generate no pulse.
    Dead-time rows are reconstructed after ITI mapping rather than being
    assigned ITI timestamps, which would silently shift all subsequent rows.
    """
    if not proceed:
        print("Stop, this is not gonna work")
        return None

    session_data = raw_BPOD['SessionData']
    iti_arr  = np.asarray(ITI.values if hasattr(ITI, 'values') else ITI, dtype=float)
    n_trials = len(session_data['TrialStartTimestamp'])

    # --- 1. Real-states-only DataFrame (no dead_time rows yet) ---
    trial_dataframes = []
    for trial_idx, trial_data in enumerate(session_data['RawEvents']['Trial']):
        states_df = pd.DataFrame.from_dict(trial_data['States']).transpose()
        states_df['state_name']   = states_df.index
        states_df['trial_number'] = trial_idx
        trial_dataframes.append(states_df)

    combined_df = pd.concat(trial_dataframes, ignore_index=True)
    combined_df = combined_df.dropna().reset_index(drop=True)   # drop unvisited states
    combined_df['state_duration'] = combined_df[1] - combined_df[0]

    n_real_states = len(combined_df)
    n_iti         = len(iti_arr)

    print(f"Real BPOD states: {n_real_states}")
    print(f"ITI pulses:       {n_iti}")

    if n_iti < n_real_states:
        print(f"Error: {n_iti} ITI pulses for {n_real_states} states — cannot proceed")
        return None
    if n_iti > n_real_states:
        print(f"Warning: {n_iti - n_real_states} extra ITI pulses — trimming")

    # --- 2. 1:1 mapping: each real state gets its ITI start timestamp ---
    combined_df['continuous_start'] = iti_arr[:n_real_states]
    combined_df['continuous_time']  = combined_df['continuous_start'] + combined_df['state_duration']

    # --- 3. Reconstruct dead_time rows from ITI gaps between trials ---
    # dead_time[N] = ITI[first_pulse_of_trial_N+1] - continuous_time[last_state_of_trial_N]
    trial_state_counts = (
        combined_df.groupby('trial_number', sort=True)
                   .size()
                   .reindex(range(n_trials), fill_value=0)
    )
    first_iti_idx = np.concatenate([[0], np.cumsum(trial_state_counts.values)[:-1]])

    dead_time_rows = []
    for trial_idx in range(n_trials):
        trial_mask = combined_df['trial_number'] == trial_idx
        if not trial_mask.any():
            continue

        dead_start = combined_df.loc[trial_mask, 'continuous_time'].max()

        if trial_idx + 1 < n_trials:
            next_idx = int(first_iti_idx[trial_idx + 1])
            dead_end = iti_arr[next_idx] if next_idx < n_real_states else dead_start
        else:
            dead_end = dead_start  # no dead_time after the final trial

        dead_duration = dead_end - dead_start
        if dead_duration > 1e-4:
            dead_time_rows.append({
                0: np.nan, 1: np.nan,
                'state_name':      'dead_time',
                'trial_number':    int(trial_idx),
                'state_duration':  dead_duration,
                'continuous_start': dead_start,
                'continuous_time':  dead_end,
            })

    if dead_time_rows:
        combined_df = pd.concat([combined_df, pd.DataFrame(dead_time_rows)], ignore_index=True)

    # --- 4. Temporal sort and trial-type annotation ---
    combined_df = combined_df.sort_values('continuous_start').reset_index(drop=True)
    combined_df['trial_type'] = combined_df['trial_number'].map(
        lambda n: session_data['TrialTypes'][int(n)]
    )

    return combined_df

def add_sound_delays(BPOD, ttlsound):
    """
    Simple function to add sound_delay states and adjust sound timing
    
    Args:
        BPOD: DataFrame with BPOD states
        ttlsound: Series/DataFrame with actual sound timing
    
    Returns:
        Modified BPOD DataFrame with sound_delay states inserted
    """
    sound_types = ['Downsweep', 'Opto_Downsweep', 'Opto_Upsweep', 'Upsweep']
    result_rows = []
    ttl_index = 0
    
    for _, row in BPOD.iterrows():
        # Add the current row
        result_rows.append(row.copy())
        
        # If this is a sound state, insert delay and modify timing
        if row['state_name'] in sound_types:
            if ttl_index >= len(ttlsound):
                raise ValueError(f"Not enough TTL values for sound state: {row['type']}")
            
            bpod_start = row['continuous_start']
            actual_sound_time = ttlsound.iloc[ttl_index] if hasattr(ttlsound, 'iloc') else ttlsound[ttl_index]
            delay_duration = actual_sound_time - bpod_start
            
            if delay_duration <= -0.001:
                raise ValueError(f"Sound plays before BPOD signal: delay = {delay_duration}")
            
            # Create sound_delay row (insert before the sound)
            delay_row = row.copy()
            delay_row['state_name'] = 'sound_delay'
            delay_row['continuous_start'] = bpod_start
            delay_row['continuous_time'] = actual_sound_time
            delay_row['state_duration'] = delay_duration
            
            # Insert delay row before the current sound row
            result_rows.insert(-1, delay_row)
            
            # Modify the sound row to start when sound actually plays
            sound_row = result_rows[-1]  # The sound row we just added
            original_sound_end = sound_row['continuous_time']
            sound_row['continuous_start'] = actual_sound_time
            # Keep the same end time to maintain alignment with following states
            sound_row['state_duration'] = original_sound_end - actual_sound_time
            
            ttl_index += 1
    
    return pd.DataFrame(result_rows).reset_index(drop=True)



    return BPOD


def Ephys_wrangle(spike_times):
    #takes raw KS vectos and turns them into a Dataframe  
    #selects only clusters that have the PHY good 
    
    good_cluster = spike_times.query('label == "mua" |label == "good"')[['unit_index','time_s']]
    Ephys_good = good_cluster.rename(columns={'unit_index': 'cluster_id', 'time_s': 'seconds'})
    return Ephys_good




def bin_Ephys(Ephys_good,bin_size=0.01):
    # raw Kilosort has only data if a spike occoured, this transforms it into evenlz space dintervals and counts how manz spikes occoured 

    max_seconds = Ephys_good['seconds'].max()
    bin_edges = np.arange(0, max_seconds + bin_size, bin_size)

    # Create a new column with bin labels representing the end of each bin
    Ephys_good['time_bin'] = pd.cut(Ephys_good['seconds'], bins=bin_edges, right=False, labels=bin_edges[1:])

    # Group by cluster_id and time_bin, then count the number of events
    result = Ephys_good.groupby(['cluster_id', 'time_bin'],observed=False).size().reset_index(name='event_count')

    # If you want to fill in missing bins with zeros
    all_combinations = pd.MultiIndex.from_product([Ephys_good['cluster_id'].unique(), bin_edges[1:]], names=['cluster_id', 'time_bin'])
    result = result.set_index(['cluster_id', 'time_bin']).reindex(all_combinations, fill_value=0).reset_index()

    return(result)


def annotate_spikes_interval_join(ephys_df, bpod_df, spike_time_col='seconds',
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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from newnewhelper import (load_mat, check_state_alignment, BPOD_wrangle_claude, 
                        add_sound_delays, Ephys_wrangle, bin_Ephys, 
                        annotate_spikes_interval_join, timeit)

def plot_psth(spike_times, stim_times, save_path=None, window=(-2, 6), bin_size=0.01, sigma=8):
    """Plots and saves a PSTH as smoothed spike rate."""
    spike_times = np.asarray(spike_times).flatten()
    stim_times = np.asarray(stim_times).flatten()
    
    bins = np.arange(window[0], window[1], bin_size)
    all_aligned_spikes = []

    for t in stim_times:
        aligned_spikes = spike_times - t
        mask = (aligned_spikes >= window[0]) & (aligned_spikes < window[1])
        all_aligned_spikes.extend(aligned_spikes[mask])

    if len(stim_times) == 0:
        print("  Warning: No stimulus times provided for plotting.")
        return

    counts, edges = np.histogram(all_aligned_spikes, bins=bins)
    rate = counts / (len(stim_times) * bin_size)
    rate_smoothed = gaussian_filter1d(rate, sigma=sigma)

    plt.figure(figsize=(8, 4))
    centers = (edges[:-1] + edges[1:]) / 2
    plt.plot(centers, rate_smoothed, color='black')
    plt.axvline(0, color='red', linestyle='--', label='Stimulus onset')
    plt.xlabel('Time (s) from stimulus')
    plt.ylabel('Spike rate (Hz)')
    plt.title('Overall PSTH')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_psth_by_type(spike_times, stim_times, types, save_path=None, window=(-2, 6), bin_size=0.01, sigma=8):
    """Plots and saves PSTHs grouped by stimulus type."""
    spike_times = np.asarray(spike_times).flatten()
    stim_times = np.asarray(stim_times).flatten()
    types = np.asarray(types)

    bins = np.arange(window[0], window[1], bin_size)
    centers = (bins[:-1] + bins[1:]) / 2
    unique_types = np.unique(types)
    colors = plt.cm.get_cmap('tab10', len(unique_types))

    plt.figure(figsize=(10, 5))

    for i, ttype in enumerate(unique_types):
        type_mask = (types == ttype)
        # Match stim times to types (assumes 1:1 mapping)
        relevant_stim_times = stim_times[:len(types)][type_mask]

        all_aligned_spikes = []
        for t in relevant_stim_times:
            aligned_spikes = spike_times - t
            mask = (aligned_spikes >= window[0]) & (aligned_spikes < window[1])
            all_aligned_spikes.extend(aligned_spikes[mask])

        if len(relevant_stim_times) > 0:
            counts, _ = np.histogram(all_aligned_spikes, bins=bins)
            rate = counts / (len(relevant_stim_times) * bin_size)
            rate_smoothed = gaussian_filter1d(rate, sigma=sigma)
            plt.plot(centers, rate_smoothed, label=f'Type: {ttype}', color=colors(i))

    plt.axvline(0, color='black', linestyle='--', label='Stimulus onset')
    plt.xlabel('Time (s) from stimulus')
    plt.ylabel('Spike rate (Hz)')
    plt.title('PSTH by Stimulus Type')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

@timeit
def process_neural_data_pipeline(basepath, output_filename='processed_data.csv', bin_size=0.01, time_window=None):
    basepath = Path(basepath)
    print(f"Starting neural data processing pipeline for: {basepath}")
    
    # --- Step 1 & 2: Loading ---
    try:
        mat_name = next(file for file in os.listdir(basepath) if file.endswith('.mat'))
        spike_path = basepath / 'spike_times' / 'spike_times.npy'
        
        spike_times_raw = pd.DataFrame(np.load(spike_path, allow_pickle=True))
        ITI_df = pd.read_csv(basepath / 'Meta' / 'State_changes.csv')
        raw_BPOD = load_mat(basepath / mat_name)
        
        ttlsound_df = pd.read_csv(basepath / 'Meta' / 'Audio.csv')
        ttlsound = ttlsound_df[ttlsound_df['edge_type'] == 'rising'].iloc[:, 0].values
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")
    
    # --- Step 3-6: Wrangling ---
    ITI_series = ITI_df.iloc[:, 0]
    proceed = check_state_alignment(ITI_series, raw_BPOD)
    if not proceed: raise RuntimeError("Alignment failed.")
        
    BPOD = BPOD_wrangle_claude(raw_BPOD, ITI_series, proceed)
    BPOD = add_sound_delays(BPOD, ttlsound)
    
    # --- Step 7-9: Ephys & Annotation ---
    Ephys_good = Ephys_wrangle(spike_times_raw)
    Ephys_binned = bin_Ephys(Ephys_good, bin_size=bin_size)
    
    # Binned annotation
    events_with_behv = annotate_spikes_interval_join(Ephys_binned, BPOD, spike_time_col='time_bin')
    # Non-binned (raw) annotation
    events_with_behv_noBin = annotate_spikes_interval_join(Ephys_good, BPOD, spike_time_col='seconds')
    
    # --- Step 10: Mapping Trial Types for Plotting ---
    mapping = {'0': 'Laser_Only', '1': 'Upsweep', '2': 'Downsweep', '3': 'Opto_Upsweep', '4': 'Opto_Downsweep'}
    trial_types_mapped = (
        BPOD.groupby("trial_number")["trial_type"]
        .unique().explode().astype(str).map(mapping).fillna('Unknown').values
    )

    # --- Step 11: Plotting ---
    plot_dir = basepath / 'plots'
    plot_dir.mkdir(exist_ok=True)
    
    # Use raw seconds for the most accurate PSTH
    raw_spike_seconds = Ephys_good['seconds'].values
    
    plot_psth(raw_spike_seconds, ttlsound, save_path=plot_dir / 'psth_overall.png')
    plot_psth_by_type(raw_spike_seconds, ttlsound, trial_types_mapped, save_path=plot_dir / 'psth_by_type.png')

    # --- Step 12: Saving CSVs ---
    # Filter columns to keep it clean
    cols_to_keep = ['cluster_id', 'state_name', 'trial_number', 'trial_type']
    
    # Save binned
    out_binned = events_with_behv[cols_to_keep + ['time_bin', 'event_count']]
    out_binned.to_csv(basepath / output_filename, index=False)
    
    # Save noBin
    out_no_bin = events_with_behv_noBin[cols_to_keep + ['seconds']]
    out_no_bin.to_csv(basepath / 'noBin.csv', index=False)
    
    BPOD.to_csv(basepath / 'BPOD.csv')

    # --- Summary ---
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print(f"Good clusters: {len(Ephys_good['cluster_id'].unique())}")
    print(f"Annotated events (binned): {len(events_with_behv)}")
    print(f"Annotated events (noBin): {len(events_with_behv_noBin)}")
    print(f"Plots saved in: {plot_dir}")
    print("="*50)
    
    return events_with_behv