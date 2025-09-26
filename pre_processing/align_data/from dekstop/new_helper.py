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

def check_state_alignment(ITI, raw_BPOD):
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

def BPOD_wrangle_claude(raw_BPOD, ITI, proceed):
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
            print('ITI matches number of states')
            ITI_trimmed = ITI
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


def add_sound_delays(BPOD, ttlsound):
    """
    Simple function to add sound_delay states and adjust sound timing
    
    Args:
        BPOD: DataFrame with BPOD states
        ttlsound: Series/DataFrame with actual sound timing
    
    Returns:
        Modified BPOD DataFrame with sound_delay states inserted
    """
    sound_types = ['Downsweep', 'Opto_Downsweep', 'Opto_Upwsweep', 'Upsweep']
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
            actual_sound_time = ttlsound.iloc[ttl_index, 0] if hasattr(ttlsound, 'iloc') else ttlsound[ttl_index]
            delay_duration = actual_sound_time - bpod_start
            
            if delay_duration <= 0:
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


def Ephys_wrangle(cluster_info,clust,times,sample_freq):
    #takes raw KS vectos and turns them into a Dataframe  
    #selects only clusters that have the PHY good or MUA label
    
    good_cluster = cluster_info.query('group == "good" ').loc[:,['cluster_id']]
    #good_cluster = cluster_info.query('group == "good" | group == "mua"').loc[:,['cluster_id']]
    Ephys_raw = pd.DataFrame({'cluster_id': clust, 'times':times}, columns=['cluster_id', 'times'])
    Ephys_raw = Ephys_raw.assign(seconds = Ephys_raw['times']/sample_freq)
    Ephys_good = Ephys_raw.merge(good_cluster, on=['cluster_id'],how='inner')
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
                                 start_col='continuous_start', end_col='continuous_time',aditional_time=30):
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
    
    last_bpod_time = bpod_sorted[end_col].max()
    
    events_with_behv = merged[
        ((merged[spike_time_col] >= merged[start_col]) & 
         (merged[spike_time_col] <= merged[end_col])) |
        ((merged[spike_time_col] > last_bpod_time) & 
         (merged[spike_time_col] <= last_bpod_time + aditional_time))
    ]
    
    print(f"Matched {len(events_with_behv)} out of {len(ephys_df)} spikes")
    return events_with_behv

@timeit
def process_neural_data_pipeline(basefolder, output_filename='processed_data.csv', bin_size=0.01):
    """
    Complete pipeline to process neural data from raw files to annotated spike data.
    
    Args:
        basefolder (str): Path to the base folder containing all data files
        output_filename (str): Name for the output CSV file (default: 'processed_data.csv')
        bin_size (float): Bin size for spike binning in seconds (default: 0.01)
    
    Returns:
        pandas.DataFrame: Final processed data with behavioral annotations
    """
    import os
    import numpy as np
    import pandas as pd
    from new_helper import (load_mat, check_state_alignment, BPOD_wrangle_claude, 
                           add_sound_delays, Ephys_wrangle, bin_Ephys, 
                           annotate_spikes_interval_join,timeit)
    
    print(f"Starting neural data processing pipeline for: {basefolder}")
    
    # Step 1: Find and construct file paths
    print("Step 1: Loading file paths...")
    try:
        mat_name = next(file for file in os.listdir(basefolder) if file.endswith('.mat'))
        last_part = basefolder.split('/')[-1]
        meta_path = f"{basefolder}/{last_part}_imec0/{last_part}_t0.imec0.ap.meta"
        print(f"  Found .mat file: {mat_name}")
        print(f"  Meta path: {meta_path}")
    except StopIteration:
        raise FileNotFoundError("No .mat file found in the base folder")
    
    # Step 2: Load all data files
    print("Step 2: Loading data files...")
    try:
        # Load cluster info
        cluster_info = pd.read_csv(basefolder + '/sorted/phy/cluster_info.tsv', sep='\t')
        print(f"  Loaded cluster info: {len(cluster_info)} clusters")
        
        # Load spike data
        clust = np.array(np.load(basefolder + '/sorted/phy/spike_clusters.npy'))
        times = np.array(np.load(basefolder + '/sorted/phy/spike_times.npy'))[:, 0]
        print(f"  Loaded spike data: {len(times)} spikes")
        
        # Load ITI data
        ITI = pd.read_csv(basefolder + '/Meta/ttl_edge_times.csv')
        print(f"  Loaded ITI data: {len(ITI)} time points")
        
        # Load BPOD data
        raw_BPOD = load_mat(basefolder + '/' + mat_name)
        print(f"  Loaded BPOD data")
        
        # Load sound TTL data
        ttlsound = pd.read_csv(basefolder + '/Meta/soundttl.csv')
        print(f"  Loaded sound TTL data: {len(ttlsound)} sound events")
        
        # Get sampling frequency
        sampling_freq = float([line.split('=')[1] for line in open(meta_path) 
                              if line.startswith('imSampRate=')][0])
        print(f"  Sampling frequency: {sampling_freq} Hz")
        
    except Exception as e:
        raise RuntimeError(f"Error loading data files: {str(e)}")
    
    # Step 3: Format ITI data
    print("Step 3: Formatting ITI data...")
    ITI = ITI.iloc[:, 0]
    print(f"  ITI data formatted: {len(ITI)} time points")
    
    # Step 4: Check state alignment
    print("Step 4: Checking state alignment...")
    proceed = check_state_alignment(ITI, raw_BPOD)
    if not proceed:
        raise RuntimeError("State alignment check failed. Cannot proceed with processing.")
    print("  ✓ State alignment check passed")
    
    # Step 5: Wrangle BPOD data
    print("Step 5: Processing BPOD data...")
    BPOD = BPOD_wrangle_claude(raw_BPOD, ITI, proceed)
    if BPOD is None:
        raise RuntimeError("BPOD wrangling failed")
    print(f"  BPOD data processed: {len(BPOD)} behavioral states")
    
    # Step 6: Add sound delays
    print("Step 6: Adding sound delays...")
    try:
        BPOD = add_sound_delays(BPOD, ttlsound)
        print(f"  Sound delays added: {len(BPOD)} total states")
    except Exception as e:
        print(f"  Warning: Could not add sound delays: {str(e)}")
        print("  Continuing without sound delay correction...")
        raise RuntimeError("Error with sound delay")
    
    # Step 7: Process electrophysiology data
    print("Step 7: Processing electrophysiology data...")
    Ephys_good = Ephys_wrangle(cluster_info, clust, times, sampling_freq)
    print(f"  Good clusters processed: {len(Ephys_good['cluster_id'].unique())} clusters, {len(Ephys_good)} spikes")
    
    # Step 8: Bin ephys data
    print(f"Step 8: Binning ephys data (bin_size={bin_size}s)...")
    Ephys_binned = bin_Ephys(Ephys_good, bin_size=bin_size)
    print(f"  Ephys data binned: {len(Ephys_binned)} time bins")
    
    # Step 9: Annotate spikes with behavioral states
    print("Step 9: Annotating spikes with behavioral states...")
    events_with_behv = annotate_spikes_interval_join(Ephys_binned, BPOD, spike_time_col='time_bin')
    print(f"  Spike annotation complete: {len(events_with_behv)} annotated events")
    
    # Step 10: Save results
    print("Step 10: Saving results...")
    events_with_behv = events_with_behv[['cluster_id','time_bin','event_count','state_name','trial_number','trial_type']]
    output_path = basefolder + '/' + output_filename
    events_with_behv.to_csv(output_path, index=False)
    print(f"  Results saved to: {output_path}")
    
    # Step 11: Print summary statistics
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Base folder: {basefolder}")
    print(f"Total clusters: {len(cluster_info)}")
    print(f"Good clusters: {len(Ephys_good['cluster_id'].unique())}")
    print(f"Total spikes: {len(times)}")
    print(f"Behavioral states: {len(BPOD)}")
    print(f"Annotated events: {len(events_with_behv)}")
    print(f"Matching rate: {len(events_with_behv)/len(Ephys_binned)*100:.1f}%")
    print(f"Output file: {output_path}")
    print("="*50)
    
    return events_with_behv