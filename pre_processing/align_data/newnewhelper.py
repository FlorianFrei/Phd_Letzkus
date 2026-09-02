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


def check_state_alignment_old(ITI, raw_BPOD, n_checks=10, tolerance=0.02):
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

def _collect_bpod_durations(session_data):
    """Shared by check_state_alignment and BPOD_wrangle_claude: flattens per-trial
    state durations in recording order, flagging which are last-in-trial
    (followed by a dead-time gap, so their next ITI interval is expected to be inflated)."""
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
    return bpod_durations, is_last_in_trial


def _find_pulse_offset(iti_arr, bpod_durations, is_last_in_trial, max_offset, n_checks=10, tolerance=0.02):
    """
    Tests candidate front-offsets (0..max_offset) and scores each by how well
    INTRA-TRIAL ITI intervals match known BPOD state durations. Boundary
    (last-in-trial) intervals are skipped since dead-time inflates them and
    they carry no information about offset correctness.
    Returns (best_offset, score, n_compared).
    """
    best_offset, best_score, best_n = 0, -1, 0
    for offset in range(max_offset + 1):
        shifted = iti_arr[offset:]
        iti_intervals = np.diff(shifted)
        n = min(n_checks, len(bpod_durations) - 1, len(iti_intervals))

        passed = total = 0
        for i in range(n):
            if is_last_in_trial[i]:
                continue
            total += 1
            if abs(iti_intervals[i] - bpod_durations[i]) < tolerance:
                passed += 1

        score = (passed / total) if total else 0.0
        if score > best_score:
            best_offset, best_score, best_n = offset, score, total
        if score == 1.0:
            break

    return best_offset, best_score, best_n


def _build_row_skeleton(session_data):
    """
    Single source of truth for row order: real states (dropna'd) interleaved
    with dead_time placeholders, in true pulse order, one row per expected
    ITI pulse. Used by BOTH check_state_alignment and BPOD_wrangle_claude so
    the two can never drift out of sync with each other.

    Returns:
        combined_df: rows with state_name/trial_number, real durations filled,
                     dead_time rows present but undated.
        known_duration: np.array, same length as combined_df. Real-state rows
                     hold their BPOD-measured duration; dead_time rows are NaN
                     (unknown — that's what we're solving for, so they must be
                     SKIPPED in comparisons, never treated as a 0 or missing slot).
    """
    n_trials = len(session_data['TrialStartTimestamp'])
    trial_dataframes = []

    for trial_idx, trial_data in enumerate(session_data['RawEvents']['Trial']):
        states_df = pd.DataFrame.from_dict(trial_data['States']).transpose()
        states_df['state_name'] = states_df.index
        states_df['trial_number'] = trial_idx
        states_df = states_df.dropna()
        states_df = states_df.sort_values(0)  # ensure recording order within trial

        if trial_idx < n_trials - 1:
            dead_row = pd.DataFrame({
                0: [np.nan], 1: [np.nan],
                'state_name': ['dead_time'],
                'trial_number': [trial_idx],
            })
            states_df = pd.concat([states_df, dead_row], ignore_index=True)

        trial_dataframes.append(states_df)

    combined_df = pd.concat(trial_dataframes, ignore_index=True)
    known_duration = np.where(
        combined_df['state_name'] == 'dead_time',
        np.nan,
        combined_df[1] - combined_df[0]
    )
    return combined_df, known_duration


def check_state_alignment(ITI, raw_BPOD, n_checks=10, tolerance=0.02):
    """
    Validates alignment using the SAME row skeleton as BPOD_wrangle_claude,
    so iti_intervals[i] and known_duration[i] always refer to the same row.
    Dead-time rows (known_duration is NaN) are skipped entirely — they carry
    no comparison information, they're not "expected mismatches" to be lenient about.
    """
    session_data = raw_BPOD['SessionData']
    iti_arr = np.asarray(ITI.values if hasattr(ITI, 'values') else ITI, dtype=float)
    combined_df, known_duration = _build_row_skeleton(session_data)

    print(f"ITI state changes: {len(iti_arr)}")
    print(f"BPOD trials:       {len(session_data['TrialStartTimestamp'])}")

    iti_intervals = np.diff(iti_arr)
    n = min(n_checks, len(known_duration), len(iti_intervals))

    print(f"\nFirst {n} rows (dead_time rows shown but not scored):")
    print(f"{'#':>3}  {'':5}  {'ITI interval':>12}  {'BPOD duration':>13}  {'diff':>8}  {'row':>10}")
    print("-" * 70)

    passed = total = 0
    for i in range(n):
        row_name = combined_df['state_name'].iloc[i]
        if np.isnan(known_duration[i]):
            print(f"{i+1:>3}  {'skip':<5}  {iti_intervals[i]:>12.4f}  {'--':>13}  {'--':>8}  {row_name:>10}")
            continue
        diff = iti_intervals[i] - known_duration[i]
        ok = abs(diff) < tolerance
        total += 1
        passed += ok
        print(f"{i+1:>3}  {'✓' if ok else '✗':<5}  {iti_intervals[i]:>12.4f}  {known_duration[i]:>13.4f}  {diff:>+8.4f}s  {row_name:>10}")

    print("-" * 70)
    if total == 0:
        print("Warning: no scorable (non-dead-time) rows in this window — "
              "increase n_checks, this result is NOT evidence of alignment")
        return False  # don't vacuously pass
    print(f"Scorable rows passing: {passed}/{total}")
    result = passed / total >= 0.5
    print("✓ Alignment acceptable\n" if result else "✗ Alignment failed — review output above\n")
    return result


def _find_pulse_offset(iti_arr, known_duration, max_offset, n_checks=10, tolerance=0.02):
    """Same fix applied here: scores candidate offsets using the shared
    known_duration skeleton, skipping NaN (dead_time) rows."""
    best_offset, best_score, best_n = 0, -1, 0
    for offset in range(max_offset + 1):
        iti_intervals = np.diff(iti_arr[offset:])
        n = min(n_checks * 3, len(known_duration), len(iti_intervals))  # widen window past baseline block

        passed = total = 0
        for i in range(n):
            if np.isnan(known_duration[i]):
                continue
            total += 1
            if abs(iti_intervals[i] - known_duration[i]) < tolerance:
                passed += 1

        score = (passed / total) if total else -1  # no data => can't judge, don't treat as 0
        if score > best_score:
            best_offset, best_score, best_n = offset, score, total
        if score == 1.0:
            break

    return best_offset, best_score, best_n


def BPOD_wrangle_claude(raw_BPOD, ITI, proceed, n_checks=10, tolerance=0.02):
    if not proceed:
        print("Stop, this is not gonna work")
        return None

    session_data = raw_BPOD['SessionData']
    iti_arr = np.asarray(ITI.values if hasattr(ITI, 'values') else ITI, dtype=float)
    combined_df, known_duration = _build_row_skeleton(session_data)

    n_rows = len(combined_df)
    n_iti = len(iti_arr)
    print(f"Rows (real states + dead_time placeholders): {n_rows}")
    print(f"ITI pulses: {n_iti}")

    if n_iti < n_rows:
        print(f"Error: {n_rows - n_iti} rows have no ITI pulse — cannot proceed")
        return None

    if n_iti > n_rows:
        extra = n_iti - n_rows
        offset, score, n_compared = _find_pulse_offset(
            iti_arr, known_duration, max_offset=extra, n_checks=n_checks, tolerance=tolerance
        )
        print(f"Front-alignment check: best offset={offset} "
              f"(match {score:.0%} over {n_compared} pairs)" if score >= 0
              else "Front-alignment check: no scorable pairs found — cannot verify, proceeding with caution")

        if 0 <= score < 0.5:
            print("Warning: front-alignment is poor even after searching offsets — "
                  "inspect manually before trusting this mapping")

        if offset > 0:
            print(f"Dropping {offset} leading pulse(s) (confirmed pre-first-state)")
            iti_arr = iti_arr[offset:]

        remaining_extra = len(iti_arr) - n_rows
        if remaining_extra > 0:
            print(f"Dropping {remaining_extra} trailing pulse(s) (after front-alignment)")
            iti_arr = iti_arr[:n_rows]

    combined_df['continuous_start'] = iti_arr[:n_rows]
    combined_df['state_duration'] = combined_df[1] - combined_df[0]
    combined_df['continuous_time'] = combined_df['continuous_start'] + combined_df['state_duration']

    dead_mask = combined_df['state_name'] == 'dead_time'
    next_start = combined_df['continuous_start'].shift(-1)
    combined_df.loc[dead_mask, 'continuous_time'] = next_start[dead_mask]
    combined_df.loc[dead_mask, 'state_duration'] = (
        combined_df.loc[dead_mask, 'continuous_time'] - combined_df.loc[dead_mask, 'continuous_start']
    )

    combined_df = combined_df.sort_values('continuous_start').reset_index(drop=True)
    combined_df['trial_type'] = combined_df['trial_number'].map(
        lambda n: session_data['TrialTypes'][int(n)]
    )
    return combined_df

def add_sound_delays_old(BPOD, ttlsound):
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

def add_sound_delays(BPOD, ttlsound, tolerance_max=0.1):
    """
    Inserts sound_delay states and re-times sound-state rows using ttlsound
    as ground truth.

    ttlsound may contain extra pulses (e.g. leading test/calibration pulses)
    that don't correspond to any real sound-state row. Blindly consuming
    ttlsound in order desyncs ttl_index permanently from the first such
    extra pulse onward. We verify the correct starting offset first, using
    the same principle as the ITI alignment check: true delays must be
    small and non-negative.
    """
    sound_types = ['Downsweep', 'Opto_Downsweep', 'Opto_Upsweep', 'Upsweep']
    ttl_arr = np.asarray(ttlsound.values if hasattr(ttlsound, 'values') else ttlsound, dtype=float)
  # rising edges only, assuming rising-falling-rising-falling...
    sound_rows = BPOD[BPOD['state_name'].isin(sound_types)]
    n_sound = len(sound_rows)
    n_ttl = len(ttl_arr)
    print(f"Sound-state rows in BPOD: {n_sound}")
    print(f"ttlsound pulses: {n_ttl}")

    if n_ttl < n_sound:
        raise ValueError(f"Not enough ttlsound pulses ({n_ttl}) for {n_sound} sound states")

    max_offset = n_ttl - n_sound
    starts = sound_rows['continuous_start'].values

    best_offset, best_score = 0, -1
    for offset in range(max_offset + 1):
        candidate = ttl_arr[offset: offset + n_sound]
        delays = candidate - starts
        valid = np.sum((delays > -0.001) & (delays < tolerance_max))
        score = valid / n_sound
        if score > best_score:
            best_offset, best_score = offset, score
        if score == 1.0:
            break

    print(f"Best ttlsound offset={best_offset} (valid delays: {best_score:.0%})")
    if best_score < 1.0:
        print("Warning: some sound rows have implausible delays even after offset search — "
              "inspect manually before trusting this mapping")

    ttl_arr = ttl_arr[best_offset:]  # apply the verified offset once, up front

    result_rows = []
    ttl_index = 0

    for _, row in BPOD.iterrows():
        result_rows.append(row.copy())

        if row['state_name'] in sound_types:
            if ttl_index >= len(ttl_arr):
                raise ValueError(f"Not enough TTL values for sound state (trial {row['trial_number']})")

            bpod_start = row['continuous_start']
            actual_sound_time = ttl_arr[ttl_index]
            delay_duration = actual_sound_time - bpod_start

            if delay_duration <= -0.001:
                raise ValueError(
                    f"Sound plays before BPOD signal: delay = {delay_duration} "
                    f"(trial {row['trial_number']}, state {row['state_name']})"
                )

            delay_row = row.copy()
            delay_row['state_name'] = 'sound_delay'
            delay_row['continuous_start'] = bpod_start
            delay_row['continuous_time'] = actual_sound_time
            delay_row['state_duration'] = delay_duration
            result_rows.insert(-1, delay_row)

            sound_row = result_rows[-1]
            original_sound_end = sound_row['continuous_time']
            sound_row['continuous_start'] = actual_sound_time
            sound_row['state_duration'] = original_sound_end - actual_sound_time

            ttl_index += 1

    return pd.DataFrame(result_rows).reset_index(drop=True)


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

def plot_psth_by_type(spike_times, stim_times, stim_types, save_path=None, window=(-2, 6), bin_size=0.01, sigma=8):
    """
    Plot PSTHs grouped by stimulus type.
    stim_times and stim_types must be 1:1 paired arrays of equal length.
    """
    spike_times = np.asarray(spike_times).flatten()
    stim_times  = np.asarray(stim_times).flatten()
    stim_types  = np.asarray(stim_types)

    assert len(stim_times) == len(stim_types), (
        f"stim_times ({len(stim_times)}) and stim_types ({len(stim_types)}) must be the same length"
    )

    bins         = np.arange(window[0], window[1], bin_size)
    centers      = (bins[:-1] + bins[1:]) / 2
    unique_types = np.unique(stim_types)
    colors       = plt.cm.get_cmap('tab10', len(unique_types))

    plt.figure(figsize=(10, 5))

    for i, ttype in enumerate(unique_types):
        type_mask            = (stim_types == ttype)
        relevant_stim_times  = stim_times[type_mask]          # direct mask, no slicing
        n                    = type_mask.sum()

        all_aligned_spikes = []
        for t in relevant_stim_times:
            aligned_spikes = spike_times - t
            mask = (aligned_spikes >= window[0]) & (aligned_spikes < window[1])
            all_aligned_spikes.extend(aligned_spikes[mask])

        if n > 0:
            counts, _ = np.histogram(all_aligned_spikes, bins=bins)
            rate         = counts / (n * bin_size)
            rate_smoothed = gaussian_filter1d(rate, sigma=sigma)
            plt.plot(centers, rate_smoothed, label=f'{ttype} (n={n})', color=colors(i))

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

        edge_types = ttlsound_df['edge_type'].astype(str).str.strip().str.lower()
        audio_edges = ttlsound_df.iloc[:, 0].to_numpy(dtype=float)

        if edge_types.nunique(dropna=True) == 1:
            # edge_type is unusable because every row has the same label.
            # Assume the rows alternate rising, falling, rising, falling, ...
            print(
                "Warning: all Audio.csv edge_type values are identical; "
                "using every second edge (0::2) as the rising-edge timestamps."
            )
            ttlsound = audio_edges[0::2]
        else:
            # Use the explicitly labelled rising edges.
            ttlsound = audio_edges[edge_types.to_numpy() == 'rising']

        print(f"Sound alignment TTLs: {len(ttlsound)}") 


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
# --- Step 10: Build unified stimulus event table ---
    # ttlsound contains only HiFi events (types 1-4); Laser_Only uses WavePlayer1
    # and has no Audio.csv entry, so its onset must come from BPOD directly.
    mapping = {
        '0': 'Laser_Only',
        '1': 'Upsweep',
        '2': 'Downsweep',
        '3': 'Opto_Upsweep',
        '4': 'Opto_Downsweep',
    }

    # Laser_Only onsets: start of each Laser_Only state row in BPOD
    laser_times = BPOD.loc[BPOD['state_name'] == 'Laser_Only', 'continuous_start'].values
    laser_types = np.array(['Laser_Only'] * len(laser_times))

    # Sound-trial types (types 1-4), one per trial, in trial order → must pair 1:1 with ttlsound
    per_trial_type = (
        BPOD.groupby('trial_number')['trial_type']
        .first()
        .apply(lambda x: mapping.get(str(int(float(x))), 'Unknown'))
    )
    sound_types_arr = per_trial_type[per_trial_type != 'Laser_Only'].values
    
    if len(sound_types_arr) != len(ttlsound):
        print(
            f"Warning: {len(sound_types_arr)} sound trial types vs "
            f"{len(ttlsound)} TTL pulses — check trial count and Audio.csv"
        )

    # Merge and sort chronologically
    all_stim_times = np.concatenate([laser_times, ttlsound])
    all_stim_types = np.concatenate([laser_types, sound_types_arr])
    sort_idx       = np.argsort(all_stim_times)
    all_stim_times = all_stim_times[sort_idx]
    all_stim_types = all_stim_types[sort_idx]

    print(f"Stimulus events: {dict(zip(*np.unique(all_stim_types, return_counts=True)))}")

    # --- Step 11: Plotting ---
    plot_dir = basepath / 'plots'
    plot_dir.mkdir(exist_ok=True)

    raw_spike_seconds = Ephys_good['seconds'].values

    # Overall PSTH uses all event onsets (Laser_Only + sounds)
    plot_psth(raw_spike_seconds, all_stim_times, save_path=plot_dir / 'psth_overall.png')
    # Per-type PSTH uses the paired (times, types) arrays
    plot_psth_by_type(raw_spike_seconds, all_stim_times, all_stim_types,
                      save_path=plot_dir / 'psth_by_type.png')

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