# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd
import zetapy as zp


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

# True:
#   Use BPOD.csv continuous_start timestamps for EVERY trial.
#   Audio.csv is completely ignored and is not required.
#
# False:
#   Use Audio.csv timestamps for sound trials (types 1-4), and BPOD
#   timestamps for Laser_Only trials (type 0).
USE_BPOD_TIMESTAMPS_ONLY = True


# ---------------------------------------------------------------------
# Trial type mapping
# ---------------------------------------------------------------------

MAPPING = {
    0: "Laser_Only",
    1: "Upsweep",
    2: "Downsweep",
    3: "Opto_Upsweep",
    4: "Opto_Downsweep",
}

# Optogenetic sound trial types use a short analysis window and shift.
OPTO_TRIAL_TYPES = {3, 4}


# ---------------------------------------------------------------------
# Compute ZETA
# ---------------------------------------------------------------------

def compute_zeta(spikes_df, event_times, max_dur, resamp):
    """
    Compute ZETA p-values for every good/MUA unit.
    """
    if len(event_times) == 0:
        return pd.DataFrame(columns=["unit", "p_val"])

    units = []
    p_values = []

    unique_units = spikes_df["unit_index"].unique()
    total_units = len(unique_units)

    for unit_number, unit in enumerate(unique_units, start=1):
        print(
            f"        Unit {unit_number}/{total_units}",
            end="\r",
            flush=True,
        )

        spike_times = spikes_df.loc[
            spikes_df["unit_index"] == unit,
            "time_s",
        ].to_numpy()

        p_value = zp.zetatest(
            spike_times,
            event_times,
            dblUseMaxDur=max_dur,
            boolPlot=False,
            intResampNum=resamp,
        )[0]

        units.append(unit)
        p_values.append(p_value)

    print(" " * 60, end="\r")

    return pd.DataFrame(
        {
            "unit": units,
            "p_val": p_values,
        }
    )


# ---------------------------------------------------------------------
# Create BPOD event times
# ---------------------------------------------------------------------

def get_bpod_event_table(bpod):
    """
    Return one BPOD event timestamp per trial.

    Requires:
        trial_number
        trial_type
        state_name
        continuous_start

    For Laser_Only trials:
        Uses the start of the 'Laser_Only' state.

    For sound trials:
        Prefer the start of 'sound_delay', if present. In BPOD files made
        by newnewhelper.py, sound_delay begins at the original BPOD sound
        command time, before Audio.csv-based timing adjustment.

        If a sound_delay row is not present, use the start of the expected
        state name instead, e.g. Upsweep, Downsweep, Opto_Upsweep, etc.
    """
    required_columns = {
        "trial_number",
        "trial_type",
        "state_name",
        "continuous_start",
    }

    missing_columns = required_columns - set(bpod.columns)

    if missing_columns:
        return None, (
            "BPOD timestamp mode requires column(s): "
            + ", ".join(sorted(missing_columns))
        )

    bpod = bpod.copy()

    bpod["trial_type"] = pd.to_numeric(
        bpod["trial_type"],
        errors="coerce",
    )

    bpod["continuous_start"] = pd.to_numeric(
        bpod["continuous_start"],
        errors="coerce",
    )

    bpod = bpod.dropna(
        subset=[
            "trial_number",
            "trial_type",
            "continuous_start",
        ]
    ).copy()

    bpod["trial_type"] = bpod["trial_type"].astype(int)
    bpod["state_name"] = bpod["state_name"].astype(str)

    # One row per trial, preserving trial order.
    trial_table = (
        bpod[
            ["trial_number", "trial_type"]
        ]
        .drop_duplicates(subset="trial_number")
        .sort_values("trial_number")
        .reset_index(drop=True)
    )

    event_rows = []
    missing_events = []

    for _, trial in trial_table.iterrows():
        trial_number = trial["trial_number"]
        trial_type = int(trial["trial_type"])

        trial_rows = bpod.loc[
            bpod["trial_number"] == trial_number
        ].copy()

        expected_state = MAPPING.get(trial_type)

        if expected_state is None:
            missing_events.append(
                f"trial {trial_number}: unknown trial_type {trial_type}"
            )
            continue

        if trial_type == 0:
            # Laser-only trials: use Laser_Only state onset.
            candidates = trial_rows.loc[
                trial_rows["state_name"] == "Laser_Only"
            ].copy()

        else:
            # Sound trials:
            # sound_delay is the original BPOD command time if it exists.
            candidates = trial_rows.loc[
                trial_rows["state_name"] == "sound_delay"
            ].copy()

            # If no sound_delay is present, use the actual named sound state.
            if candidates.empty:
                candidates = trial_rows.loc[
                    trial_rows["state_name"] == expected_state
                ].copy()

        if candidates.empty:
            missing_events.append(
                f"trial {trial_number} ({expected_state}): "
                "no matching BPOD state"
            )
            continue

        # If there is more than one matching row, use the first in time.
        event_time = candidates["continuous_start"].min()

        event_rows.append(
            {
                "trial_number": trial_number,
                "trial_type": trial_type,
                "event_time": event_time,
            }
        )

    if missing_events:
        preview = "; ".join(missing_events[:10])

        if len(missing_events) > 10:
            preview += f"; ... plus {len(missing_events) - 10} more"

        return None, (
            f"could not find a BPOD event time for "
            f"{len(missing_events)} trial(s): {preview}"
        )

    event_table = pd.DataFrame(event_rows)

    if event_table.empty:
        return None, "no usable BPOD event timestamps found"

    return event_table, None


# ---------------------------------------------------------------------
# Create Audio/BPOD event times
# ---------------------------------------------------------------------

def get_audio_event_table(bpod, audio_file):
    """
    Use Audio.csv for sound trials and BPOD Laser_Only state starts for
    Laser_Only trials.

    Audio behavior:
    - If both rising and falling labels exist: use all rising rows.
    - If all edge labels are identical: use every second raw Audio row.
    """
    required_bpod_columns = {
        "trial_number",
        "trial_type",
        "state_name",
        "continuous_start",
    }

    missing_columns = required_bpod_columns - set(bpod.columns)

    if missing_columns:
        return None, (
            "Audio mode still requires BPOD column(s) for Laser_Only trials: "
            + ", ".join(sorted(missing_columns))
        )

    timestamps = pd.read_csv(audio_file)

    required_timestamp_columns = {
        "timestamps",
        "edge_type",
    }

    missing_columns = required_timestamp_columns - set(timestamps.columns)

    if missing_columns:
        return None, (
            "missing Audio column(s): "
            + ", ".join(sorted(missing_columns))
        )

    bpod = bpod.copy()

    bpod["trial_type"] = pd.to_numeric(
        bpod["trial_type"],
        errors="coerce",
    )

    bpod["continuous_start"] = pd.to_numeric(
        bpod["continuous_start"],
        errors="coerce",
    )

    bpod = bpod.dropna(
        subset=[
            "trial_number",
            "trial_type",
        ]
    ).copy()

    bpod["trial_type"] = bpod["trial_type"].astype(int)
    bpod["state_name"] = bpod["state_name"].astype(str)

    trial_table = (
        bpod[
            ["trial_number", "trial_type"]
        ]
        .drop_duplicates(subset="trial_number")
        .sort_values("trial_number")
        .reset_index(drop=True)
    )

    timestamps["timestamps"] = pd.to_numeric(
        timestamps["timestamps"],
        errors="coerce",
    )

    timestamps = timestamps.dropna(
        subset=["timestamps"]
    ).reset_index(drop=True)

    edge_types = (
        timestamps["edge_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unique_edge_types = set(edge_types.unique())

    if "rising" in unique_edge_types and "falling" in unique_edge_types:
        sound_timestamps = timestamps.loc[
            edge_types == "rising",
            "timestamps",
        ].reset_index(drop=True)

        timestamp_method = "explicitly labelled rising edges"

    elif len(unique_edge_types) == 1:
        sound_timestamps = timestamps[
            "timestamps"
        ].iloc[::2].reset_index(drop=True)

        timestamp_method = (
            "every second Audio edge "
            f"(all labels: {sorted(unique_edge_types)})"
        )

    else:
        return None, (
            "cannot determine Audio sound onsets; edge_type values are "
            f"{sorted(unique_edge_types)}"
        )

    # Audio exists only for sound trial types 1-4.
    sound_trial_table = trial_table.loc[
        trial_table["trial_type"] != 0
    ].copy()

    if len(sound_timestamps) != len(sound_trial_table):
        return None, (
            "sound alignment mismatch: "
            f"Audio sound timestamps={len(sound_timestamps)}, "
            f"BPOD sound trials={len(sound_trial_table)}"
        )

    sound_trial_table["event_time"] = sound_timestamps.to_numpy()

    # Laser_Only trials obtain timing directly from BPOD.
    laser_trials = trial_table.loc[
        trial_table["trial_type"] == 0
    ].copy()

    laser_rows = bpod.loc[
        bpod["state_name"] == "Laser_Only",
        [
            "trial_number",
            "continuous_start",
        ],
    ].copy()

    laser_rows = (
        laser_rows
        .dropna(subset=["continuous_start"])
        .groupby("trial_number", as_index=False)["continuous_start"]
        .min()
    )

    laser_event_table = laser_trials.merge(
        laser_rows,
        on="trial_number",
        how="left",
    ).rename(
        columns={"continuous_start": "event_time"}
    )

    missing_laser = laser_event_table["event_time"].isna().sum()

    if missing_laser:
        return None, (
            f"missing BPOD Laser_Only onset for "
            f"{missing_laser} Laser_Only trial(s)"
        )

    event_table = pd.concat(
        [
            sound_trial_table[
                ["trial_number", "trial_type", "event_time"]
            ],
            laser_event_table[
                ["trial_number", "trial_type", "event_time"]
            ],
        ],
        ignore_index=True,
    ).sort_values(
        "trial_number"
    ).reset_index(
        drop=True
    )

    print(f"Audio selection method: {timestamp_method}")

    return event_table, None


# ---------------------------------------------------------------------
# Run ZETA for one recording folder
# ---------------------------------------------------------------------

def run_zeta_for_folder(
    folder_path,
    folder_number,
    total_folders,
):
    """
    Run ZETA separately for trial types 0 through 4 for one folder.

    Returns:
        success, reason
    """
    folder_path = Path(folder_path)

    print("\n" + "=" * 80)
    print(
        f"Folder {folder_number}/{total_folders}: "
        f"{folder_path}"
    )
    print("=" * 80)

    spike_file = (
        folder_path
        / "spike_times"
        / "spike_times.npy"
    )

    bpod_file = folder_path / "BPOD.csv"
    audio_file = folder_path / "Meta" / "Audio.csv"

    zeta_folder = folder_path / "Meta" / "Zeta"

    # Audio is not required when using only BPOD timestamps.
    required_files = [
        spike_file,
        bpod_file,
    ]

    if not USE_BPOD_TIMESTAMPS_ONLY:
        required_files.append(audio_file)

    missing_files = [
        file_path
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        reason = (
            "missing required file(s): "
            + ", ".join(
                file_path.name
                for file_path in missing_files
            )
        )

        print(reason)
        print("Skipping folder.")

        return False, reason

    zeta_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Load spikes
    # -----------------------------------------------------------------

    spikes = np.load(
        spike_file,
        allow_pickle=True,
    )

    spikes_df = pd.DataFrame(spikes)

    required_spike_columns = {
        "unit_index",
        "time_s",
        "label",
    }

    missing_columns = required_spike_columns - set(spikes_df.columns)

    if missing_columns:
        reason = (
            "missing spike column(s): "
            + ", ".join(sorted(missing_columns))
        )

        print(reason)
        print("Skipping folder.")

        return False, reason

    spikes_df = spikes_df.loc[
        spikes_df["label"].isin(["mua", "good"])
    ].copy()

    if spikes_df.empty:
        reason = "no units labelled 'good' or 'mua'"

        print(reason)
        print("Skipping folder.")

        return False, reason

    total_units = spikes_df["unit_index"].nunique()

    print(f"Loaded {total_units} good/MUA units.")

    # -----------------------------------------------------------------
    # Load BPOD
    # -----------------------------------------------------------------

    bpod = pd.read_csv(bpod_file)

    # -----------------------------------------------------------------
    # Select event times
    # -----------------------------------------------------------------

    if USE_BPOD_TIMESTAMPS_ONLY:
        print(
            "Timing mode: BPOD-only "
            "(Audio.csv ignored)."
        )

        event_table, error_reason = get_bpod_event_table(bpod)

    else:
        print(
            "Timing mode: Audio for sound trials; "
            "BPOD for Laser_Only trials."
        )

        event_table, error_reason = get_audio_event_table(
            bpod,
            audio_file,
        )

    if error_reason is not None:
        print(f"Event-time error: {error_reason}")
        print("Skipping folder.")

        return False, error_reason

    trial_types_present = set(
        event_table["trial_type"]
        .astype(int)
        .unique()
    )

    print(
        "Trial types with usable event times: "
        f"{sorted(trial_types_present)}"
    )

    print(
        "Event counts by trial type: "
        f"{event_table['trial_type'].value_counts().sort_index().to_dict()}"
    )

    # -----------------------------------------------------------------
    # Run ZETA separately for each trial type
    # -----------------------------------------------------------------

    total_trial_types = len(MAPPING)

    for trial_number, (
        trial_type,
        output_name,
    ) in enumerate(
        MAPPING.items(),
        start=1,
    ):
        if trial_type not in trial_types_present:
            print(
                f"[{trial_number}/{total_trial_types}] "
                f"{output_name}: trial type not present; skipping."
            )
            continue

        output_file = zeta_folder / f"{output_name}.csv"

        if output_file.exists():
            print(
                f"[{trial_number}/{total_trial_types}] "
                f"{output_name}: output already exists; skipping."
            )
            continue

        event_times = event_table.loc[
            event_table["trial_type"] == trial_type,
            "event_time",
        ].to_numpy(
            dtype=float
        )

        # Trial types 3 and 4:
        # - shift timestamps by -0.1 seconds
        # - use a 0.1-second analysis window
        if trial_type in OPTO_TRIAL_TYPES:
            event_times = event_times - 0.1
            max_dur = 0.1
            resamp = 500
            timestamp_shift = "-0.1 s"

        else:
            max_dur = 1.0
            resamp = 1000
            timestamp_shift = "none"

        print(
            f"[{trial_number}/{total_trial_types}] "
            f"Running {output_name}: "
            f"{len(event_times)} trials, "
            f"max_dur={max_dur}, "
            f"shift={timestamp_shift}"
        )

        zeta_results = compute_zeta(
            spikes_df=spikes_df,
            event_times=event_times,
            max_dur=max_dur,
            resamp=resamp,
        )

        zeta_results.to_csv(
            output_file,
            index=False,
        )

        print(f"    Saved: {output_file}")

    print(
        f"Finished folder "
        f"{folder_number}/{total_folders}"
    )

    return True, "complete"


# ---------------------------------------------------------------------
# Run ZETA for every recording folder
# ---------------------------------------------------------------------

def run_zeta_for_root(root_dir):
    """
    Run ZETA analysis for every recording folder under root_dir.

    Prints a final per-folder summary:
        folder_name: complete
    or:
        folder_name: fail : reason
    """
    root_dir = Path(root_dir)

    folders = sorted(
        spike_times_dir.parent
        for spike_times_dir in root_dir.rglob("spike_times")
        if spike_times_dir.is_dir()
    )

    total_folders = len(folders)

    if total_folders == 0:
        print(
            f"No recording folders found under: "
            f"{root_dir}"
        )
        return

    print(f"Found {total_folders} recording folders.")

    if USE_BPOD_TIMESTAMPS_ONLY:
        print(
            "Using BPOD-only timing. "
            "Audio.csv will be ignored."
        )
    else:
        print(
            "Using Audio.csv timing for sound trials and "
            "BPOD timing for Laser_Only trials."
        )

    print("Existing ZETA CSV files will be skipped.")
    print("Results are saved in each folder's Meta/Zeta directory.")

    summary = []

    for folder_number, folder_path in enumerate(
        folders,
        start=1,
    ):
        try:
            success, reason = run_zeta_for_folder(
                folder_path=folder_path,
                folder_number=folder_number,
                total_folders=total_folders,
            )

        except Exception as exc:
            success = False
            reason = f"{type(exc).__name__}: {exc}"

            print(
                f"\nERROR processing folder "
                f"{folder_number}/{total_folders}:"
            )
            print(f"    {folder_path}")
            print(f"    {reason}")

        summary.append(
            {
                "folder": folder_path.name,
                "success": success,
                "reason": reason,
            }
        )

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ZETA PROCESSING SUMMARY")
    print("=" * 80)

    for result in summary:
        if result["success"]:
            print(f"{result['folder']}: complete")
        else:
            print(
                f"{result['folder']}: "
                f"fail : {result['reason']}"
            )

    completed = sum(
        result["success"]
        for result in summary
    )

    failed = len(summary) - completed

    print("-" * 80)
    print(f"Complete: {completed}/{len(summary)}")
    print(f"Failed:   {failed}/{len(summary)}")
    print("=" * 80)
    print("All folders finished.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    root = r"I:\Data\raw"
    run_zeta_for_root(root)