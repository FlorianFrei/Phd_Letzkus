from pathlib import Path
import os
import shutil
import threading
import time
import warnings
from datetime import datetime

# Suppress the requests/urllib3 version mismatch warning.
# It appears once per worker subprocess (n_jobs=-3 spawns many), is harmless,
# and does not affect pipeline behaviour.
# The os.environ line propagates the filter to all child processes.
warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::Warning:requests,ignore:urllib3.*doesn't match a supported version"
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.interpolate import interp1d

import spikeinterface.full as si

try:
    import torch
except Exception:
    torch = None

try:
    import bombcell as bc  # noqa: F401
except Exception:
    bc = None


# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
GLOBAL_JOB_KWARGS = dict(n_jobs=-3, chunk_duration="10s", progress_bar=True)
si.set_global_job_kwargs(**GLOBAL_JOB_KWARGS)

AP_STREAM   = "imec0.ap"
NIDQ_STREAM = "nidq"
SORTER_NAME = "kilosort4"

# KS4 writes large binary temp files during sorting.
# Point this at a fast LOCAL SSD (not the same spinning drive as your data).
# Set to None to let KS4 choose its own temp dir (often also slow on Windows).
KS4_SCRATCH_DIR = None

SPEED_CH_A       = 6
SPEED_CH_B       = 5
CAMERA_ANALOG_CH = 3
DIGITAL_WORD_CH  = 8

PULSES_PER_REV = 900
BIT_NAME_PAIRS = [(1, "State_changes"), (3, "Audio")]

# ── Skip / overwrite behaviour ─────────────────────────────────────────────────
# SKIP_IF_EXISTS = True  →  skip any step whose output already exists on disk.
#                            Re-running the script after a crash/interruption
#                            will only compute what is missing.
# SKIP_IF_EXISTS = False →  honour OVERWRITE_* flags below (original behaviour).
SKIP_IF_EXISTS            = True
OVERWRITE_SORTED_FOLDER   = False   # only used when SKIP_IF_EXISTS = False
OVERWRITE_ANALYZER_FOLDER = False   # only used when SKIP_IF_EXISTS = False

# Extensions computed on the SortingAnalyzer, in order.
# Value = any extra kwargs beyond GLOBAL_JOB_KWARGS.
EXTENSIONS_TO_COMPUTE = {
    "random_spikes":        {},
    "waveforms":            {},
    "templates":            {},
    "noise_levels":         {},
    "unit_locations":       {},
    "correlograms":         {},
    "spike_amplitudes":     {},
    "principal_components": {"n_components": 5, "mode": "by_channel_local"},
    "spike_locations":      {},
    "template_metrics":     {},
    "quality_metrics":      {},
}


# -----------------------------------------------------------------------------
# Logging  (timestamps + flush so every line appears immediately)
# -----------------------------------------------------------------------------
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------
def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def save_figure(fig_or_obj, save_path):
    save_path = Path(save_path)
    if hasattr(fig_or_obj, "savefig"):
        fig = fig_or_obj
    elif hasattr(fig_or_obj, "figure") and hasattr(fig_or_obj.figure, "savefig"):
        fig = fig_or_obj.figure
    else:
        raise TypeError(f"Cannot save figure-like object of type: {type(fig_or_obj)}")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Skip-detection helpers
# -----------------------------------------------------------------------------
def _try_load_sorting(path):
    """Return a loaded Sorting from a sorter-result folder, or None on failure."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return si.read_sorter_folder(path)
    except Exception as exc:
        log(f"  [WARN] Could not load sorting from {path}: {exc}")
        return None


def _try_load_analyzer(path):
    """Return a loaded SortingAnalyzer, or None on failure."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return si.load_sorting_analyzer(path)
    except Exception as exc:
        log(f"  [WARN] Could not load analyzer from {path}: {exc}")
        return None


def _extension_on_disk(analyzer_path, ext_name):
    """
    Check whether a SortingAnalyzer extension exists on disk.
    """
    analyzer_path = Path(analyzer_path)

    possible_paths = [
        analyzer_path / "extensions" / ext_name,
        analyzer_path / ext_name,  # compatibility with alternate layouts
    ]

    return any(path.exists() for path in possible_paths)


# -----------------------------------------------------------------------------
# Analyzer extension computation  (per-extension skip check)
# -----------------------------------------------------------------------------

def extension_exists(analyzer, extension_name):
    try:
        return analyzer.has_extension(extension_name)
    except AttributeError:
        try:
            return extension_name in analyzer.get_computed_extensions()
        except Exception:
            return False
        
def compute_extensions(analyzer, job_kwargs, skip_existing=True):
    for ext_name, extra_kwargs in EXTENSIONS_TO_COMPUTE.items():
        if skip_existing and extension_exists(analyzer, ext_name):
            log(f"  [SKIP] Extension '{ext_name}' already exists")
            continue

        log(f"  Computing : {ext_name} …")
        t0 = time.time()

        try:
            analyzer.compute(
                ext_name,
                **extra_kwargs,
                **job_kwargs,
            )
            log(f"  Done      : {ext_name} ({time.time() - t0:.0f} s)")
        except Exception as e:
            log(f"  [WARN] Failed to compute '{ext_name}': {e}")


# -----------------------------------------------------------------------------
# Signal-processing helpers
# -----------------------------------------------------------------------------
def ttl_from_analog(signal, fs, hysteresis=0.1, filt_kernel=5):
    signal = np.asarray(signal).squeeze()

    if signal.size == 0:
        return np.array([], dtype=np.int8), np.array([], dtype=int), np.array([], dtype=int)

    if filt_kernel is not None and filt_kernel > 1 and len(signal) >= filt_kernel:
        if filt_kernel % 2 == 0:
            filt_kernel += 1
        signal_f = medfilt(signal, kernel_size=filt_kernel)
    else:
        signal_f = signal

    v_low  = np.percentile(signal_f, 5)
    v_high = np.percentile(signal_f, 95)

    if np.isclose(v_low, v_high):
        digital = (signal_f > v_high).astype(np.int8)
        d = np.diff(digital)
        return digital, np.where(d == 1)[0] + 1, np.where(d == -1)[0] + 1

    v_lo = v_low  + hysteresis * (v_high - v_low)
    v_hi = v_high - hysteresis * (v_high - v_low)

    above_hi     = signal_f > v_hi
    below_lo     = signal_f < v_lo
    hi_crossings = np.where(np.diff(above_hi.astype(np.int8)) == 1)[0] + 1
    lo_crossings = np.where(np.diff(below_lo.astype(np.int8)) == 1)[0] + 1

    changes = np.full(len(signal_f), np.nan)
    changes[0]            = 1.0 if signal_f[0] > v_hi else 0.0
    changes[hi_crossings] = 1.0
    changes[lo_crossings] = 0.0

    mask      = ~np.isnan(changes)
    last_seen = np.where(mask, np.arange(len(changes)), 0)
    np.maximum.accumulate(last_seen, out=last_seen)
    digital = changes[last_seen].astype(np.int8)

    d = np.diff(digital)
    return digital, np.where(d == 1)[0] + 1, np.where(d == -1)[0] + 1


def quadrature_speed_direction(sigA, sigB, fs, pulses_per_rev=900,
                               hysteresis=0.1, max_gap_s=0.020):
    sigA = np.asarray(sigA).squeeze()
    sigB = np.asarray(sigB).squeeze()

    sig_len   = min(len(sigA), len(sigB))
    sigA      = sigA[:sig_len]
    sigB      = sigB[:sig_len]
    time_full = np.arange(sig_len) / fs

    digA, rising_A, falling_A = ttl_from_analog(sigA, fs, hysteresis)
    digB, rising_B, falling_B = ttl_from_analog(sigB, fs, hysteresis)

    edges = np.concatenate([rising_A, falling_A, rising_B, falling_B])
    dirs  = np.concatenate([
        np.where(digB[rising_A]  == 0,  1, -1),
        np.where(digB[falling_A] == 1,  1, -1),
        np.where(digA[rising_B]  == 1,  1, -1),
        np.where(digA[falling_B] == 0,  1, -1),
    ])

    if len(edges) < 2:
        return time_full * 1000, np.zeros(sig_len), np.zeros(sig_len)

    sort_idx = np.argsort(edges, kind="stable")
    edges    = edges[sort_idx]
    dirs     = dirs[sort_idx]

    edges, unique_idx = np.unique(edges, return_index=True)
    dirs = dirs[unique_idx]

    if len(edges) < 2:
        return time_full * 1000, np.zeros(sig_len), np.zeros(sig_len)

    t_edges      = edges / fs
    deg_per_edge = 360.0 / (pulses_per_rev * 4)
    dt           = np.diff(t_edges)
    t_mid        = (t_edges[1:] + t_edges[:-1]) / 2

    valid = dt > 0
    if not np.any(valid):
        return time_full * 1000, np.zeros(sig_len), np.zeros(sig_len)

    dt_valid    = dt[valid]
    t_mid_valid = t_mid[valid]
    dirs_valid  = dirs[:-1][valid]
    speed_edges = (deg_per_edge * dirs_valid) / dt_valid

    long_gaps   = np.where(dt > max_gap_s)[0]
    t_zeros     = t_edges[long_gaps] + max_gap_s
    speed_zeros = np.zeros(len(long_gaps))
    dir_zeros   = np.zeros(len(long_gaps))

    t_mid_aug = np.concatenate([t_mid_valid, t_zeros])
    speed_aug = np.concatenate([speed_edges, speed_zeros])
    dir_aug   = np.concatenate([dirs_valid,  dir_zeros])

    sort      = np.argsort(t_mid_aug, kind="stable")
    t_mid_aug = t_mid_aug[sort]
    speed_aug = speed_aug[sort]
    dir_aug   = dir_aug[sort]

    t_mid_aug, uniq_idx = np.unique(t_mid_aug, return_index=True)
    speed_aug = speed_aug[uniq_idx]
    dir_aug   = dir_aug[uniq_idx]

    f_speed = interp1d(t_mid_aug, speed_aug, kind="previous",
                       bounds_error=False, fill_value=0)
    f_dir   = interp1d(t_mid_aug, dir_aug,   kind="previous",
                       bounds_error=False, fill_value=0)

    return time_full * 1000, f_speed(time_full), f_dir(time_full)


def extract_ttl_from_bit(digital_word, bit, sampling_rate,
                         plot_path=None, savename="ttl", plot=True):
    digital_word = np.asarray(digital_word).squeeze().astype(np.uint64, copy=False)

# Convert the extracted bit to a signed type before np.diff
    ttl_signal = ((digital_word >> bit) & 1).astype(np.int8)
    time_axis    = np.arange(len(ttl_signal)) / sampling_rate

    if plot and plot_path is not None:
        plt.figure(figsize=(15, 3))
        plt.plot(time_axis, ttl_signal, linewidth=1)
        plt.title(f"Isolated Bit {bit} ({savename})")
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.tight_layout()
        plt.savefig(Path(plot_path) / f"ttl_bit_{bit}_{savename}.png", dpi=200)
        plt.close()

    diff = np.diff(ttl_signal)

    rising_idx = np.flatnonzero(diff > 0) + 1
    falling_idx = np.flatnonzero(diff < 0) + 1

    edges = np.concatenate([
        rising_idx / sampling_rate,
        falling_idx / sampling_rate,
    ])

    types = (
        ["rising"] * len(rising_idx)
        + ["falling"] * len(falling_idx)
    )

    return (
        pd.DataFrame({"timestamps": edges, "edge_type": types})
        .sort_values("timestamps")
        .reset_index(drop=True)
    )


def extract_and_save_ttl_events(data, bit_name_pairs, save_path, plot_path, plot=True):
    save_path = ensure_dir(save_path)
    plot_path = ensure_dir(plot_path)

    try:
        digital_signals = data.get_traces(return_in_uV=False)
    except TypeError:
        digital_signals = data.get_traces(return_scaled=False)

    if digital_signals.ndim != 2 or digital_signals.shape[1] <= DIGITAL_WORD_CH:
        log(f"[WARN] Digital word channel {DIGITAL_WORD_CH} not available. Skipping TTL extraction.")
        return

    digital_word  = digital_signals[:, DIGITAL_WORD_CH]
    sampling_rate = data.get_sampling_frequency()

    for bit, savename in bit_name_pairs:
        ttl_df = extract_ttl_from_bit(digital_word, bit, sampling_rate,
                                      plot_path=plot_path, savename=savename, plot=plot)
        ttl_df.to_csv(save_path / f"{savename}.csv", index=False)


# -----------------------------------------------------------------------------
# Bombcell + spike times
# -----------------------------------------------------------------------------
def compute_bombcell_labels(analyzer, meta_path, plots_path):
    if not hasattr(si, "bombcell_label_units"):
        log("[WARN] spikeinterface.bombcell_label_units not available.")
        return None

    try:
        bc_results = si.bombcell_label_units(
            analyzer,
            thresholds=si.bombcell_get_default_thresholds(),
        )
    except Exception as e:
        log(f"[WARN] Bombcell labeling failed: {e}")
        return None

    bc_df = pd.DataFrame(bc_results).copy()

    if "bombcell_label" not in bc_df.columns:
        log("[WARN] 'bombcell_label' column not found in Bombcell results.")
        log(f"[WARN] Bombcell columns: {list(bc_df.columns)}")
        return None

    n_units = len(analyzer.unit_ids)

    if len(bc_df) != n_units:
        raise ValueError(
            f"Bombcell returned {len(bc_df)} rows, "
            f"but analyzer contains {n_units} units."
        )

    # This is the same mapping used in your previous code:
    #
    # labels[0] corresponds to spike_times.npy unit_index 0
    # labels[1] corresponds to spike_times.npy unit_index 1
    # etc.
    labels = bc_df["bombcell_label"].to_numpy()

    # Explicitly record the unit number used by to_spike_vector().
    bc_df.insert(
        0,
        "unit_index",
        np.arange(n_units, dtype=np.int64),
    )

    # Optional: retain the actual SpikeInterface unit ID too.
    bc_df.insert(
        1,
        "unit_id",
        list(analyzer.unit_ids),
    )

    bc_df.to_csv(
        Path(meta_path) / "bombcell_results.csv",
        index=False,
    )

    # This is equivalent to your previous code and preserves the same
    # unit ordering.
    analyzer.sorting.set_property(
        "bombcell_label",
        labels,
    )

    try:
        w = si.plot_unit_labels(
            analyzer,
            labels,
            ylims=(-300, 100),
        )
        save_figure(
            w,
            Path(plots_path) / "bombcell_labels.png",
        )
    except Exception as e:
        log(f"[WARN] Could not save Bombcell label plot: {e}")

    return labels


def save_spike_times(analyzer, recording, spike_times_path, labels=None):
    spike_times_path = ensure_dir(spike_times_path)

    spikes   = pd.DataFrame(analyzer.sorting.to_spike_vector())
    spike_df = spikes.drop(columns=["segment_index"], errors="ignore")
    sf_ap    = recording.get_sampling_frequency()

    if "sample_index" in spike_df.columns:
        spike_df["time_s"] = spike_df["sample_index"] / sf_ap

    if labels is not None:
        if "unit_index" in spike_df.columns:
            label_map = pd.Series(labels, index=np.arange(len(analyzer.unit_ids)))
            spike_df["label"] = spike_df["unit_index"].map(label_map)
        elif "unit_id" in spike_df.columns:
            label_map = pd.Series(labels, index=analyzer.unit_ids)
            spike_df["label"] = spike_df["unit_id"].map(label_map)

    spike_df.to_csv(spike_times_path / "spike_times.csv", index=False)
    np.save(spike_times_path / "spike_times.npy", spike_df.to_records(index=False))
    log(f"  Saved spike times → {spike_times_path}")

# -----------------------------------------------------------------------------
# Unit extremum channel + quality metrics
# -----------------------------------------------------------------------------

def save_unit_extremum_channels(analyzer, recording, output_csv):
    """
    Save the channel with the largest waveform/template extremum for each unit.

    The probe tip is assumed to be at the minimum y-coordinate in the probe
    contact geometry. Distance is measured along the probe y-axis.
    """
    output_csv = Path(output_csv)

    try:
        # For extracellular recordings, negative peaks are normally used.
        # outputs="id" returns channel IDs rather than channel indices.
        try:
            extremum_channels = si.get_template_extremum_channel(
                analyzer,
                peak_sign="neg",
                mode="extremum",
                outputs="id",
            )
        except TypeError:
            # Compatibility with older SpikeInterface versions that may not
            # support outputs="id".
            extremum_indices = si.get_template_extremum_channel(
                analyzer,
                peak_sign="neg",
                mode="extremum",
                outputs="index",
            )

            recording_channel_ids = list(recording.get_channel_ids())
            extremum_channels = {
                unit_id: recording_channel_ids[int(channel_index)]
                for unit_id, channel_index in extremum_indices.items()
            }

        probe = recording.get_probe()
        contact_positions = np.asarray(probe.contact_positions)

        if contact_positions.ndim != 2 or contact_positions.shape[1] < 2:
            raise ValueError(
                "Probe contact geometry does not contain 2-D positions."
            )

        recording_channel_ids = list(recording.get_channel_ids())

        # The probe tip is taken to be the minimum y-coordinate.
        tip_y = np.nanmin(contact_positions[:, 1])

        rows = []

        for unit_id in analyzer.unit_ids:
            channel_id = extremum_channels[unit_id]

            # Find the recording channel index corresponding to the channel ID.
            if channel_id in recording_channel_ids:
                channel_index = recording_channel_ids.index(channel_id)
            else:
                # Some SpikeInterface versions return a numeric channel index.
                channel_index = int(channel_id)

            if channel_index >= len(contact_positions):
                raise IndexError(
                    f"Channel index {channel_index} is outside probe geometry."
                )

            channel_position = contact_positions[channel_index]

            # Distance along the probe shaft from the tip.
            distance_from_tip_um = float(channel_position[1] - tip_y)

            rows.append({
                "unit_id": unit_id,
                "extremum_channel": channel_id,
                "dist_from_tip_um": distance_from_tip_um,
            })

        pd.DataFrame(
            rows,
            columns=[
                "unit_id",
                "extremum_channel",
                "dist_from_tip_um",
            ],
        ).to_csv(output_csv, index=False)

        log(f"  Saved unit extremum channels -> {output_csv}")

    except Exception as e:
        log(f"[WARN] Could not save unit extremum channel CSV: {e}")


def save_quality_metrics(analyzer, output_csv):
    """
    Save the quality_metrics SortingAnalyzer extension as Meta/metrics.csv.
    """
    output_csv = Path(output_csv)

    try:
        quality_metrics_extension = analyzer.get_extension("quality_metrics")
        metrics_df = quality_metrics_extension.get_data()

        if not isinstance(metrics_df, pd.DataFrame):
            metrics_df = pd.DataFrame(metrics_df)

        metrics_df = metrics_df.copy()
        metrics_df.index.name = "unit_id"
        metrics_df.reset_index().to_csv(output_csv, index=False)
        log(f"  Saved quality metrics -> {output_csv}")

    except Exception as e:
        log(f"[WARN] Could not save quality metrics CSV: {e}")
# -----------------------------------------------------------------------------
# Main per-folder pipeline
# -----------------------------------------------------------------------------
def process_folder(basefolder):
    base_path = Path(basefolder)
    if not base_path.exists():
        log(f"[SKIP] Folder does not exist: {base_path}")
        return

    log("=" * 60)
    log(f"Processing : {base_path}")
    log("=" * 60)

    meta_path        = ensure_dir(base_path / "Meta")
    plots_path       = ensure_dir(base_path / "plots")
    spike_times_path = ensure_dir(base_path / "spike_times")
    sorted_path      = base_path / "sorted"
    analyzer_path    = base_path / "analyzer"

    # New output files
    extremum_csv = meta_path / "unit_extremum_channel.csv"
    metrics_csv  = meta_path / "metrics.csv"

    # ── AP recording ──────────────────────────────────────────────────────────
    log("Reading AP recording …")
    try:
        recording = si.read_spikeglx(str(base_path), stream_id=AP_STREAM,
                                     load_sync_channel=False)
        sf_ap = recording.get_sampling_frequency()
        dur   = recording.get_num_frames() / sf_ap
        log(f"  {recording.get_num_channels()} ch  |  {dur:.1f} s  |  {sf_ap/1e3:.1f} kHz")
    except Exception as e:
        log(f"[ERROR] Could not read AP stream: {e}")
        return

    # ── Pre-processing ────────────────────────────────────────────────────────
    # (lazy — no data is actually read until compute() is called)
    log("Building preprocessing chain …")
    try:
        rec1 = si.bandpass_filter(recording)
        rec1 = si.phase_shift(rec1)
        log("  Detecting bad channels …")
        bad_channel_ids, _ = si.detect_bad_channels(rec1, method="coherence+psd")
        log(f"  Bad channels: {bad_channel_ids}")
        rec1 = si.interpolate_bad_channels(recording=rec1, bad_channel_ids=bad_channel_ids)
        rec1 = si.common_reference(rec1, operator="median", reference="global")
        log("  Preprocessing chain ready")
    except Exception as e:
        log(f"[ERROR] Preprocessing failed: {e}")
        return

    # ── CUDA info ─────────────────────────────────────────────────────────────
    if torch is not None:
        try:
            cuda_ok = torch.cuda.is_available()
            log(f"CUDA available: {cuda_ok}")
            if cuda_ok and torch.cuda.device_count() > 0:
                log(f"CUDA device   : {torch.cuda.get_device_name(0)}")
        except Exception as e:
            log(f"[WARN] Could not query CUDA info: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Sorting → Analyzer → Extensions
    #
    # Priority order when SKIP_IF_EXISTS = True:
    #   1. Analyzer folder exists  → load it directly (skip sorting entirely)
    #   2. Sorted folder exists    → load sorting, create analyzer
    #   3. Neither exists          → run sorter, create analyzer
    # ─────────────────────────────────────────────────────────────────────────
    analyzer    = None
    Sorting_KS4 = None

    # Step 1 — try to load existing analyzer ──────────────────────────────────
    if SKIP_IF_EXISTS and analyzer_path.exists():
        log(f"Loading existing analyzer : {analyzer_path}")
        analyzer = _try_load_analyzer(analyzer_path)
        if analyzer is not None:
            log("  [SKIP] Sorting + analyzer creation skipped (loaded from disk)")

    # Step 2 — need to build an analyzer ─────────────────────────────────────
    if analyzer is None:

        # Try loading existing sorting first
        if SKIP_IF_EXISTS and sorted_path.exists():
            log(f"Loading existing sorting  : {sorted_path}")
            Sorting_KS4 = _try_load_sorting(sorted_path)
            if Sorting_KS4 is not None:
                log(f"  [SKIP] Sorting skipped ({len(Sorting_KS4.unit_ids)} units loaded from disk)")

        # Run the sorter if we still don't have a Sorting object
        if Sorting_KS4 is None:
            if not SKIP_IF_EXISTS and OVERWRITE_SORTED_FOLDER:
                remove_dir(sorted_path)

            from spikeinterface.sorters import installed_sorters
            if SORTER_NAME not in installed_sorters():
                log(f"[ERROR] {SORTER_NAME} is not installed. Aborting folder.")
                return

            # ── Explicit pre-deletion in the main process ─────────────────────
            # Do NOT rely on remove_existing_folder=True inside run_sorter:
            # on Windows the sorter subprocess can silently hang on rmtree when
            # the folder is large or has stale file handles from a prior run.
            if sorted_path.exists():
                log(f"  Removing incomplete sorted folder (this may take a moment) …")
                try:
                    remove_dir(sorted_path)
                    log("  Removed OK")
                except Exception as e:
                    log(f"  [WARN] Could not remove {sorted_path}: {e} — KS4 will try anyway")

            # ── Prepare KS4 scratch dir ───────────────────────────────────────
            ks4_params = {}
            if KS4_SCRATCH_DIR is not None:
                scratch = Path(KS4_SCRATCH_DIR)
                try:
                    scratch.mkdir(parents=True, exist_ok=True)
                    ks4_params["scratch_dir"] = str(scratch)
                    log(f"  KS4 scratch dir : {scratch}")
                except Exception as e:
                    log(f"  [WARN] Could not create scratch dir {scratch}: {e} — using KS4 default")

            # ── Background watcher: print sorted/ size every 60 s ─────────────
            # KS4 runs in a subprocess; this is the only way to see it's alive.
            _stop_watcher = threading.Event()

            def _watch_sorted(path, stop):
                while not stop.wait(60):
                    if path.exists():
                        try:
                            files      = list(path.rglob("*"))
                            n_files    = len(files)
                            total_gb   = sum(
                                f.stat().st_size for f in files if f.is_file()
                            ) / 1e9
                            newest     = max(
                                (f for f in files if f.is_file()),
                                key=lambda f: f.stat().st_mtime,
                                default=None,
                            )
                            newest_str = newest.name if newest else "—"
                            log(f"  [KS4 ⟳] sorted/ → {n_files} files, "
                                f"{total_gb:.2f} GB, last written: {newest_str}")
                        except Exception:
                            pass

            watcher = threading.Thread(
                target=_watch_sorted, args=(sorted_path, _stop_watcher), daemon=True
            )
            watcher.start()
            log(f"Running {SORTER_NAME}  (watcher prints progress every 60 s) …")
            t0 = time.time()
            try:
                Sorting_KS4 = si.run_sorter(
                    sorter_name=SORTER_NAME,
                    recording=rec1,
                    folder=str(sorted_path),
                    remove_existing_folder=False,   # we already deleted it above
                    verbose=True,
                    **ks4_params,
                )
                log(f"  Sorting done — {len(Sorting_KS4.unit_ids)} units  ({time.time() - t0:.0f} s)")
            except Exception as e:
                log(f"[ERROR] Sorting failed: {e}")
                return
            finally:
                _stop_watcher.set()     # always stop the watcher thread

        # Create the analyzer
        if not SKIP_IF_EXISTS and OVERWRITE_ANALYZER_FOLDER:
            remove_dir(analyzer_path)

        log("Creating SortingAnalyzer …")
        t0 = time.time()
        try:
            analyzer = si.create_sorting_analyzer(
                Sorting_KS4,
                rec1,
                sparse=True,
                format="binary_folder",


                
                folder=str(analyzer_path),
            )
            log(f"  Analyzer created ({time.time() - t0:.0f} s)")
        except Exception as e:
            log(f"[ERROR] Analyzer creation failed: {e}")
            return

    # Step 3 — extensions (each checked individually) ─────────────────────────
    log("Computing analyzer extensions …")
    compute_extensions(analyzer, GLOBAL_JOB_KWARGS, SKIP_IF_EXISTS)

    # ── Unit extremum channels ─────────────────────────────────────────────────
    if SKIP_IF_EXISTS and extremum_csv.exists():
        log("[SKIP] Unit extremum channel CSV already exists")
    else:
        log("Saving unit extremum channels ...")
        save_unit_extremum_channels(
            analyzer=analyzer,
            recording=recording,
            output_csv=extremum_csv,
        )

    # ── Quality metrics ───────────────────────────────────────────────────────
    if SKIP_IF_EXISTS and metrics_csv.exists():
        log("[SKIP] Quality metrics CSV already exists")
    else:
        log("Saving quality metrics ...")
        save_quality_metrics(
            analyzer=analyzer,
            output_csv=metrics_csv,
        )

    # ── Bombcell ──────────────────────────────────────────────────────────────
    labels       = None
    bombcell_csv = meta_path / "bombcell_results.csv"

    if SKIP_IF_EXISTS and bombcell_csv.exists():
        log("[SKIP] Bombcell results already exist")

        try:
            existing_bc = pd.read_csv(bombcell_csv)

            if "bombcell_label" not in existing_bc.columns:
                raise ValueError(
                    "'bombcell_label' column is missing from Bombcell CSV"
                )

            if "unit_index" in existing_bc.columns:
                # Reload labels using the exact unit numbering stored in the CSV.
                labels = np.empty(len(analyzer.unit_ids), dtype=object)
                labels[:] = None

                for _, row in existing_bc.iterrows():
                    unit_index = int(row["unit_index"])
                    if 0 <= unit_index < len(labels):
                        labels[unit_index] = row["bombcell_label"]

                if any(label is None for label in labels):
                    raise ValueError(
                        "Bombcell CSV does not contain one valid row per unit"
                    )
            else:
                # Compatibility with an older Bombcell CSV.
                log(
                    "[WARN] Existing Bombcell CSV has no unit_index column. "
                    "Assuming its row order matches analyzer.unit_ids."
                )
                labels = existing_bc["bombcell_label"].to_numpy()

        except Exception as e:
            log(f"[WARN] Could not reload bombcell CSV: {e}")
            labels = None

    else:
        log("Running Bombcell …")
        labels = compute_bombcell_labels(
            analyzer,
            meta_path,
            plots_path,
        )

    # ── Spike times ───────────────────────────────────────────────────────────
    spike_csv = spike_times_path / "spike_times.csv"
    if SKIP_IF_EXISTS and spike_csv.exists():
        log("[SKIP] Spike times already saved")
    else:
        log("Saving spike times …")
        save_spike_times(analyzer, recording, spike_times_path, labels=labels)

    # ── NIDQ / Camera / TTLs / Speed ─────────────────────────────────────────
    speed_csv      = meta_path / "speed.csv"
    camttl_csv     = meta_path / "camttl.csv"
    ttl_csvs_exist = all((meta_path / f"{name}.csv").exists() for _, name in BIT_NAME_PAIRS)
    nidq_all_done  = speed_csv.exists() and camttl_csv.exists() and ttl_csvs_exist

    if SKIP_IF_EXISTS and nidq_all_done:
        log("[SKIP] All NIDQ outputs already exist")
    else:
        log("Processing NIDQ …")
        try:
            event       = si.read_spikeglx(str(base_path), stream_id=NIDQ_STREAM,
                                            load_sync_channel=False)
            sf_nidq     = event.get_sampling_frequency()
            channel_ids = list(event.get_channel_ids())

            max_needed = max(CAMERA_ANALOG_CH, SPEED_CH_A, SPEED_CH_B, DIGITAL_WORD_CH)
            if len(channel_ids) <= max_needed:
                log("[WARN] NIDQ does not have enough channels. Skipping.")
            else:
                # Camera TTL ──────────────────────────────────────────────────
                if not (SKIP_IF_EXISTS and camttl_csv.exists()):
                    log("  Camera TTL …")
                    cam_signal = event.get_traces(
                        channel_ids=[channel_ids[CAMERA_ANALOG_CH]]).squeeze()
                    _, _, falling_cam = ttl_from_analog(cam_signal, sf_nidq)
                    if len(falling_cam) > 0:
                        pd.DataFrame({"camttl": [falling_cam[0] / sf_nidq]}).to_csv(
                            camttl_csv, index=False)
                        log(f"  Saved camttl.csv")
                else:
                    log("  [SKIP] camttl.csv")

                # TTL bits ────────────────────────────────────────────────────
                if not (SKIP_IF_EXISTS and ttl_csvs_exist):
                    log("  Extracting TTL bits …")
                    extract_and_save_ttl_events(event, BIT_NAME_PAIRS, meta_path,
                                                plots_path, plot=True)
                    log("  Saved TTL CSVs")
                else:
                    log("  [SKIP] TTL CSVs")

                # Quadrature speed ────────────────────────────────────────────
                if not (SKIP_IF_EXISTS and speed_csv.exists()):
                    log("  Quadrature speed decode …")
                    t0   = time.time()
                    sigA = event.get_traces(channel_ids=[channel_ids[SPEED_CH_A]]).squeeze()
                    sigB = event.get_traces(channel_ids=[channel_ids[SPEED_CH_B]]).squeeze()
                    ts_ms, speed, direction = quadrature_speed_direction(
                        sigA, sigB, sf_nidq,
                        pulses_per_rev=PULSES_PER_REV,
                        hysteresis=0.1,
                        max_gap_s=0.020,
                    )
                    pd.DataFrame({"time_ms": ts_ms, "speed": speed,
                                  "direction": direction}).to_csv(speed_csv, index=False)
                    log(f"  Saved speed.csv  ({time.time() - t0:.0f} s)")

                    stride = max(1, len(speed) // 5000)
                    plt.figure(figsize=(10, 4))
                    plt.plot(ts_ms[::stride] / 1000, speed[::stride],
                             color="crimson", linewidth=1.5, alpha=0.8)
                    plt.title(f"Velocity — {base_path.name}", fontsize=12, pad=10)
                    plt.xlabel("Time (s)")
                    plt.ylabel("Speed (deg/s)")
                    plt.grid(True, linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    plot_save_path = plots_path / "speed_overview.png"
                    plt.savefig(plot_save_path, dpi=200, bbox_inches="tight")
                    plt.close()
                    log(f"  Saved speed_overview.png")
                else:
                    log("  [SKIP] speed.csv")

        except Exception as e:
            log(f"[WARN] NIDQ processing failed: {e}")

    log(f"=== Done: {base_path} ===\n")


# -----------------------------------------------------------------------------
# Batch runner
# -----------------------------------------------------------------------------
def run_batch(folders):
    for folder in folders:
        try:
            process_folder(folder)
        except Exception as e:
            log(f"[ERROR] Failed on {folder}: {e}")


if __name__ == "__main__":
    folders = [
        r"I:\Data\raw\8181\8181_naive"
  

    ]
    run_batch(folders)