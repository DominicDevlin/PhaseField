import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Config (easy to change)
# =========================
BASE_PHASE_DIR = "data/sig13low"   # where subfolders like "x-b-c" live
BASE_MASS_DIR  = "data/sig13low/mass"

X_SELECTOR = "*"       # "*" for all x; or e.g. "3" to restrict to x=3 only
FIXED_B    = "9"       # matches the middle token in both dir and mass filename
FIXED_C    = "12"      # matches the last token in both dir and mass filename

# Phase processing parameters
Y_MIN_CUTOFF = 1.5
BIN_WIDTH    = 0.015
Y_INIT       = 1.7

# Mass processing parameters
C2_STOP_THRESHOLD = 1e-3     # also stop mass curves the moment c2 < this (optional)
APPLY_C2_THRESHOLD = True    # set False to ignore the mass threshold; peak time still applies

# Plot labels/titles
PHASE_TITLE = "y_threshold vs Time (stops after first peak)"
MASS_TITLE  = "c2 mass vs Time (truncated to first-peak time from phase plot)"
XLABEL_TIME = "Time"
YLABEL_PHASE = "y_threshold (phase-style)"
YLABEL_MASS  = "c2 mass"


# =========================
# Helpers (phase)
# =========================
def iter_time_files_c1(directory):
    """
    Yield (time:int, path:str) for files named 'c1-<time>.dat' in a directory.
    """
    file_re = re.compile(r'^c1-(\d+)\.dat$')
    try:
        for fname in os.listdir(directory):
            m = file_re.match(fname)
            if m:
                t = int(m.group(1))
                yield t, os.path.join(directory, fname)
    except FileNotFoundError:
        return

def compute_y_threshold(filepath, y_min_cutoff=1.5, bin_width=0.02, y_init=1.7):
    """
    Mirrors the logic in your phase script's process_data_file:
      - restrict to y >= y_min_cutoff
      - create y-bins of width bin_width
      - find the lowest bin where ALL c1 values in that bin are < 0.5
      - return y_start - y_init (or np.nan if not found)
    """
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 0 or data.shape[1] != 3:
            return np.nan

        data = data[data[:, 1] >= y_min_cutoff]
        if data.shape[0] == 0:
            return np.nan

        y_max = np.max(data[:, 1])
        bin_edges = np.arange(y_min_cutoff, y_max + bin_width, bin_width)

        for i in range(len(bin_edges) - 1):
            y_start, y_end = bin_edges[i], bin_edges[i + 1]
            mask = (data[:, 1] >= y_start) & (data[:, 1] < y_end)
            pts = data[mask]
            if pts.shape[0] == 0:
                continue
            c1_vals = pts[:, 2]
            if np.all(c1_vals < 0.5):
                return y_start - y_init
        return np.nan
    except Exception:
        return np.nan


# =========================
# Helpers (mass)
# =========================
def parse_triplet_from_name(fname):
    """
    Parse x, b, c from a filename 'mass-x-b-c.txt'.
    Returns (x, b, c) as strings or None if not matched.
    """
    m = re.match(r"mass-(\d+)-(\d+)-(\d+)\.txt$", fname)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)

def load_mass_file(path, skip_interval=None):
    data = np.loadtxt(path, usecols=(0, 2))  # load only time and c2
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got {data.shape[1]}")

    t  = data[:, 0]
    c2 = data[:, 1]

    # sort by time
    order = np.argsort(t)
    t, c2 = t[order], c2[order]
    
    skip_interval=400
    # downsample: keep every N-th point
    if skip_interval and skip_interval > 1:
        t  = t[::skip_interval]
        c2 = c2[::skip_interval]

    return t, c2

def truncate_at_threshold(t, y, thresh):
    """
    Truncate arrays at the first index where y < thresh (exclude that point and beyond).
    If never below threshold, return arrays unchanged.
    """
    below = np.where(y < thresh)[0]
    if below.size > 0:
        cut = below[0]
        return t[:cut], y[:cut]
    return t, y

def truncate_at_time(t, y, t_max):
    """
    Keep data points with t <= t_max (inclusive).
    """
    if t.size == 0:
        return t, y
    cut = np.searchsorted(t, t_max, side="right")
    return t[:cut+2], y[:cut+2]


# =========================
# Combined main
# =========================
def main():
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']
    # 1) Discover parameter sets (x-b-c) from phase subdirectories and order them by legend_lbl
    subdir_glob = os.path.join(BASE_PHASE_DIR, f"{X_SELECTOR}-{FIXED_B}-{FIXED_C}")
    subdirs = [d for d in glob.glob(subdir_glob) if os.path.isdir(d)]

    entries = []  # list of dicts: {"x": int, "b": str, "c": str, "subdir": str, "legend_lbl": float}
    for sd in subdirs:
        label = os.path.basename(sd)  # e.g., "13-9-12"
        parts = label.split("-")
        if len(parts) != 3:
            continue
        x_str, b_str, c_str = parts
        if b_str != FIXED_B or c_str != FIXED_C:
            continue
        try:
            x_val = float(x_str)
        except ValueError:
            continue
        legend_lbl = x_val * 2.0 / 3.0
        entries.append({"x": x_val, "b": b_str, "c": c_str, "subdir": sd, "legend_lbl": legend_lbl})

    # sort by legend label ascending
    entries.sort(key=lambda e: e["legend_lbl"])

    if not entries:
        print(f"No matching phase subdirectories found under {BASE_PHASE_DIR} with pattern {X_SELECTOR}-{FIXED_B}-{FIXED_C}")
        return

    # Prepare a consistent color mapping across both figures
    default_colors = plt.rcParams.get('axes.prop_cycle', None)
    color_list = default_colors.by_key()['color'] if default_colors and 'color' in default_colors.by_key() else None
    if not color_list:
        color_list = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

    # 2) Build phase curves, determine first-peak times, and plot FIGURE 1
    stop_time_by_x = {}  # key: int(x), value: t_peak (float)
    plt.figure(figsize=(8, 6))
    for idx, e in enumerate(entries):
        subdir = e["subdir"]
        x_display = int(e["x"]) if e["x"].is_integer() else e["x"]
        color = color_list[idx % len(color_list)]

        # collect y_threshold vs time
        elongation_map = {}
        for t, fpath in sorted(iter_time_files_c1(subdir), key=lambda x: x[0]):
            y_thr = compute_y_threshold(
                fpath,
                y_min_cutoff=Y_MIN_CUTOFF,
                bin_width=BIN_WIDTH,
                y_init=Y_INIT
            )
            if not np.isnan(y_thr):
                elongation_map[float(t)] = float(y_thr)

        if not elongation_map:
            print(f"Warning: no valid y_threshold data in {subdir}")
            continue

        times_sorted = sorted(elongation_map.keys())
        yvals_sorted = [elongation_map[t] for t in times_sorted]
        max_y = np.max(yvals_sorted)
        # first index where y hits global max
        try:
            peak_idx = next(i for i, y in enumerate(yvals_sorted) if np.isclose(y, max_y, atol=1e-12))
        except StopIteration:
            peak_idx = len(yvals_sorted) - 1
        
        peak_idx = peak_idx + 1

        keep_times = times_sorted[:peak_idx + 1]
        yvals_kept = [elongation_map[t] for t in keep_times]
        t_peak = keep_times[-1]
        stop_time_by_x[int(float(e["x"]))] = t_peak  # store as int key for mass filename matching

        plt.plot(keep_times, yvals_kept, marker=markers[idx], linestyle='-', color=color, label=f"{e['legend_lbl']:g}", linewidth=3, markersize=7)

    plt.xlabel(XLABEL_TIME)
    plt.ylabel(YLABEL_PHASE)
    plt.title(PHASE_TITLE)
    plt.grid(False)
    plt.tight_layout()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(title="sigma (= x·2/3)")
    plt.show()

    # 3) Plot corresponding MASS curves truncated to the SAME stop time (and optional c2<thresh) — FIGURE 2
    plt.figure(figsize=(8, 6))
    for idx, e in enumerate(entries):
        color = color_list[idx % len(color_list)]
        x_int = int(float(e["x"]))  # mass filenames are integral x per your pattern
        t_peak = stop_time_by_x.get(x_int, None)
        mass_path = os.path.join(BASE_MASS_DIR, f"mass-{x_int}-{FIXED_B}-{FIXED_C}.txt")

        if t_peak is None:
            print(f"Skipping mass-{x_int}-{FIXED_B}-{FIXED_C}.txt: no peak time from phase plot.")
            continue
        if not os.path.isfile(mass_path):
            print(f"Missing mass file: {mass_path}")
            continue

        try:
            t, c2 = load_mass_file(mass_path)  # sorted by time
        except Exception as ex:
            print(f"Skipping {os.path.basename(mass_path)}: {ex}")
            continue

        # First, truncate by shared stop time
        t_cut, c2_cut = truncate_at_time(t, c2, t_peak)

        # Optional: also truncate if c2 < threshold (earliest condition wins)
        if APPLY_C2_THRESHOLD:
            t_cut, c2_cut = truncate_at_threshold(t_cut, c2_cut, C2_STOP_THRESHOLD)

        if t_cut.size == 0:
            print(f"{os.path.basename(mass_path)}: nothing to plot after truncation; skipping.")
            continue

        label = f"x={x_int} (t≤{t_peak:g})"
        plt.plot(t_cut, c2_cut, marker=markers[idx], linestyle='-', color=color, label=label, linewidth=3, markersize=7)

    plt.xlabel(XLABEL_TIME)
    plt.ylabel(YLABEL_MASS)
    plt.title(MASS_TITLE + (f" (and c2<{C2_STOP_THRESHOLD})" if APPLY_C2_THRESHOLD else ""))
    plt.grid(False)
    plt.tight_layout()
    plt.xlim(left=0)
    plt.ylim(0,1.02)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
