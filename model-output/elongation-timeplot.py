import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helpers for the new c1/c2 data
# -----------------------------

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
        # Expect columns: x, y, c1
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

# -----------------------------
# Main plotting script
# -----------------------------

def main():
    # Base directory and pattern of subfolders (same style as your old script)
    base_path = "data/sig13low/"
    pattern = "15-9-*"
    subdir_pattern = os.path.join(base_path, pattern)
    subdirectories = glob.glob(subdir_pattern)

    # Build (legend_lbl, subdir) pairs and sort by legend_lbl ascending
    entries = []
    for subdir in subdirectories:
        subdir_label = os.path.basename(subdir)
        parts = subdir_label.split("-")
        try:
            sigma_input = float(parts[2])
        except ValueError:
            continue
        legend_lbl = sigma_input * 2.0 / 3.0
        entries.append((legend_lbl, subdir))
    entries.sort(key=lambda x: x[0])  # smallest -> largest

    plt.figure(figsize=(8, 6))

    for legend_lbl, subdir in entries:
        # Collect y_threshold vs time for this subdir
        elongation_map = {}
        for t, fpath in sorted(iter_time_files_c1(subdir), key=lambda x: x[0]):
            y_thr = compute_y_threshold(
                fpath,
                y_min_cutoff=1.5,  # same defaults as your phase script
                bin_width=0.015,
                y_init=1.7
            )
            if not np.isnan(y_thr):
                elongation_map[float(t)] = float(y_thr)

        if not elongation_map:
            continue

        # ---- Trim after first global maximum (post-process elongation_map) ----
        times_sorted = sorted(elongation_map.keys())
        yvals_sorted = [elongation_map[t] for t in times_sorted]

        max_y = np.max(yvals_sorted)
        # first index where y hits the global max (tolerance for float noise)
        peak_idx = next(i for i, y in enumerate(yvals_sorted) if np.isclose(y, max_y, atol=1e-9))

        # keep only entries up to and including the first peak
        keep_times = times_sorted[:peak_idx + 1]
        times = keep_times
        yvals = [elongation_map[t] for t in keep_times]

        plt.plot(times, yvals, marker='o', linestyle='-', label=f"{legend_lbl:g}")

    plt.xlabel("Time")
    plt.ylabel("y_threshold (phase-style)")
    plt.title("y_threshold vs Time (stops after first peak)")
    plt.grid(False)
    plt.tight_layout()
    plt.xlim(left=0)  # keep open-ended right limit
    plt.legend(title="sigma")
    # plt.xlim(0,500)
    # plt.ylim(0,0.5)
    plt.show()

if __name__ == "__main__":
    main()
