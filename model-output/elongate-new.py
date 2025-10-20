import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# --- Match timedata.py's phase processing parameters ---
Y_MIN_CUTOFF = 1.5
BIN_WIDTH    = 0.015
Y_INIT       = 1.7
main_directory = 'data/diff-phase'

# ----------------------------- Unchanged helpers -----------------------------

def find_latest_file(directory):
    """
    Finds the data file corresponding to the latest time in a given directory.
    Files are expected to be in the format 'c1-TIME.dat'.
    """
    latest_time = -1
    latest_file = None
    file_pattern = re.compile(r'c1-(\d+)\.dat$')
    try:
        for filename in os.listdir(directory):
            match = file_pattern.match(filename)
            if match:
                time = int(match.group(1))
                if time > latest_time:
                    latest_time = time
                    latest_file = os.path.join(directory, filename)
    except FileNotFoundError:
        print(f"Warning: Directory not found: {directory}")
        return None
    if latest_file is None:
        print(f"Warning: No 'c1-*.dat' files found in {directory}")
    return latest_file


def process_data_file(filepath, y_min_cutoff=1.5, bin_width=0.05, y_init=1.7):
    """
    Discretizes y-space into bins to handle adaptive mesh data. Finds the
    lowest y-bin (>= y_min_cutoff) for which all c1 values of data points
    within that bin are less than 0.5.

    Returns:
        float: (y_start - y_init) for the first qualifying bin, or np.nan.
    """
    try:
        data = np.loadtxt(filepath)

        if data.ndim == 0 or data.shape[1] != 3:
            print(f"Warning: Invalid or empty data file: {filepath}")
            return np.nan

        # y >= cutoff
        data = data[data[:, 1] >= y_min_cutoff]
        if data.shape[0] == 0:
            print(f"  -> No data points with y >= {y_min_cutoff} found in {filepath}.")
            return np.nan

        # Bin edges
        y_max_data = np.max(data[:, 1])
        bin_edges = np.arange(y_min_cutoff, y_max_data + bin_width, bin_width)

        # Find first bin (from bottom) where ALL c1 < 0.5
        for i in range(len(bin_edges) - 1):
            y_start, y_end = bin_edges[i], bin_edges[i + 1]
            mask = (data[:, 1] >= y_start) & (data[:, 1] < y_end)
            points_in_bin = data[mask]
            if points_in_bin.shape[0] > 0:
                c1_vals = points_in_bin[:, 2]
                if np.all(c1_vals < 0.5):
                    return y_start - y_init

        print(f"  -> No y-bin found where all c1 < 0.5 in {filepath}.")
        return np.nan

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return np.nan


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

# ------------------------- New: multi-line elongation plot -------------------------

def generate_elongation_lines(root_directory, filter_gamma13=None, group_tol=1e-9):
    """
    Scan all parameter directories under root_directory, compute the MAX elongation
    (formerly 'color') across all c1-<time>.dat files per directory, then plot:

      x-axis:  gamma_12  (original phase-diagram x)
      y-axis:  elongation length = max_y_threshold_across_time
      lines:   one line per distinct gamma_23 (original phase-diagram y),
               connecting points that share the same gamma_23, sorted by gamma_12

    Args:
        root_directory (str): path containing parameter subfolders "g12_raw-diff".
                              The 'diff' part can now be an integer or a float.
        filter_gamma13 (float or None): if given, only include points whose gamma_13
            (scaled) is within group_tol of this value.
        group_tol (float): tolerance used for grouping floats originating from ints.
    """
    # --- FIX START ---
    # MODIFIED: Allow 'diff' to be a float (e.g., "0.5") by matching digits and dots.
    # The original r'^(\d+)-(\d+)$' only matched integers.
    dir_pattern = re.compile(r'^(\d+)-([\d.]+)$')
    # --- FIX END ---
    results = []
    print(f"Scanning root directory: {root_directory}")

    for dir_name in os.listdir(root_directory):
        full_path = os.path.join(root_directory, dir_name)
        if not os.path.isdir(full_path):
            continue
        m = dir_pattern.match(dir_name)
        
        if not m:
            continue

        # --- FIX START ---
        # MODIFIED: Cast g12_raw to int and diff to float separately.
        # The original `map(int, m.groups())` would fail on "0.5".
        try:
            g12_raw = int(m.group(1))
            diff = float(m.group(2))
        except ValueError:
            print(f"Warning: Could not parse numbers from directory name '{dir_name}'. Skipping.")
            continue
        # --- FIX END ---

        print(f"Processing directory: {dir_name} -> g12_raw={g12_raw}, diff={diff}")

        # Scale like your original script
        scale = (2/3) / 100.0
        g12 = g12_raw * scale

        # if (filter_gamma13 is not None) and (not np.isclose(g13, filter_gamma13, atol=group_tol)):
        #     continue

        # Collect max elongation across all available times for this directory
        y_thresholds = []
        for _, fpath in sorted(iter_time_files_c1(full_path), key=lambda x: x[0]):
            y_thr = process_data_file(fpath, y_min_cutoff=Y_MIN_CUTOFF, bin_width=BIN_WIDTH, y_init=Y_INIT)
            if not np.isnan(y_thr):
                y_thresholds.append(float(y_thr))

        if len(y_thresholds) == 0:
            print(f"  -> No valid y_threshold from any c1-*.dat in {dir_name}")
            continue

        max_elong = float(np.max(y_thresholds))
        results.append((g12, diff, max_elong))

    if not results:
        print("\nNo data could be processed. Exiting.")
        return

    # Convert to array for convenience
    data = np.array(results)  # columns: g12, diff, elong
    g12_vals = data[:, 0]
    diff = data[:, 1]
    elong_vals = data[:, 2]

    # Group by (approximately) equal diff
    # Use rounding to a fixed number of decimals based on your quantization step
    keys = np.round(diff, 6)
    unique_keys = np.unique(keys)

    # Prepare plot
    fig, ax = plt.subplots(figsize=(9, 6))

    # For each unique g23/diff line, pull all matching points and plot as a line with markers
    for k in np.sort(unique_keys):
        mask = (keys == k)
        x_line = g12_vals[mask]
        y_line = elong_vals[mask]

        # Sort by x (gamma_12) so lines connect left->right
        order = np.argsort(x_line)
        x_line = x_line[order]
        y_line = y_line[order]

        # Label with the original y of the phase diagram (diff)
        label_val = f"{k:g}"
        label = rf"diff = ${label_val}$"

        # --- MODIFICATION START ---

        # We need at least 3 points to create a smooth cubic spline curve.
        # If we have fewer, we'll just plot the original points and line.
        if len(x_line) > 2:
            # 1. Create a smooth interpolation function
            cs = CubicSpline(x_line, y_line)

            # 2. Generate a dense set of x-points for a smooth line
            x_smooth = np.linspace(x_line.min(), x_line.max(), 300)
            y_smooth = cs(x_smooth)

            # 3. Plot the smooth line (without markers) and get its color
            # The comma in `line,` unpacks the single-element list returned by plot
            line, = ax.plot(x_smooth, y_smooth, linewidth=1.5, label=label)

            # 4. Plot the original data points on top with larger markers
            # We use linestyle='none' and specify the color to match the smooth line.
            ax.plot(x_line, y_line, marker='o', linestyle='none', markersize=8, color=line.get_color())
        else:
            # Fallback for lines with too few points to interpolate
            ax.plot(x_line, y_line, marker='o', linewidth=1.5, markersize=8, label=label)

    # Axes/labels
    ax.set_xlabel(r"$\gamma_{12}$ (Surface Tension 1–2)", fontsize=13, weight='bold')
    ax.set_ylabel("Elongation length (Δy from threshold bin)", fontsize=13, weight='bold')
    title = "Elongation vs $\\gamma_{12}$ (lines grouped by diff)"
    if filter_gamma13 is not None:
        title += rf"   [filtered: $\gamma_{{13}} \approx {filter_gamma13:.4f}$]"
    ax.set_title(title, fontsize=15, weight='bold')

    # Nice bounds
    ax.set_xlim(g12_vals.min() - 0.01, g12_vals.max() + 0.01)
    ymin = np.nanmin(elong_vals)
    ymax = np.nanmax(elong_vals)
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.legend(title=r"Lines: distinct diff values", loc='best', fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage:
    #   - plot all data grouped by the 'diff' value from the directory name.
    #   - OR only keep a slice at a particular γ13 (uncomment and set value)
    #
    # generate_elongation_lines(main_directory)
    #
    # If you want to only include points with γ13 ≈ 0.06 (for example):
    # generate_elongation_lines(main_directory, filter_gamma13=0.06)
    generate_elongation_lines(main_directory)