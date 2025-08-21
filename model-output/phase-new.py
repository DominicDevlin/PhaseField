import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# --- Match timedata.py's phase processing parameters ---
Y_MIN_CUTOFF = 1.5
BIN_WIDTH    = 0.015
Y_INIT       = 1.7

# --- (The find_latest_file and process_data_file functions are unchanged) ---

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


# --- Helpers adopted from timedata.py ---
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


def generate_phase_diagram(root_directory, fixed_gamma13=6, vmax=2.5):
    """
    Process all parameter directories under root_directory, compute the
    MAX y_threshold across all c1-<time>.dat files in each directory,
    and plot a phase diagram colored by that maximum (like timedata.py).
    """
    dir_pattern = re.compile(r'^(\d+)-(\d+)-(\d+)$')
    results = []
    print(f"Scanning root directory: {root_directory}")

    for dir_name in os.listdir(root_directory):
        full_path = os.path.join(root_directory, dir_name)
        if not os.path.isdir(full_path):
            continue

        match = dir_pattern.match(dir_name)
        if not match:
            continue

        g12_raw, g13_raw, g23_raw = map(int, match.groups())  # g13 from folder name
        g12 = (2/3) * g12_raw / 100.0
        g13 = (2/3) * g13_raw / 100.0
        g23 = (2/3) * g23_raw / 100.0
        print(f"Processing directory: {dir_name} -> (g12={g12}, g23={g23})")

        # --- NEW: scan ALL c1-*.dat files and take the MAX y_threshold across times ---
        y_thresholds = []
        # We can reuse process_data_file but pass params to match timedata.py
        for _, fpath in sorted(iter_time_files_c1(full_path), key=lambda x: x[0]):
            y_thr = process_data_file(
                fpath,
                y_min_cutoff=Y_MIN_CUTOFF,
                bin_width=BIN_WIDTH,
                y_init=Y_INIT
            )
            if not np.isnan(y_thr):
                y_thresholds.append(float(y_thr))

        if len(y_thresholds) == 0:
            print(f"  -> No valid y_threshold from any c1-*.dat in {dir_name}")
            continue

        max_y_value = float(np.max(y_thresholds))  # MAX across all times
        results.append((g12, g23, max_y_value))

    if not results:
        print("\nNo data could be processed. Exiting.")
        return

    data_points = np.array(results)
    g12_vals = data_points[:, 0]
    g23_vals = data_points[:, 1]
    y_color_vals = data_points[:, 2]

    # --- Create the theoretical wetting regime field ---
    print("\nCreating theoretical wetting background...")

    g12_min, g12_max = g12_vals.min() - 0.01, g12_vals.max() + 0.01
    g23_min, g23_max = g23_vals.min() - 0.01, g23_vals.max() + 0.01

    g12_grid, g23_grid = np.meshgrid(
        np.linspace(g12_min, g12_max, 500),
        np.linspace(g23_min, g23_max, 500)
    )

    # Use the g13 value from the function arg for the background (fixed line)
    regime_grid = np.zeros_like(g12_grid, dtype=int)
    regime_grid[g12_grid > fixed_gamma13*(2/3)/100.0 + g23_grid] = 1
    regime_grid[fixed_gamma13*(2/3)/100.0 > g12_grid + g23_grid] = 2
    regime_grid[g23_grid > g12_grid + fixed_gamma13*(2/3)/100.0] = 3

    regime_colors = ['#d3d3d3', '#ffcccc', '#d4edda', '#cce5ff']  # Grey, Red, Blue, Green
    cmap_regimes = mcolors.ListedColormap(regime_colors)

    # --- Plot ---
    print("Generating combined phase diagram plot...")
    fig, ax = plt.subplots(figsize=(9, 9))

    ax.imshow(
        regime_grid,
        origin='lower',
        extent=[g12_min, g12_max, g23_min, g23_max],
        cmap=cmap_regimes
    )

    scatter = ax.scatter(
        g12_vals, g23_vals,
        c=y_color_vals,
        cmap='viridis',
        s=1000,
        edgecolors='k',
        zorder=10,
        vmin=0,
        vmax=vmax
    )

    # Colorbar & labels
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Max y_threshold across time (c1 < 0.5 bin)', fontsize=12, weight='bold')

    legend_labels = {
        0: 'Partial Wetting / Dewetting',
        1: 'Droplet Dewets from Solid ($\\gamma_{12} > \\gamma_{13} + \\gamma_{23}$)',
        2: 'Droplet Wets Solid ($\\gamma_{13} > \\gamma_{12} + \\gamma_{23}$)',
        3: 'Solid Wets Droplet ($\\gamma_{23} > \\gamma_{12} + \\gamma_{13}$)'
    }
    patches = [mpatches.Patch(color=regime_colors[i], label=label) for i, label in legend_labels.items()]
    # ax.legend(handles=patches, bbox_to_anchor=(1.05, 0.6), loc='upper left',
    #           title='Theoretical Wetting Regimes', fontsize=10)

    ax.set_xlabel('$\\gamma_{12}$ (Surface Tension 1-2)', fontsize=14, weight='bold')
    ax.set_ylabel('$\\gamma_{23}$ (Surface Tension 2-3)', fontsize=14, weight='bold')
    ax.set_title(f'Phase Diagram of Wetting Behavior (for $\\gamma_{{13}} = {fixed_gamma13}$)', fontsize=16, weight='bold')

    ax.set_xlim(g12_min, g12_max)
    ax.set_ylim(g23_min, g23_max)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_aspect('equal', adjustable='box')

    plt.xticks(np.arange(0.02, 0.18, 0.02))
    plt.yticks(np.arange(0.02, 0.18, 0.02))

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


if __name__ == '__main__':
    main_directory = 'data/sig13low'
    generate_phase_diagram(main_directory, fixed_gamma13=12, vmax=2.2)
