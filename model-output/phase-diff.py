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
main_directory = 'data/diff-phase'

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
            return np.nan
        data = data[data[:, 1] >= y_min_cutoff]
        if data.shape[0] == 0:
            return np.nan
        y_max_data = np.max(data[:, 1])
        bin_edges = np.arange(y_min_cutoff, y_max_data + bin_width, bin_width)
        for i in range(len(bin_edges) - 1):
            y_start, y_end = bin_edges[i], bin_edges[i + 1]
            mask = (data[:, 1] >= y_start) & (data[:, 1] < y_end)
            points_in_bin = data[mask]
            if points_in_bin.shape[0] > 0:
                if np.all(points_in_bin[:, 2] < 0.5):
                    return y_start - y_init
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


def generate_phase_diagram(root_directory, fixed_gamma13=0.06, fixed_gamma23=0.06, vmax=2.5):
    """
    Processes parameter directories to plot a phase diagram of g12 vs. diff.
    The theoretical wetting regimes are shown as vertical bands based on the
    fixed g13 and g23 values.
    """
    dir_pattern = re.compile(r'^(\d+)-([\d.]+)$')
    results = []
    print(f"Scanning root directory: {root_directory}")

    for dir_name in os.listdir(root_directory):
        full_path = os.path.join(root_directory, dir_name)
        if not os.path.isdir(full_path):
            continue
        
        m = dir_pattern.match(dir_name)
        if not m:
            continue

        try:
            g12_raw = int(m.group(1))
            # --- FIX: The second variable is `diff`, as you correctly stated ---
            diff = float(m.group(2))
        except ValueError:
            print(f"Warning: Could not parse numbers from directory name '{dir_name}'. Skipping.")
            continue
        
        # Scale g12 according to the simulation setup
        g12 = (2/3) * g12_raw / 100.0
        print(f"Processing directory: {dir_name} -> (g12={g12:.4f}, diff={diff})")

        # Scan ALL c1-*.dat files and take the MAX y_threshold across times
        y_thresholds = []
        for _, fpath in sorted(iter_time_files_c1(full_path), key=lambda x: x[0]):
            y_thr = process_data_file(
                fpath, y_min_cutoff=Y_MIN_CUTOFF, bin_width=BIN_WIDTH, y_init=Y_INIT
            )
            if not np.isnan(y_thr):
                y_thresholds.append(float(y_thr))

        if not y_thresholds:
            print(f"  -> No valid y_threshold from any c1-*.dat in {dir_name}")
            continue

        max_y_value = float(np.max(y_thresholds))
        results.append((g12, diff, max_y_value))

    if not results:
        print("\nNo data could be processed. Exiting.")
        return

    data_points = np.array(results)
    g12_vals = data_points[:, 0]
    diff_vals = data_points[:, 1] # --- FIX: This is the 'diff' variable ---
    y_color_vals = data_points[:, 2]

    # --- FIX: Create theoretical wetting regimes as vertical lines/regions ---
    print("\nCreating theoretical wetting background...")

    # Calculate the single critical value for g12 where dewetting begins
    g12_critical_dewet = fixed_gamma13 + fixed_gamma23

    # Define plot boundaries with a small margin
    g12_min, g12_max = g12_vals.min() - 0.01, g12_vals.max() + 0.01
    diff_min, diff_max = diff_vals.min() - 0.2, diff_vals.max() + 0.2
    
    # --- Plot ---
    print("Generating combined phase diagram plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- FIX: Use axvspan to shade the background regions ---
    # Partial Wetting Region (g12 < g13 + g23)
    ax.axvspan(g12_min, g12_critical_dewet, color='#d3d3d3', alpha=0.6, zorder=0,
               label=f'Partial Wetting ($\\gamma_{{12}} < {g12_critical_dewet:.2f}$)')
    
    # Complete Dewetting Region (g12 > g13 + g23)
    ax.axvspan(g12_critical_dewet, g12_max, color='#ffcccc', alpha=0.6, zorder=0,
               label=f'Dewetting ($\\gamma_{{12}} > {g12_critical_dewet:.2f}$)')

    # Plot the simulation data points on top
    scatter = ax.scatter(
        g12_vals, diff_vals, # --- FIX: Plot g12 vs diff ---
        c=y_color_vals,
        cmap='viridis',
        s=250,
        edgecolors='k',
        zorder=10,
        vmin=0,
        vmax=vmax
    )

    # Colorbar & labels
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.9)
    cbar.set_label('Max Interface Height (y_threshold)', fontsize=12, weight='bold')
    
    # Add a legend for the background regions
    ax.legend(title='Theoretical Wetting Regimes', loc='upper left')

    ax.set_xlabel('$\\gamma_{12}$ (Surface Tension 1-2)', fontsize=14, weight='bold')
    # --- FIX: Y-axis is now correctly labeled 'diff' ---
    ax.set_ylabel('diff', fontsize=14, weight='bold')
    # --- FIX: Title reflects the fixed constants ---
    ax.set_title(f'Phase Diagram for $\\gamma_{{13}} = {fixed_gamma13}$ and $\\gamma_{{23}} = {fixed_gamma23}$',
                 fontsize=16, weight='bold')

    ax.set_xlim(g12_min, g12_max)
    ax.set_ylim(diff_min, diff_max)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Call the function with the correct fixed values for g13 and g23.
    generate_phase_diagram(
        main_directory,
        fixed_gamma13=0.06,
        fixed_gamma23=0.06,
        vmax=2.2
    )