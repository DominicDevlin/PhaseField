import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Config (Central Control)
# =========================
BASE_PHASE_DIR = "data/diff-phase"   # where subfolders like "3-0.5" live
BASE_MASS_DIR  = "data/diff-phase/mass"

# ---------------------------------------------------------
# SELECTOR SETTINGS (Modified for 2 Variables)
# Format: VAR1-VAR2 (e.g., 3-0.5)
# Set the specific number/string for fixed parameters.
# Set "*" for the parameter you want to vary.
# ---------------------------------------------------------
SELECT_VAR1 = "15"     # Fixed First Variable (e.g., "3")
SELECT_VAR2 = "*"     # Variable Second Variable (e.g., "0.5", "1", "1.5")

# Phase processing parameters
Y_MIN_CUTOFF = 1.5
BIN_WIDTH    = 0.015
Y_INIT       = 1.7

# Mass processing parameters
C2_STOP_THRESHOLD = 1e-3     
APPLY_C2_THRESHOLD = True    

# Plot labels
XLABEL_TIME  = "Time"
YLABEL_PHASE = "y_threshold (phase-style)"
YLABEL_MASS  = "c2 mass"

# =========================
# Helpers (phase) - Unchanged
# =========================
def iter_time_files_c1(directory):
    """ Yield (time:int, path:str) for files named 'c1-<time>.dat'. """
    file_re = re.compile(r'^c1-(\d+)\.dat$')
    try:
        for fname in os.listdir(directory):
            m = file_re.match(fname)
            if m:
                yield int(m.group(1)), os.path.join(directory, fname)
    except FileNotFoundError:
        return

def compute_y_threshold(filepath, y_min_cutoff=1.5, bin_width=0.02, y_init=1.7):
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 0 or data.shape[1] != 3: return np.nan
        data = data[data[:, 1] >= y_min_cutoff]
        if data.shape[0] == 0: return np.nan

        y_max = np.max(data[:, 1])
        bin_edges = np.arange(y_min_cutoff, y_max + bin_width, bin_width)

        for i in range(len(bin_edges) - 1):
            y_start, y_end = bin_edges[i], bin_edges[i + 1]
            mask = (data[:, 1] >= y_start) & (data[:, 1] < y_end)
            pts = data[mask]
            if pts.shape[0] == 0: continue
            if np.all(pts[:, 2] < 0.5):
                return y_start - y_init
        return np.nan
    except Exception:
        return np.nan

# =========================
# Helpers (mass) - Unchanged
# =========================
def load_mass_file(path):
    data = np.loadtxt(path, usecols=(0, 2))  
    if data.ndim == 1: data = data.reshape(1, -1)
    t, c2 = data[:, 0], data[:, 1]
    order = np.argsort(t)
    return t[order], c2[order]

def truncate_at_threshold(t, y, thresh):
    below = np.where(y < thresh)[0]
    if below.size > 0:
        cut = below[0]
        return t[:cut], y[:cut]
    return t, y

def truncate_at_time(t, y, t_max):
    if t.size == 0: return t, y
    cut = np.searchsorted(t, t_max, side="right")
    return t[:cut+2], y[:cut+2]

# =========================
# Main
# =========================
def main():
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']
    
    # Generate colors based on standard cycle
    default_colors = plt.rcParams.get('axes.prop_cycle', None)
    color_list = default_colors.by_key()['color'] if default_colors else ["blue", "orange", "green", "red"]

    # 1) Search for directories matching the 2-variable pattern
    glob_pattern = f"{SELECT_VAR1}-{SELECT_VAR2}"
    subdir_glob = os.path.join(BASE_PHASE_DIR, glob_pattern)
    subdirs = [d for d in glob.glob(subdir_glob) if os.path.isdir(d)]

    entries = [] 
    
    # 2) Parse found directories
    for sd in subdirs:
        label = os.path.basename(sd)  # e.g., "3-0.5"
        parts = label.split("-")
        
        # CHANGED: Check for exactly 2 parts
        if len(parts) != 2: continue
        
        v1_str, v2_str = parts

        # Verify strict matching
        if SELECT_VAR1 != "*" and v1_str != SELECT_VAR1: continue
        if SELECT_VAR2 != "*" and v2_str != SELECT_VAR2: continue

        # Determine Legend Label and Sort Key dynamically
        try:
            v1_val = float(v1_str)
            v2_val = float(v2_str)
        except ValueError: continue

        # Logic: Determine which variable is changing to set legend labels
        if SELECT_VAR1 == "*":
            sort_val = v1_val
            legend_lbl = f"{v1_val:g}" 
            var_name = "Var 1"
        elif SELECT_VAR2 == "*":
            sort_val = v2_val
            legend_lbl = f"{v2_val:g}"
            var_name = "Var 2"
        else:
            # Fallback if specific file selected
            sort_val = 0
            legend_lbl = f"{v1_str}-{v2_str}"
            var_name = "Run"

        entries.append({
            "v1": v1_str, "v2": v2_str, 
            "subdir": sd, 
            "legend_lbl": legend_lbl,
            "sort_val": sort_val
        })

    # Sort by the variable parameter
    entries.sort(key=lambda e: e["sort_val"])

    if not entries:
        print(f"No matching subdirectories found for pattern: {glob_pattern}")
        return

    # Dictionary to share peak time between Phase and Mass plots
    # Key = tuple(v1, v2)
    stop_time_map = {} 

    # ==========================
    # FIGURE 1: Phase
    # ==========================
    plt.figure(figsize=(8, 6))
    
    for idx, e in enumerate(entries):
        color = color_list[idx % len(color_list)]
        
        elongation_map = {}
        for t, fpath in sorted(iter_time_files_c1(e["subdir"]), key=lambda x: x[0]):
            y_thr = compute_y_threshold(fpath, Y_MIN_CUTOFF, BIN_WIDTH, Y_INIT)
            if not np.isnan(y_thr):
                elongation_map[float(t)] = float(y_thr)

        if not elongation_map:
            print(f"Warning: no data in {e['subdir']}")
            continue

        times = sorted(elongation_map.keys())
        yvals = [elongation_map[t] for t in times]
        
        # Find peak
        max_y = np.max(yvals)
        try:
            peak_idx = next(i for i, y in enumerate(yvals) if np.isclose(y, max_y, atol=1e-12))
        except StopIteration:
            peak_idx = len(yvals) - 1
        
        peak_idx += 1 
        t_peak = times[min(peak_idx, len(times)-1)]
        
        # Store for mass plot using unique key
        stop_time_map[(e['v1'], e['v2'])] = t_peak

        plt.plot(times[:peak_idx+1], yvals[:peak_idx+1], 
                 marker=markers[idx % len(markers)], linestyle='-', color=color, 
                 label=e['legend_lbl'], linewidth=3, markersize=7)

    plt.xlabel(XLABEL_TIME)
    plt.ylabel(YLABEL_PHASE)
    plt.title(f"y_threshold vs Time (Varying {var_name})")
    plt.legend(title=var_name)
    plt.grid(False)
    plt.tight_layout()
    #plt.xlim(left=0,right=3100)
    plt.xscale("log")
    plt.ylim(bottom=0)
    plt.show()


    # ==========================
    # FIGURE 2: Mass
    # ==========================
    plt.figure(figsize=(8, 6))
    
    for idx, e in enumerate(entries):
        color = color_list[idx % len(color_list)]
        
        # Unique key to retrieve stop time
        key = (e['v1'], e['v2'])
        t_peak = stop_time_map.get(key)
        
        # CHANGED: Dynamic filename construction for 2 variables
        # Assuming format: mass-VAR1-VAR2.txt
        mass_filename = f"mass-{e['v1']}-{e['v2']}.txt"
        mass_path = os.path.join(BASE_MASS_DIR, mass_filename)

        if t_peak is None:
            continue 
        
        if not os.path.isfile(mass_path):
            print(f"Missing mass file: {mass_path}")
            continue

        try:
            t, c2 = load_mass_file(mass_path)
            
            # Truncate
            t, c2 = truncate_at_time(t, c2, t_peak)
            if APPLY_C2_THRESHOLD:
                t, c2 = truncate_at_threshold(t, c2, C2_STOP_THRESHOLD)

            if t.size > 0:
                plt.plot(t, c2, marker=markers[idx % len(markers)], linestyle='-', 
                         color=color, label=f"{e['legend_lbl']} (t≤{t_peak:g})", 
                         linewidth=3, markersize=7)
        except Exception as ex:
            print(f"Error reading {mass_filename}: {ex}")

    plt.xlabel(XLABEL_TIME)
    plt.ylabel(YLABEL_MASS)
    plt.title(f"Mass vs Time (Varying {var_name})")
    plt.legend(title=var_name)
    plt.grid(False)
    plt.tight_layout()
    # plt.xlim(left=0)
    plt.xscale("log")
    plt.ylim(0, 1.02)
    plt.show()

if __name__ == "__main__":
    main()