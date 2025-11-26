import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Config (Central Control)
# =========================
BASE_PHASE_DIR = "data/sig13low"   # where subfolders like "x-b-c" live
BASE_MASS_DIR  = "data/sig13low/mass"

# ---------------------------------------------------------
# SELECTOR SETTINGS
# Set the specific number for fixed parameters.
# Set "*" for the parameter you want to vary (the loop variable).
# ---------------------------------------------------------
SELECT_X = "13"    # Fixed X
SELECT_B = "9"     # Fixed B
SELECT_C = "*"     # "*" = This is the variable (c)

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
# Helpers (phase)
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
# Helpers (mass)
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

    # 1) Search for directories matching the pattern
    glob_pattern = f"{SELECT_X}-{SELECT_B}-{SELECT_C}"
    subdir_glob = os.path.join(BASE_PHASE_DIR, glob_pattern)
    subdirs = [d for d in glob.glob(subdir_glob) if os.path.isdir(d)]

    entries = [] 
    
    # 2) Parse found directories
    for sd in subdirs:
        label = os.path.basename(sd)  # e.g., "13-9-12"
        parts = label.split("-")
        if len(parts) != 3: continue
        
        x_str, b_str, c_str = parts

        # Verify strict matching (in case glob is too loose on some OS)
        if SELECT_X != "*" and x_str != SELECT_X: continue
        if SELECT_B != "*" and b_str != SELECT_B: continue
        if SELECT_C != "*" and c_str != SELECT_C: continue

        # Determine Legend Label and Sort Key dynamically
        try:
            x_val, b_val, c_val = float(x_str), float(b_str), float(c_str)
        except ValueError: continue

        # Logic: If varying X, use the sigma formula. If varying B or C, use raw number.
        if SELECT_X == "*":
            sort_val = x_val
            legend_lbl = f"{x_val * 2.0 / 3.0:.2f}" 
            var_name = "sigma (x*2/3)"
        elif SELECT_C == "*":
            sort_val = c_val
            legend_lbl = f"{int(c_val)}" # Assuming C is usually integer
            var_name = "c"
        elif SELECT_B == "*":
            sort_val = b_val
            legend_lbl = f"{int(b_val)}"
            var_name = "b"
        else:
            # Fallback if nothing is "*", should typically not happen given logic
            sort_val = 0
            legend_lbl = "single"
            var_name = "run"

        entries.append({
            "x": x_str, "b": b_str, "c": c_str, 
            "subdir": sd, 
            "legend_lbl": legend_lbl,
            "sort_val": sort_val
        })

    # Sort by the variable parameter
    entries.sort(key=lambda e: e["sort_val"])

    if not entries:
        print(f"No matching phase subdirectories found for pattern: {glob_pattern}")
        return

    # Dictionary to share peak time between Phase and Mass plots
    # Key = tuple(x, b, c) to be absolutely unique
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
            # find first index where y is close to max
            peak_idx = next(i for i, y in enumerate(yvals) if np.isclose(y, max_y, atol=1e-12))
        except StopIteration:
            peak_idx = len(yvals) - 1
        
        peak_idx += 1 # Include the peak point
        
        t_peak = times[min(peak_idx, len(times)-1)]
        
        # Store for mass plot
        stop_time_map[(e['x'], e['b'], e['c'])] = t_peak

        plt.plot(times[:peak_idx+1], yvals[:peak_idx+1], 
                 marker=markers[idx % len(markers)], linestyle='-', color=color, 
                 label=e['legend_lbl'], linewidth=3, markersize=7)

    plt.xlabel(XLABEL_TIME)
    plt.ylabel(YLABEL_PHASE)
    plt.title(f"y_threshold vs Time (Varying {var_name})")
    plt.legend(title=var_name)
    plt.grid(False)
    plt.tight_layout()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()


    # ==========================
    # FIGURE 2: Mass
    # ==========================
    plt.figure(figsize=(8, 6))
    
    for idx, e in enumerate(entries):
        color = color_list[idx % len(color_list)]
        
        # Unique key to retrieve stop time
        key = (e['x'], e['b'], e['c'])
        t_peak = stop_time_map.get(key)
        
        # Dynamic filename construction based on this specific entry
        mass_filename = f"mass-{e['x']}-{e['b']}-{e['c']}.txt"
        mass_path = os.path.join(BASE_MASS_DIR, mass_filename)

        if t_peak is None:
            continue # Skipped in phase plot (no data)
        
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
    plt.xlim(left=0)
    plt.ylim(0, 1.02)
    plt.show()

if __name__ == "__main__":
    main()