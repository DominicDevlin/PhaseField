import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# --- NEW FUNCTION TO FIND THE CORRECT FILE BASED ON DATA ---

def find_analysis_file(directory):
    """
    Finds the most recent data file that meets a specific physical criterion.

    The logic is as follows:
    1. Scan the directory for all 'c1-TIME.dat' files.
    2. Sort them by time in descending order (latest first).
    3. Iterate backwards from the latest time.
    4. For each time, load the corresponding 'c1-TIME.dat' and 'c2-TIME.dat' files.
    5. Check if for any data point, the sum of concentrations (c1 + c2) is > 0.8.
    6. The first file (i.e., the most recent one) that satisfies this condition
       is considered the correct file for analysis.
    7. Return the path to this 'c1' file. If no file satisfies the condition, return None.

    Args:
        directory (str): The path to the simulation data directory.

    Returns:
        str or None: The full path to the qualifying c1 data file, or None if none qualify.
    """
    file_pattern = re.compile(r'c1-(\d+)\.dat$')
    timed_files = []
    try:
        for filename in os.listdir(directory):
            match = file_pattern.match(filename)
            if match:
                time = int(match.group(1))
                timed_files.append((time, filename))
    except FileNotFoundError:
        print(f"Warning: Directory not found: {directory}")
        return None

    if not timed_files:
        print(f"Warning: No 'c1-*.dat' files found in {directory}")
        return None

    # Sort files by time, descending (latest first)
    timed_files.sort(key=lambda x: x[0], reverse=True)

    # Iterate backwards through time to find the first valid file
    for time, c1_filename in timed_files:
        c1_filepath = os.path.join(directory, c1_filename)
        # We need the corresponding c2 file to check the condition
        c2_filepath = os.path.join(directory, f'c2-{time}.dat')

        if not os.path.exists(c2_filepath):
            # If the c2 partner doesn't exist, we can't check, so we skip
            print(f"  -> Skipping time {time}, c2 file not found: {c2_filepath}")
            continue

        try:
            # Load the third column (concentration) from both files
            c1_vals = np.loadtxt(c1_filepath, usecols=2)
            c2_vals = np.loadtxt(c2_filepath, usecols=2)

            # Ensure data is not empty and shapes match
            if c1_vals.ndim == 0 or c1_vals.shape != c2_vals.shape:
                print(f"  -> Warning: Skipping time {time}, data is empty or mismatched.")
                continue

            # Check the condition: is there ANY point where c1+c2 > 0.8?
            if np.any((c1_vals >0.3) & (c2_vals > 0.3)):
                print(f"  -> Found qualifying file at time {time}. Using: {c1_filepath}")
                return c1_filepath  # This is the file we want
            
            if (np.all(c2_vals < 0.9)):
                print(f" -> Fully differentiated qualifying file at time {time}. Using: {c1_filepath}")

        except Exception as e:
            print(f"  -> Error processing files for time {time} in {directory}: {e}")
            continue

    # If the loop completes, no file met the criteria
    print(f"  -> No file found in {directory} where any(c1+c2 > 0.8). Returning final file")
    time, c1_filename = timed_files[0]
    c1_filepath = os.path.join(directory, c1_filename)
    return c1_filepath


# --- (The process_data_file function is unchanged) ---
    
def process_data_file(filepath, y_min_cutoff=1.5, bin_width=0.05, y_init=1.7):
    """
    Discretizes y-space into bins to handle adaptive mesh data. Finds the
    lowest y-bin (>= y_min_cutoff) for which all c1 values of data points
    within that bin are less than 0.5.

    Args:
        filepath (str): The path to the c1 data file.
        y_min_cutoff (float): The minimum y-value to start the analysis from.
        bin_width (float): The height of each y-bin for discretization.

    Returns:
        float: The starting y-coordinate of the lowest qualifying bin, or np.nan.
    """
    try:
        data = np.loadtxt(filepath)
        
        if data.ndim == 0 or data.shape[1] != 3:
            print(f"Warning: Invalid or empty data file: {filepath}")
            return np.nan

        # Step 1: Filter data to the relevant y-range (y >= 1.5)
        data = data[data[:, 1] >= y_min_cutoff]

        if data.shape[0] == 0:
            print(f"  -> No data points with y >= {y_min_cutoff} found in {filepath}.")
            return np.nan

        # Step 2: Define the y-bins for the analysis
        y_max_data = np.max(data[:, 1])
        bin_edges = np.arange(y_min_cutoff, y_max_data + bin_width, bin_width)

        # Step 3: Iterate through each bin from the bottom up
        for i in range(len(bin_edges) - 1):
            y_start = bin_edges[i]
            y_end = bin_edges[i+1]
            
            mask = (data[:, 1] >= y_start) & (data[:, 1] < y_end)
            points_in_bin = data[mask]
            
            if points_in_bin.shape[0] > 0:
                c1_values_in_bin = points_in_bin[:, 2]
                
                if np.all(c1_values_in_bin < 0.5):
                    return y_start - y_init

        # If the loop completes, no bin was found that met the criteria
        print(f"  -> No y-bin found where all c1 < 0.5 in {filepath}.")
        return np.nan

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return np.nan
    

# --- (The main plotting function is updated to use the new file finder) ---

def generate_phase_diagram(root_directory, fixed_gamma13=6):
    """
    Main function to process data and plot the phase diagram overlaid on a
    theoretical wetting regime field.
    
    Args:
        root_directory (str): Path to the main data directory.
        fixed_gamma13 (float): The value of gamma13 to use for the theoretical background.
    """
    dir_pattern = re.compile(r'^(\d+)-(\d+)-(\d+)$')
    results = []
    print(f"Scanning root directory: {root_directory}")
    for dir_name in os.listdir(root_directory):
        full_path = os.path.join(root_directory, dir_name)
        if os.path.isdir(full_path):
            match = dir_pattern.match(dir_name)
            if match:
                gamma12, fixed_gamma13_val, gamma23 = map(int, match.groups())
                
                # Check if the fixed_gamma13 from the directory name matches the expected one
                if fixed_gamma13_val != fixed_gamma13:
                    continue

                gamma12 = gamma12 / 100
                # fixed_gamma13 is already set by the function argument
                gamma23 = gamma23 / 100
                
                print(f"Processing directory: {dir_name} -> (g12={gamma12}, g23={gamma23})")
                
                # *** MODIFICATION HERE: Use the new function to find the correct file ***
                analysis_file = find_analysis_file(full_path)
                
                if analysis_file:
                    # Now process the file that was found
                    max_y_value = process_data_file(analysis_file)
                    if not np.isnan(max_y_value):
                        results.append((gamma12, gamma23, max_y_value))
                    else:
                        # This message now refers to the result of process_data_file, not the file finding
                        print(f"  -> No valid final y-value found for (g12={gamma12}, g23={gamma23}).")
    
    if not results:
        print("\nNo data could be processed. Exiting.")
        return

    data_points = np.array(results)
    g12_vals = data_points[:, 0]
    g23_vals = data_points[:, 1]
    y_color_vals = data_points[:, 2]

    # --- 2. Create the Theoretical Wetting Regime Field (Unchanged) ---
    print("\nCreating theoretical wetting background...")
    
    fixed_gamma13_float = fixed_gamma13 / 100
    g12_min, g12_max = g12_vals.min() - 0.02, g12_vals.max() + 0.02
    g23_min, g23_max = g23_vals.min() - 0.02, g23_vals.max() + 0.02
    
    g12_grid, g23_grid = np.meshgrid(
        np.linspace(g12_min, g12_max, 500),
        np.linspace(g23_min, g23_max, 500)
    )

    regime_grid = np.zeros_like(g12_grid)
    regime_grid[g12_grid > fixed_gamma13_float + g23_grid] = 1
    regime_grid[fixed_gamma13_float > g12_grid + g23_grid] = 2
    regime_grid[g23_grid > g12_grid + fixed_gamma13_float] = 3

    regime_colors = ['#d3d3d3', '#ffcccc', '#cce5ff', '#d4edda']
    cmap_regimes = mcolors.ListedColormap(regime_colors)
    
    # --- 3. Plotting (Unchanged) ---
    print("Generating combined phase diagram plot...")
    fig, ax = plt.subplots(figsize=(12, 9))

    im = ax.imshow(regime_grid, origin='lower', 
                   extent=[g12_min, g12_max, g23_min, g23_max],
                   aspect='auto', cmap=cmap_regimes)

    scatter = ax.scatter(g12_vals, g23_vals, c=y_color_vals, 
                         cmap='viridis', s=1000, edgecolors='k', zorder=10, vmin=0, vmax=2.5)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Detachment height (h - h$_0$)', fontsize=14, weight='bold')

    legend_labels = {
        0: 'Partial Wetting / Dewetting',
        1: 'Droplet Dewets from Solid ($\gamma_{12} > \gamma_{13} + \gamma_{23}$)',
        2: 'Droplet Wets Solid ($\gamma_{13} > \gamma_{12} + \gamma_{23}$)',
        3: 'Solid Wets Droplet ($\gamma_{23} > \gamma_{12} + \gamma_{13}$)'
    }
    patches = [mpatches.Patch(color=regime_colors[i], label=label) for i, label in legend_labels.items()]
    ax.legend(handles=patches, loc='best', title='Theoretical Wetting Regimes', fontsize=10)

    ax.set_xlabel('$\gamma_{12}$ (Surface Tension 1-2)', fontsize=14, weight='bold')
    ax.set_ylabel('$\gamma_{23}$ (Surface Tension 2-3)', fontsize=14, weight='bold')
    ax.set_title(f'Phase Diagram of Wetting Behavior (for $\gamma_{13} = {fixed_gamma13_float}$)', fontsize=16, weight='bold')
    
    ax.set_xlim(g12_min, g12_max)
    ax.set_ylim(g23_min, g23_max)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Adjust ticks to be more readable
    x_ticks = np.unique(np.round(g12_vals, 2))
    y_ticks = np.unique(np.round(g23_vals, 2))
    plt.xticks(x_ticks) 
    plt.yticks(y_ticks) 

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    #plt.savefig('phase_diagram_with_background.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main_directory = 'data/highdiff'
    
    # The second argument is the fixed gamma13 value (as an integer, e.g., 6 for 0.06)
    # used in your simulations. This will be used to filter directories and for the background plot.
    generate_phase_diagram(main_directory, fixed_gamma13=9)