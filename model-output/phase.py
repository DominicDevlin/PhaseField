import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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
        # Find the maximum y-value in the data to set the upper limit for our bins
        y_max_data = np.max(data[:, 1])
        
        # Create an array of bin edges from the cutoff to the max y-value
        bin_edges = np.arange(y_min_cutoff, y_max_data + bin_width, bin_width)

        # Step 3: Iterate through each bin from the bottom up
        for i in range(len(bin_edges) - 1):
            y_start = bin_edges[i]
            y_end = bin_edges[i+1]
            
            # Find all data points whose y-coordinate falls into the current bin
            # The mask defines the slice: [y_start, y_end)
            mask = (data[:, 1] >= y_start) & (data[:, 1] < y_end)
            points_in_bin = data[mask]
            
            # IMPORTANT: We must ensure the bin is not empty. An empty bin doesn't
            # represent a physical region where the condition is met.
            if points_in_bin.shape[0] > 0:
                # Get the c1 values (third column) for all points in this bin
                c1_values_in_bin = points_in_bin[:, 2]
                
                # Check if ALL of these c1 values are less than 0.5
                if np.all(c1_values_in_bin < 0.5):
                    # Since we are iterating from the lowest y upwards, the first
                    # bin that satisfies this is our answer. Return its starting edge.
                    return y_start - y_init

        # If the loop completes, no bin was found that met the criteria
        print(f"  -> No y-bin found where all c1 < 0.5 in {filepath}.")
        return np.nan

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return np.nan
    
    

def generate_phase_diagram(root_directory, fixed_gamma13=6):
    """
    Main function to process data and plot the phase diagram overlaid on a
    theoretical wetting regime field.
    
    Args:
        root_directory (str): Path to the main data directory.
        fixed_gamma13 (float): The value of gamma13 to use for the theoretical background.
    """
    # --- 1. Data Processing (Same as before) ---
    dir_pattern = re.compile(r'^(\d+)-(\d+)-(\d+)$')
    results = []
    print(f"Scanning root directory: {root_directory}")
    for dir_name in os.listdir(root_directory):
        full_path = os.path.join(root_directory, dir_name)
        if os.path.isdir(full_path):
            match = dir_pattern.match(dir_name)
            if match:
                gamma12, fixed_gamma13, gamma23 = map(int, match.groups()) # We get g13 but ignore it for the axes
                gamma12 = gamma12 / 100
                fixed_gamma13 = fixed_gamma13 / 100
                gamma23 = gamma23 / 100
                print(f"Processing directory: {dir_name} -> (g12={gamma12}, g23={gamma23})")
                latest_file = find_latest_file(full_path)
                if latest_file:
                    max_y_value = process_data_file(latest_file)
                    if not np.isnan(max_y_value):
                        results.append((gamma12, gamma23, max_y_value))
                    else:
                        print(f"  -> No valid data point (c1>0.5) found for (g12={gamma12}, g23={gamma23}).")

    if not results:
        print("\nNo data could be processed. Exiting.")
        return

    data_points = np.array(results)
    g12_vals = data_points[:, 0]
    g23_vals = data_points[:, 1]
    y_color_vals = data_points[:, 2]

    # --- 2. Create the Theoretical Wetting Regime Field ---
    print("\nCreating theoretical wetting background...")
    
    # Define the boundaries of our plot based on the data, with some padding
    g12_min, g12_max = g12_vals.min() - 0.02, g12_vals.max() + 0.02
    g23_min, g23_max = g23_vals.min() - 0.02, g23_vals.max() + 0.02
    
    # Create a grid of points
    g12_grid, g23_grid = np.meshgrid(
        np.linspace(g12_min, g12_max, 500),
        np.linspace(g23_min, g23_max, 500)
    )

    # Define regime conditions based on the triangle inequality
    # We assign an integer to each regime.
    # 0: Partial Wetting (Young's Regime) - The default
    # 1: Droplet Dewets from Solid (g12 > g13 + g23)
    # 2: Droplet Wets Solid (g13 > g12 + g23)
    # 3: Solid Wets Droplet (g23 > g12 + g13)
    regime_grid = np.zeros_like(g12_grid)
    regime_grid[g12_grid > fixed_gamma13 + g23_grid] = 1
    regime_grid[fixed_gamma13 > g12_grid + g23_grid] = 2
    regime_grid[g23_grid > g12_grid + fixed_gamma13] = 3

    # Define discrete colors for the regimes
    # Using light, distinct colors for the background
    regime_colors = ['#d3d3d3', '#ffcccc', '#cce5ff', '#d4edda'] # Grey, Red, Blue, Green
    cmap_regimes = mcolors.ListedColormap(regime_colors)
    
    # --- 3. Plotting ---
    print("Generating combined phase diagram plot...")
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot the background field first
    im = ax.imshow(regime_grid, origin='lower', 
                   extent=[g12_min, g12_max, g23_min, g23_max],
                   aspect='auto', cmap=cmap_regimes)

    # Plot the simulation data points on top
    scatter = ax.scatter(g12_vals, g23_vals, c=y_color_vals, 
                         cmap='viridis', s=1000, edgecolors='k', zorder=10, vmin=0, vmax=2.5)

    # --- 4. Legends and Labels ---
    
    # Colorbar for the scatter plot data
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Max y-value (for c1 > 0.5)', fontsize=12, weight='bold')

    # Custom legend for the background regimes
    legend_labels = {
        0: 'Partial Wetting / Dewetting',
        1: 'Droplet Dewets from Solid ($\gamma_{12} > \gamma_{13} + \gamma_{23}$)',
        2: 'Droplet Wets Solid ($\gamma_{13} > \gamma_{12} + \gamma_{23}$)',
        3: 'Solid Wets Droplet ($\gamma_{23} > \gamma_{12} + \gamma_{13}$)'
    }
    patches = [mpatches.Patch(color=regime_colors[i], label=label) for i, label in legend_labels.items()]
    # ax.legend(handles=patches, bbox_to_anchor=(1.05, 0.6), loc='upper left',
    #           title='Theoretical Wetting Regimes', fontsize=10)

    # Set axis labels and title
    ax.set_xlabel('$\gamma_{12}$ (Surface Tension 1-2)', fontsize=14, weight='bold')
    ax.set_ylabel('$\gamma_{23}$ (Surface Tension 2-3)', fontsize=14, weight='bold')
    ax.set_title(f'Phase Diagram of Wetting Behavior (for $\gamma_{13} = {fixed_gamma13}$)', fontsize=16, weight='bold')
    
    ax.set_xlim(g12_min, g12_max)
    ax.set_ylim(g23_min, g23_max)
    ax.tick_params(axis='both', which='major', labelsize=12)
    # these are wrong
    plt.xticks(np.arange(0.06, 0.28, 0.02)) 
    plt.yticks(np.arange(0.06, 0.28, 0.02)) 

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    #plt.savefig('phase_diagram_with_background.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # --- Option 1: Point the script to your actual data directory ---
    # Replace this with the path to your folder containing '3-6-3', '9-6-3', etc.
    main_directory = 'data/sig13high'
    

    # The second argument is the fixed gamma13 value used for the background.
    # This should match the value used in your simulations.
    generate_phase_diagram(main_directory, fixed_gamma13=6)