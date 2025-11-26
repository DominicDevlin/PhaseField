import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
import os
import sys

# ---------------------------------------------------------------
# 1. Animation and Data Configuration
# ---------------------------------------------------------------
tensions = '12-9-12'
data_directory = 'data/sig13low/' + tensions + '/'
time_step_increment = 20
output_filename = 'movies/' + tensions + '.mp4'
fps = 6
hold_frames = 10

manual_numbers = False
start_time = 20
end_time = 500

# <<< CHANGE 1: DEFINE PLOT LIMITS AND A BASE FIGURE WIDTH
x_limits = (-2, 2)
y_limits = (0, 5) # Your new desired vertical limit
fig_width_inches = 8 # Let's keep the width consistent

# ===============================================================
#  HELPER FUNCTIONS FOR RGB COMPOSITE PLOTTING
# ===============================================================

def plot_rgb_composite_from_data(ax, x, y, c1, c2):
    """
    Interpolates scattered phase-field data (c1, c2) onto a regular grid
    and plots it as an RGB image on the given axes.
    R = c1, G = c2, B = c3 (where c3 = 1 - c1 - c2)
    """
    # --- 1. Calculate c3 using the conservation constraint ---
    c3 = 1.0 - c1 - c2

    # --- 2. Define a regular grid for the output image ---
    x_min, x_max = x_limits[0], x_limits[1]
    y_min, y_max = y_limits[0], y_limits[1]
    grid_x = np.linspace(x_min, x_max, 400)
    grid_y = np.linspace(y_min, y_max, 600)
    gx, gy = np.meshgrid(grid_x, grid_y)

    # --- 3. Interpolate scattered data onto the regular grid ---
    # NOTE: This section is where the original error occurred. We now pre-validate
    # the data, so this is not expected to fail during the animation rendering.
    triang = Triangulation(x, y)
    interp_c1 = LinearTriInterpolator(triang, c1)
    interp_c2 = LinearTriInterpolator(triang, c2)
    interp_c3 = LinearTriInterpolator(triang, c3)

    grid_c1 = interp_c1(gx, gy)
    grid_c2 = interp_c2(gx, gy)
    grid_c3 = interp_c3(gx, gy)

    # --- 4. Stack the grids into an RGB image ---
    grid_c1 = np.nan_to_num(grid_c1)
    grid_c2 = np.nan_to_num(grid_c2)
    grid_c3 = np.nan_to_num(grid_c3)

    rgb_image = np.clip(np.dstack((grid_c3, grid_c2, grid_c1)), 0, 1)

    # --- 5. Plot the RGB image ---
    ax.imshow(rgb_image, origin='lower',
              extent=[x_min, x_max, y_min, y_max],
              interpolation='bilinear')

def add_rgb_legend(ax):
    """Adds a triangular RGB legend to the corner of the plot."""
    c1_vert = [0.05, 0.05]
    c2_vert = [0.20, 0.05]
    c3_vert = [0.125, 0.18]
    verts = np.array([c1_vert, c2_vert, c3_vert])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    legend_tri = Triangulation(verts[:, 0], verts[:, 1])
    ax.tricontourf(legend_tri, colors[:, 0], 100, cmap='Reds', transform=ax.transAxes, antialiased=False)
    ax.tricontourf(legend_tri, colors[:, 1], 100, cmap='Greens', transform=ax.transAxes, antialiased=False)
    ax.tricontourf(legend_tri, colors[:, 2], 100, cmap='Blues', transform=ax.transAxes, antialiased=False)
    
    triangle_border = Polygon(verts, transform=ax.transAxes, facecolor='none', edgecolor='black', linewidth=1.5)
    ax.add_patch(triangle_border)

    ax.text(c1_vert[0], c1_vert[1] - 0.03, 'c1', transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(c2_vert[0], c2_vert[1] - 0.03, 'c2', transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(c3_vert[0], c3_vert[1] + 0.02, 'c3', transform=ax.transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')

# ===============================================================
# DATA LOADING AND VALIDATION
# ===============================================================

if not os.path.isdir(data_directory):
    print(f"Error: The specified data directory does not exist: '{data_directory}'")
    sys.exit(1)

time_numbers = []
for filename in os.listdir(data_directory):
    if filename.startswith('c1-') and filename.endswith('.dat'):
        try:
            time_str = filename.replace('c1-', '').replace('.dat', '')
            time_numbers.append(int(time_str))
        except ValueError:
            print(f"Warning: Could not parse time step from filename '{filename}'. Skipping.")

if not time_numbers:
    print(f"Error: No valid data files found in '{data_directory}'")
    sys.exit(1)

if (manual_numbers):
    first_time_step = start_time
    last_time_step = end_time
else:    
    first_time_step = min(time_numbers)
    last_time_step = max(time_numbers)
print(f"Found data from time t={first_time_step} to t={last_time_step}.")
time_steps = range(first_time_step, last_time_step + 1, time_step_increment)

try:
    dir_name = os.path.basename(os.path.normpath(data_directory))
    gamma_parts = dir_name.split('-')
    g12 = float(gamma_parts[0]) / 150.0
    g13 = float(gamma_parts[1]) / 150.0
    g23 = float(gamma_parts[2]) / 150.0
    gamma_text = (fr"$\gamma_{{12}} = {g12:.2f}$" + "\n" +
                  fr"$\gamma_{{13}} = {g13:.2f}$" + "\n" +
                  fr"$\gamma_{{23}} = {g23:.2f}$")
except (IndexError, ValueError):
    print(f"Warning: Could not parse gamma values from directory name '{data_directory}'.")
    gamma_text = "gamma values\nnot found"

# ---------------------------------------------------------------
# 2. NEW: Pre-validate all time steps before animating
# ---------------------------------------------------------------
print("\nValidating data files and checking for triangulation issues...")
valid_time_steps = []
for time in time_steps:
    file1 = os.path.join(data_directory, f'c1-{time}.dat')
    file2 = os.path.join(data_directory, f'c2-{time}.dat')

    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"  - Skipping t={time}: Data files not found.")
        continue

    try:
        df = pd.read_csv(file1, header=None, names=['x', 'y', 'conc'], delimiter='\t')
        x = df['x'].values
        y = df['y'].values
        c1 = df['conc'].values

        # This is the crucial test that mimics the failing code path.
        # We create a Triangulation and a LinearTriInterpolator to see if it fails.
        triang = Triangulation(x, y)
        _ = LinearTriInterpolator(triang, c1)

        # If we get here without an error, the data is good for this frame.
        valid_time_steps.append(time)

    except RuntimeError as e:
        # This catches the specific "Triangulation is invalid" error.
        if "Triangulation is invalid" in str(e):
            print(f"  - Skipping t={time}: Invalid triangulation. The data is likely degenerate.")
        else:
            # If it's a different RuntimeError, we might want to know.
            print(f"  - Skipping t={time} due to an unexpected runtime error: {e}")
    except Exception as e:
        # Catch other potential errors (e.g., empty files, parsing issues).
        print(f"  - Skipping t={time} due to a data loading error: {e}")

if not valid_time_steps:
    print("\nError: No valid data frames could be processed after validation.")
    sys.exit(1)
    
print(f"\nValidation complete. Using {len(valid_time_steps)} of {len(time_steps)} possible frames for the animation.")


# ---------------------------------------------------------------
# 3. Set up the Figure and Axes for Animation
# ---------------------------------------------------------------
data_width = x_limits[1] - x_limits[0]
data_height = y_limits[1] - y_limits[0]
aspect_ratio = data_height / data_width
fig_height_inches = fig_width_inches * aspect_ratio

# Now create the figure with the correct size
fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
# The total number of frames is now based on the count of VALID frames.
total_frames = len(valid_time_steps) + hold_frames

# ---------------------------------------------------------------
# 4. Define the Animation Update Function
# ---------------------------------------------------------------
def update(frame_number):
    """
    Clears the current plot and draws the data for the given frame_number
    using the list of pre-validated time steps.
    """
    # The index now corresponds to the list of *valid* time steps.
    last_data_index = len(valid_time_steps) - 1
    time_index = min(frame_number, last_data_index)
    
    # This prevents errors if valid_time_steps is empty (though we check this earlier).
    if time_index < 0:
        print("Warning: No valid time steps to process.")
        return
        
    # Get the time value from our validated list.
    time = valid_time_steps[time_index]
    print(f"Processing frame {frame_number + 1}/{total_frames} (time = {time})...")
    ax.clear()

    file1 = os.path.join(data_directory, f'c1-{time}.dat')
    file2 = os.path.join(data_directory, f'c2-{time}.dat')

    # Note: We don't strictly need this check anymore because of pre-validation,
    # but it's good practice to keep it.
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"Warning: Data files for time {time} not found. Skipping frame.")
        ax.text(0.5, 0.5, f'Data not found for t={time}', ha='center', va='center', transform=ax.transAxes)
        ax.set_ylim(0, 6); ax.set_xlim(-2,2)
        return

    df = pd.read_csv(file1, header=None, names=['x', 'y', 'conc'], delimiter='\t')
    df2 = pd.read_csv(file2, header=None, names=['x', 'y', 'conc'], delimiter='\t')

    x = df['x'].values
    y = df['y'].values
    c1 = df['conc'].values
    c2 = df2['conc'].values

    # Plotting for the current frame will now succeed.
    plot_rgb_composite_from_data(ax, x, y, c1, c2)

    # Set plot properties
    ax.set_aspect('equal', adjustable='box')
    # ax.set_xlabel('X coordinate')
    # ax.set_ylabel('Y coordinate')

    time_text = f"t = {time}"
    # ax.text(0.05, 0.95, time_text,
    #         transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left',
    #         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1, alpha=0.8))

    # ax.text(0.95, 0.95, gamma_text,
    #         transform=ax.transAxes, fontsize=12, va='top', ha='right',
    #         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1, alpha=0.8))
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.axis('off')

    # add_rgb_legend(ax)

# ---------------------------------------------------------------
# 5. Create and Save the Animation
# ---------------------------------------------------------------
# The animation is now created with the correct number of total frames.
ani = FuncAnimation(fig, update, frames=total_frames, blit=False)

print(f"\nSaving animation to {output_filename}... (This may take a while)")
ani.save(output_filename, writer='ffmpeg', fps=fps, dpi=150, savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0})
print("Animation saved successfully!")