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

# ===============================================================
#  NEW HELPER FUNCTIONS FOR RGB COMPOSITE PLOTTING
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
    # These limits should match the final plot's ax.set_xlim/ylim
    x_min, x_max = -2, 2
    y_min, y_max = 0, 6
    # Define image resolution. Higher numbers = better quality, slower processing.
    grid_x = np.linspace(x_min, x_max, 400)
    grid_y = np.linspace(y_min, y_max, 600)
    gx, gy = np.meshgrid(grid_x, grid_y)

    # --- 3. Interpolate scattered data onto the regular grid ---
    triang = Triangulation(x, y)
    interp_c1 = LinearTriInterpolator(triang, c1)
    interp_c2 = LinearTriInterpolator(triang, c2)
    interp_c3 = LinearTriInterpolator(triang, c3)

    grid_c1 = interp_c1(gx, gy)
    grid_c2 = interp_c2(gx, gy)
    grid_c3 = interp_c3(gx, gy)

    # --- 4. Stack the grids into an RGB image ---
    # Handle NaNs outside the data's convex hull by making them black
    grid_c1 = np.nan_to_num(grid_c1)
    grid_c2 = np.nan_to_num(grid_c2)
    grid_c3 = np.nan_to_num(grid_c3)

    # Clip values to the valid [0, 1] range for colors and stack them
    rgb_image = np.clip(np.dstack((grid_c1, grid_c2, grid_c3)), 0, 1)

    # --- 5. Plot the RGB image ---
    ax.imshow(rgb_image, origin='lower',
              extent=[x_min, x_max, y_min, y_max],
              interpolation='bilinear')

def add_rgb_legend(ax):
    """Adds a triangular RGB legend to the corner of the plot."""
    # Define triangle vertices in axis-fraction coordinates (from 0 to 1)
    c1_vert = [0.05, 0.05]  # Red
    c2_vert = [0.20, 0.05]  # Green
    c3_vert = [0.125, 0.18] # Blue
    verts = np.array([c1_vert, c2_vert, c3_vert])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Use tricontourf on the small triangle to create the color gradient
    legend_tri = Triangulation(verts[:, 0], verts[:, 1])
    ax.tricontourf(legend_tri, colors[:, 0], 100, cmap='Reds', transform=ax.transAxes, antialiased=False)
    ax.tricontourf(legend_tri, colors[:, 1], 100, cmap='Greens', transform=ax.transAxes, antialiased=False)
    ax.tricontourf(legend_tri, colors[:, 2], 100, cmap='Blues', transform=ax.transAxes, antialiased=False)
    
    # Add a black border to the triangle
    triangle_border = Polygon(verts, transform=ax.transAxes, facecolor='none', edgecolor='black', linewidth=1.5)
    ax.add_patch(triangle_border)

    # Add labels
    ax.text(c1_vert[0], c1_vert[1] - 0.03, 'c1', transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(c2_vert[0], c2_vert[1] - 0.03, 'c2', transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(c3_vert[0], c3_vert[1] + 0.02, 'c3', transform=ax.transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')

# ---------------------------------------------------------------
# 1. Animation and Data Configuration
# ---------------------------------------------------------------
data_directory = 'data/sig13low/12-9-15/'
time_step_increment = 20
output_filename = 'concentration_movie_rgb.mp4'
fps = 5
hold_frames = 10

if not os.path.isdir(data_directory):
    print(f"Error: The specified data directory does not exist: '{data_directory}'")
    sys.exit(1)

time_numbers = []
all_files = os.listdir(data_directory)
for filename in all_files:
    if filename.startswith('c1-') and filename.endswith('.dat'):
        try:
            time_str = filename.replace('c1-', '').replace('.dat', '')
            time_numbers.append(int(time_str))
        except ValueError:
            print(f"Warning: Could not parse time step from filename '{filename}'. Skipping.")

if not time_numbers:
    print(f"Error: No valid data files found in '{data_directory}'")
    sys.exit(1)

first_time_step = min(time_numbers)
last_time_step = max(time_numbers)
print(f"Found data from time t={first_time_step} to t={last_time_step}.")
time_steps = range(first_time_step, last_time_step + 1, time_step_increment)

try:
    dir_name = os.path.basename(os.path.normpath(data_directory))
    gamma_parts = dir_name.split('-')
    g12 = float(gamma_parts[0]) / 100.0
    g13 = float(gamma_parts[1]) / 100.0
    g23 = float(gamma_parts[2]) / 100.0
    gamma_text = (fr"$\gamma_{{12}} = {g12:.2f}$" + "\n" +
                  fr"$\gamma_{{13}} = {g13:.2f}$" + "\n" +
                  fr"$\gamma_{{23}} = {g23:.2f}$")
except (IndexError, ValueError):
    print(f"Warning: Could not parse gamma values from directory name '{data_directory}'.")
    gamma_text = "gamma values\nnot found"

# ---------------------------------------------------------------
# 2. Set up the Figure and Axes for Animation
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 12))
total_frames = len(time_steps) + hold_frames

# ---------------------------------------------------------------
# 3. Define the Animation Update Function
# ---------------------------------------------------------------
def update(frame_number):
    """
    Clears the current plot and draws the data for the given frame_number
    using the RGB composite method.
    """
    last_data_index = len(time_steps) - 1
    time_index = min(frame_number, last_data_index)
    
    if time_index < 0:
        print("Warning: No time steps to process.")
        return
        
    time = time_steps[time_index]
    print(f"Processing frame {frame_number + 1}/{total_frames} (time = {time})...")
    ax.clear()

    file1 = os.path.join(data_directory, f'c1-{time}.dat')
    file2 = os.path.join(data_directory, f'c2-{time}.dat')

    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"Warning: Data files for time {time} not found. Skipping frame.")
        ax.text(0.5, 0.5, f'Data not found for t={time}', ha='center', va='center', transform=ax.transAxes)
        ax.set_ylim(0, 6); ax.set_xlim(-2,2)
        return

    df = pd.read_csv(file1, header=None, names=['x', 'y', 'conc'], delimiter='\t')
    df2 = pd.read_csv(file2, header=None, names=['x', 'y', 'conc'], delimiter='\t')

    # --- EXTRACT DATA FOR PLOTTING ---
    x = df['x'].values
    y = df['y'].values
    c1 = df['conc'].values
    c2 = df2['conc'].values

    # --- PLOTTING FOR THE CURRENT FRAME (NEW METHOD) ---
    plot_rgb_composite_from_data(ax, x, y, c1, c2)

    # --- Set plot properties (same as before) ---
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')

    time_text = f"t = {time}"
    ax.text(0.05, 0.95, time_text,
            transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1, alpha=0.8))

    ax.text(0.95, 0.95, gamma_text,
            transform=ax.transAxes, fontsize=12, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1, alpha=0.8))

    ax.set_ylim(0, 6)
    ax.set_xlim(-2,2)

    # --- ADD THE RGB LEGEND ---
    # add_rgb_legend(ax)

# ---------------------------------------------------------------
# 4. Create and Save the Animation
# ---------------------------------------------------------------
ani = FuncAnimation(fig, update, frames=total_frames, blit=False)

print(f"Saving animation to {output_filename}... (This may take a while)")
ani.save(output_filename, writer='ffmpeg', fps=fps, dpi=150)
print("Animation saved successfully!")