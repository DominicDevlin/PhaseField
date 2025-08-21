#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib.collections import PolyCollection
from pathlib import Path
import os  

# ------------------------------------------------------------
# User inputs
# ------------------------------------------------------------
# Directory name assumed to encode gamma values like "12-9-12"
data_directory = Path('data/sig13low/6-9-12')
time = 720  # integer time label matching filenames: c1-20.dat, c2-20.dat


fig, ax = plt.subplots(figsize=(8, 12))
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
    rgb_image = np.clip(np.dstack((grid_c3, grid_c2, grid_c1)), 0, 1)

    # --- 5. Plot the RGB image ---
    ax.imshow(rgb_image, origin='lower',
              extent=[x_min, x_max, y_min, y_max],
              interpolation='bilinear')

try:
    # os.path.normpath removes any trailing slash, os.path.basename gets the last part
    dir_name = os.path.basename(os.path.normpath(data_directory))
    # Split the name (e.g., '9-6-9') into parts
    gamma_parts = dir_name.split('-')
    # Convert parts to numbers, divide by 100, and format the text
    g12 = float(gamma_parts[0]) / 100.0
    g13 = float(gamma_parts[1]) / 100.0
    g23 = float(gamma_parts[2]) / 100.0
    gamma_text = (fr"$\gamma_{{12}} = {g12:.2f}$" + "\n" +
                  fr"$\gamma_{{13}} = {g13:.2f}$" + "\n" +
                  fr"$\gamma_{{23}} = {g23:.2f}$")
except (IndexError, ValueError):
    # If the directory name is not in the expected format, display a message
    print(f"Warning: Could not parse gamma values from directory name '{data_directory}'.")
    gamma_text = "gamma values\nnot found"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
cols = ['x', 'y', 'conc']
df = pd.read_csv(data_directory / f'c1-{time}.dat',
                    header=None, names=cols, delim_whitespace=True)
df2 = pd.read_csv(data_directory / f'c2-{time}.dat',
                    header=None, names=cols, delim_whitespace=True)

x = df['x'].values
y = df['y'].values
c = df['conc'].values - df2['conc'].values

# Create triangulation.
triang = Triangulation(x, y)

# Set plot properties for the current frame
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')

# Add text indicating the current time to the plot.
time_text = f"t = {time}"
ax.text(0.05, 0.95, time_text,
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        va='top',
        ha='left',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1, alpha=0.8))

# Add surface tension text to the upper right corner
ax.text(0.95, 0.95, gamma_text,
        transform=ax.transAxes,
        fontsize=12,
        va='top',
        ha='right',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1, alpha=0.8))

# Lock the axes view
ax.set_ylim(0, 6)
ax.set_xlim(-2,2)
# --- EXTRACT DATA FOR PLOTTING ---
x = df['x'].values
y = df['y'].values
c1 = df['conc'].values
c2 = df2['conc'].values
plot_rgb_composite_from_data(ax, x, y, c1, c2)



plt.tight_layout()
plt.show()
