import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation
import os   # Used for file system operations
import sys  # Used to exit the script gracefully

# ---------------------------------------------------------------
# 1. Animation and Data Configuration
# ---------------------------------------------------------------

# --- Main settings to configure ---
# Directory containing the data files
data_directory = 'data/sigma13low/15-9-9/'
# The increment between time steps (e.g., files are t=20, t=40, etc.)
time_step_increment = 20

# --- MODIFICATION START: Automatically find the time steps ---

# Check if the data directory exists
if not os.path.isdir(data_directory):
    print(f"Error: The specified data directory does not exist: '{data_directory}'")
    sys.exit(1) # Exit the script with an error code

# Find all time step numbers from the filenames (e.g., from 'c1-640.dat')
time_numbers = []
all_files = os.listdir(data_directory)
for filename in all_files:
    # We'll base our search on the 'c1' files
    if filename.startswith('c1-') and filename.endswith('.dat'):
        try:
            # Extract the numerical part of the filename
            # 'c1-640.dat' -> '640'
            time_str = filename.replace('c1-', '').replace('.dat', '')
            time_numbers.append(int(time_str))
        except ValueError:
            # This handles cases where the file might be named 'c1-final.dat' etc.
            print(f"Warning: Could not parse time step from filename '{filename}'. Skipping.")

# Check if any valid data files were found
if not time_numbers:
    print(f"Error: No valid data files (like 'c1-20.dat') were found in '{data_directory}'")
    sys.exit(1)

# Determine the start and end of the time sequence
first_time_step = min(time_numbers)
last_time_step = max(time_numbers)

print(f"Found data from time t={first_time_step} to t={last_time_step}.")

# Define the time steps for the animation automatically
# The range goes up to last_time_step + 1 because the 'stop' value in range is exclusive.
time_steps = range(first_time_step, last_time_step + 1, time_step_increment)

# This variable controls how long the final plot is displayed.
hold_frames = 10
# --- MODIFICATION END ---


# --- Parse surface tension (gamma) values from directory name ---
# This part extracts the numbers from the end of the data_directory path,
# divides them by 100, and formats them into a text string for the plot.
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

# Output movie file name
output_filename = 'concentration_movie.mp4'
fps = 5  # Frames per second for the output movie

# ---------------------------------------------------------------
# 2. Set up the Figure and Axes for Animation
#    This is done ONCE before the animation starts.
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 12))

# Define the contour levels and colormap.
# Keeping these constant across all frames ensures a consistent color scale.
levels = np.linspace(-1.03, 1.03, 40)
cmap = get_cmap('bwr')

# Calculate total frames for the animation
# This will be the number of data steps plus the number of held frames.
total_frames = len(time_steps) + hold_frames


# ---------------------------------------------------------------
# 3. Define the Animation Update Function
#    This function will be called for each frame of the movie.
# ---------------------------------------------------------------
def update(frame_number):
    """
    Clears the current plot and draws the data for the given frame_number.
    For frame numbers beyond the available data, it redraws the last frame.
    """
    # Determine which time step to display
    # If the current frame_number is within the range of our data, use it.
    # Otherwise, use the index of the last data point to "hold" the frame.
    last_data_index = len(time_steps) - 1
    time_index = min(frame_number, last_data_index)
    
    # Check if the time_index is valid before accessing time_steps
    if time_index < 0:
        # This case can happen if time_steps is empty, but we've already checked for that.
        # It's good practice to keep it for robustness.
        print("Warning: No time steps to process.")
        return
        
    time = time_steps[time_index]

    # Let the user know the progress
    print(f"Processing frame {frame_number + 1}/{total_frames} (time = {time})...")

    # Clear the previous frame's content from the axes
    ax.clear()

    # --- Load data for the current time step ---
    file1 = os.path.join(data_directory, f'c1-{time}.dat')
    file2 = os.path.join(data_directory, f'c2-{time}.dat')

    # Gracefully handle missing files (e.g., if a step in the sequence is missing)
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"Warning: Data files for time {time} not found. Skipping frame.")
        # Draw an empty frame with a message
        ax.text(0.5, 0.5, f'Data not found for t={time}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'2D Concentration Distribution (File Missing)')
        ax.set_ylim(0, 6) # Keep axes consistent even on error frames
        ax.set_xlim(-2,2)
        return

    df = pd.read_csv(file1, header=None, names=['x', 'y', 'conc'], delimiter='\t')
    df2 = pd.read_csv(file2, header=None, names=['x', 'y', 'conc'], delimiter='\t')

    # --- Process data ---
    x = df['x'].values
    y = df['y'].values
    c = df['conc'].values - df2['conc'].values

    # Create triangulation.
    triang = Triangulation(x, y)

    # --- Plotting for the current frame ---
    contour_f = ax.tricontourf(triang, c, levels=levels, cmap=cmap)

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


# ---------------------------------------------------------------
# 4. Create the Animation
# ---------------------------------------------------------------
# FuncAnimation will call the 'update' function for each frame.
ani = FuncAnimation(fig, update, frames=total_frames, blit=False)


# ---------------------------------------------------------------
# 5. Save the Animation
#    This will create an MP4 file. This process may take a few moments.
# ---------------------------------------------------------------
print(f"Saving animation to {output_filename}... (This may take a while)")
ani.save(output_filename, writer='ffmpeg', fps=fps, dpi=150)
print("Animation saved successfully!")

# To display the animation interactively instead of saving, you would use:
# plt.show()