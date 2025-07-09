import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation
import os # Added to check for file existence

# ---------------------------------------------------------------
# 1. Animation and Data Configuration
# ---------------------------------------------------------------

# --- Main settings to configure ---
# Directory containing the data files
# Create dummy data for demonstration if the directory doesn't exist
data_directory = 'data/sig13low/6-6-15/'



# Define the time steps for the animation
# Creates a sequence: 20, 40, 60, ..., 240
time_steps = range(20, 350, 20)

# Output movie file name
output_filename = 'concentration_movie.mp4'
fps = 3  # Frames per second for the output movie

# ---------------------------------------------------------------
# 2. Set up the Figure and Axes for Animation
#    This is done ONCE before the animation starts.
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 12))

# Define the contour levels and colormap.
# Keeping these constant across all frames ensures a consistent color scale.
levels = np.linspace(-1.03, 1.03, 40)
cmap = get_cmap('bwr')

# ---------------------------------------------------------------
# 3. Define the Animation Update Function
#    This function will be called for each frame of the movie.
# ---------------------------------------------------------------
def update(frame_number):
    """
    Clears the current plot and draws the data for the given frame_number.
    """
    # Get the corresponding time for the current frame
    time = time_steps[frame_number]

    # Let the user know the progress
    print(f"Processing frame {frame_number + 1}/{len(time_steps)} (time = {time})...")

    # Clear the previous frame's content from the axes
    ax.clear()

    # --- Load data for the current time step ---
    file1 = os.path.join(data_directory, f'c1-{time}.dat')
    file2 = os.path.join(data_directory, f'c2-{time}.dat')

    # Gracefully handle missing files
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"Warning: Data files for time {time} not found. Skipping frame.")
        # Draw an empty frame with a message
        ax.text(0.5, 0.5, f'Data not found for t={time}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'2D Concentration Distribution (File Missing)')
        return

    df = pd.read_csv(file1, header=None, names=['x', 'y', 'conc'], delimiter='\t')
    df2 = pd.read_csv(file2, header=None, names=['x', 'y', 'conc'], delimiter='\t')

    # --- Process data ---
    x = df['x'].values
    y = df['y'].values
    c = df['conc'].values - df2['conc'].values

    # Create triangulation. This assumes the x, y grid is the same for all files.
    # If the grid changes, this is the correct place to do it.
    triang = Triangulation(x, y)

    # --- Plotting for the current frame ---
    # Use the pre-defined levels and colormap for consistency
    contour_f = ax.tricontourf(triang, c, levels=levels, cmap=cmap)

    # Set plot properties for the current frame
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    # ax.set_title(f'2D Concentration Distribution at Time = {time}')

    # Add a colorbar for each frame. While slightly inefficient, it's the
    # simplest robust way when using ax.clear().
    # Note: To have a non-disappearing colorbar, it must be created outside the update loop.
    # For simplicity here, we omit it, but for a final plot, you'd manage it differently.
    # fig.colorbar(contour_f, ax=ax, label="Concentration")

    # *** MODIFICATION START ***
    # Add text indicating the current time to the plot.
    # transform=ax.transAxes places the text based on axes fractions (0,0 is bottom-left, 1,1 is top-right).
    # This ensures the text stays in the same visual position regardless of data limits.
    time_text = f"t = {time}"
    ax.text(0.05, 0.95, time_text,
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top', # Vertical alignment
            ha='left', # Horizontal alignment
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1, alpha=0.8))
    # *** MODIFICATION END ***

    # Optional: If you want to lock the axes view
    ax.set_ylim(0, 6)
    ax.set_xlim(-2,2)


# ---------------------------------------------------------------
# 4. Create the Animation
# ---------------------------------------------------------------
# FuncAnimation will call the 'update' function for each frame.
# 'frames' is the total number of frames to generate.
# 'interval' is the delay between frames in milliseconds (only affects live view).
# 'blit=False' is important here because we are redrawing the entire axes.
ani = FuncAnimation(fig, update, frames=len(time_steps), blit=False)

# This adds a colorbar that will stay fixed for the entire animation
# We draw the first frame to get the mappable object for the colorbar
# update(0)
# fig.colorbar(ax.collections[0], ax=ax, label="Concentration")

# ---------------------------------------------------------------
# 5. Save the Animation
#    This will create an MP4 file. This process may take a few moments.
# ---------------------------------------------------------------
print(f"Saving animation to {output_filename}... (This may take a while)")
ani.save(output_filename, writer='ffmpeg', fps=fps, dpi=150)
print("Animation saved successfully!")

# To display the animation interactively instead of saving, you would use:
# plt.show()