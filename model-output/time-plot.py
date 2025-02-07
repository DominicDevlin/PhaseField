import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def integrate_concentration_with_delaunay(x, y, c):
    """
    x, y, c are all 1D numpy arrays of the same length:
      - x[i], y[i] = coordinates of the i-th point
      - c[i]       = concentration at the i-th point

    Returns the approximate integral (area-weighted total).
    """
    for cc in c:
        cc = cc**2 * (3-2*cc)

    # Step 1: Build a set of 2D points
    points = np.column_stack((x, y))

    # Step 2: Create a Delaunay triangulation
    tri = Delaunay(points)

    # Step 3: Integrate over each triangle in the triangulation
    total_concentration = 0.0

    for simplex in tri.simplices:
        # simplex will be a list of the 3 vertex indices
        # e.g. [i, j, k] for the triangle
        x_coords = points[simplex, 0]
        y_coords = points[simplex, 1]
        
        # The concentration at each triangle vertex
        c_vals = c[simplex]

        # (A) Compute triangle area
        # A common formula for area of a triangle in 2D is:
        #   area = | x1(y2-y3) + x2(y3-y1) + x3(y1-y2) | / 2
        # We'll do this using a determinant approach:
        mat = np.array([
            [x_coords[0], y_coords[0], 1],
            [x_coords[1], y_coords[1], 1],
            [x_coords[2], y_coords[2], 1]
        ])
        area = 0.5 * abs(np.linalg.det(mat))

        # (B) Approximate the triangle’s average concentration 
        #     by the average of its vertices:
        c_avg = np.mean(c_vals)

        # (C) Add the “triangle’s integrated concentration” 
        #     to the total
        total_concentration += c_avg * area

    return total_concentration  

def main():
    # Set the path to the subdirectory containing your .dat files
    data_path = "examples/0.889-64-51.3"    # <-- Replace with your path
    file_pattern = os.path.join(data_path, "rho_data-*.dat")

    # Prepare a dictionary to hold time -> total concentration
    time_concentration_map = {}

    # Get all .dat files
    data_files = glob.glob(file_pattern)
    
    # Regex to extract time from the filename "data-XXX.dat"
    # This will capture anything between "data-" and ".dat" as the time
    time_regex = re.compile(r"rho_data-(.+)\.dat")

    for fpath in data_files:
        # Extract time from filename
        fname = os.path.basename(fpath)
        match = time_regex.match(fname)
        if not match:
            print(f"Warning: Skipping file with unexpected name: {fname}")
            continue
        time_str = match.group(1)

        try:
            # Convert the time string to float
            time_val = float(time_str)
        except ValueError:
            print(f"Warning: Could not convert {time_str} to float. Skipping.")
            continue
        
        # Load data from file
        # Each line: x, y, concentration
        # data[:, 0] -> x
        # data[:, 1] -> y
        # data[:, 2] -> concentration
        data = np.loadtxt(fpath)
        
        # Sum up the concentration column
        total_concentration = integrate_concentration_with_delaunay(data[:,0], data[:,1], data[:,2])
        
        # np.sum(data[:, 2])

        # Store in the dictionary
        time_concentration_map[time_val] = total_concentration

    # Sort times for plotting
    sorted_times = sorted(time_concentration_map.keys())
    sorted_concentrations = [time_concentration_map[t] for t in sorted_times]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_times, sorted_concentrations, marker='o', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Total Order Parameter Concentration")
    plt.title("Total Concentration vs. Time")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
