import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def integrate_concentration_with_delaunay(x, y, phi, rho):
    """
    x, y, c are all 1D numpy arrays of the same length:
      - x[i], y[i] = coordinates of the i-th point
      - c[i]       = concentration at the i-th point

    Returns the approximate integral (area-weighted total).
    """
    rhophiarea=[]

    for i in range(len(phi)):
        cc = (phi[i] < 1) * rho[i]**2 * (3-2*rho[i]) + (phi[i] > 1) * rho[i]**2 * (3-2*rho[i])
        rhophiarea.append(cc)

    rhophiarea = np.array(rhophiarea)

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
        c_vals = rhophiarea[simplex]

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
    # Base directory containing subdirectories (e.g., "examples")
    base_path = "examples/64/"
    pattern = "0.889-64-*"
    # Pattern matching subdirectories of the form "0.889-64-xx.x"
    subdir_pattern = os.path.join(base_path, pattern)
    subdirectories = glob.glob(subdir_pattern)

    # Regex to extract time from filenames "phi_data-XXX.dat"
    time_regex = re.compile(r"phi_data-(?:.*-)?([\d.]+)\.dat")

    
    # Prepare the plot
    plt.figure(figsize=(8, 6))

    # Loop over each subdirectory
    for subdir in subdirectories:
        # Use the last part of the subdirectory name for the legend label.
        subdir_label = os.path.basename(subdir)  # e.g., "0.889-64-51.3"
        # Optionally, extract just the "xx.x" part:
        label_parts = subdir_label.split("-")
        if len(label_parts) >= 3:
            label = label_parts[-1]
        else:
            label = subdir_label

        # Pattern for the phi data files in the current subdirectory
        file_pattern = os.path.join(subdir, "phi_data-*.dat")
        data_files = glob.glob(file_pattern)
        
        # Dictionary to hold time -> total concentration for this subdirectory
        time_concentration_map = {}

        for fpath in data_files:
            fname = os.path.basename(fpath)
            match = time_regex.match(fname)
            if not match:
                print(f"Warning: Skipping file with unexpected name: {fname}")
                continue

            time_str = match.group(1)
            try:
                time_val = float(time_str)
            except ValueError:
                print(f"Warning: Could not convert {time_str} to float. Skipping.")
                continue

            # Load phi data (assumes each row is: x, y, concentration)
            try:
                data = np.loadtxt(fpath)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue

            # Construct the corresponding rho filename (replace "phi" with "rho")
            rhofname = fpath.replace("phi", "rho")
            try:
                datarho = np.loadtxt(rhofname)
            except Exception as e:
                print(f"Error loading {rhofname}: {e}")
                continue

            # Compute the total concentration (using your integration function)
            total_concentration = integrate_concentration_with_delaunay(
                data[:, 0], data[:, 1], data[:, 2], datarho[:, 2]
            )

            time_concentration_map[time_val] = total_concentration

        # If no data was collected from this subdirectory, skip plotting
        if not time_concentration_map:
            continue

        # Sort the times and extract corresponding concentrations
        sorted_times = sorted(time_concentration_map.keys())
        sorted_concentrations = [time_concentration_map[t] for t in sorted_times]

        # Plot the line for the current subdirectory
        plt.plot(sorted_times, sorted_concentrations, marker='o', linestyle='-', label=label)

    # Finalize the plot
    plt.xlabel("Time")
    plt.ylabel("Total Order Parameter Concentration")
    plt.title("Total Concentration vs. Time for Different Subdirectories")
    plt.legend(title="Subdirectory (xx.x)")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
