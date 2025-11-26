import os
import numpy as np
import re
from scipy.spatial import Delaunay

# =================CONFIGURATION=================
ROOT_DIR = 'data/diff-phase'
OUTPUT_SUBDIR = 'mass'
# ===============================================

def get_triangle_area(pts):
    """
    Calculates the area of a triangle given 3 points (x, y).
    Formula: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    """
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(x[0]*(y[1] - y[2]) + x[1]*(y[2] - y[0]) + x[2]*(y[0] - y[1]))

def integrate_concentration_with_delaunay(x, y, c):
    """
    Calculates the volume integral of concentration 'c' over the domain defined by x, y
    using Delaunay triangulation.
    
    Volume = Sum( Area_triangle * Average_Concentration_triangle )
    """
    # Combine x and y into points coordinates
    points = np.column_stack((x, y))

    # Create Delaunay triangulation
    tri = Delaunay(points)

    total_volume = 0.0

    # Iterate over every triangle (simplex) in the mesh
    for simplex in tri.simplices:
        # simplex contains the indices of the 3 points forming the triangle
        
        # Get the 3 coordinate points [ [x1,y1], [x2,y2], [x3,y3] ]
        tri_pts = points[simplex]
        
        # Get the 3 concentration values at these points
        tri_c = c[simplex]

        # Calculate Area of the triangle
        area = get_triangle_area(tri_pts)

        # Calculate Average concentration of the triangle (Linear interpolation)
        avg_c = np.mean(tri_c)

        # Volume contribution is Area * Average Height (Concentration)
        total_volume += area * avg_c

    return total_volume

def parse_filename(filename):
    """
    Parses 'c1-240.dat' to return ('c1', 240)
    """
    # Regex to match c(number)-(time).dat
    match = re.match(r'c(\d+)-(\d+)\.dat', filename)
    if match:
        field_id = f"c{match.group(1)}"
        time_point = int(match.group(2))
        return field_id, time_point
    return None, None

def process_directories():
    # Create the output mass directory if it doesn't exist
    mass_dir_path = os.path.join(ROOT_DIR, OUTPUT_SUBDIR)
    if not os.path.exists(mass_dir_path):
        os.makedirs(mass_dir_path)
        print(f"Created output directory: {mass_dir_path}")

    # Iterate over subdirectories in sig13low
    for item in os.listdir(ROOT_DIR):
        subdir_path = os.path.join(ROOT_DIR, item)

        # Check if it is a directory and ignore the 'mass' folder itself
        if os.path.isdir(subdir_path) and item != OUTPUT_SUBDIR:
            print(f"Processing directory: {item}...")
            
            # Dictionary to store data: { time: { 'c1': val, 'c2': val } }
            time_data = {}

            # List all files in the subdirectory
            files = os.listdir(subdir_path)
            
            for fname in files:
                if not fname.endswith('.dat'):
                    continue

                field, time = parse_filename(fname)
                
                if field and time is not None:
                    filepath = os.path.join(subdir_path, fname)
                    
                    # Load data: X (col 0), Y (col 1), Conc (col 2)
                    try:
                        data = np.loadtxt(filepath)
                        if data.shape[0] < 3:
                            continue # Not enough points for triangulation
                            
                        x = data[:, 0]
                        y = data[:, 1]
                        c = data[:, 2]

                        # Calculate volume
                        volume = integrate_concentration_with_delaunay(x, y, c)

                        if time not in time_data:
                            time_data[time] = {}
                        
                        time_data[time][field] = volume
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")

            # Prepare data for writing
            # We need columns: Time, Vol_c1, Vol_c2
            # Assuming c1 and c2 are the fields based on your description
            output_lines = []
            
            # Sort by time
            sorted_times = sorted(time_data.keys())
            
            for t in sorted_times:
                vol_c1 = time_data[t].get('c1', 0.0)
                # Note: Prompt mentioned c3 in text but c2 in filenames. 
                # Using c2 based on filenames c2-240.dat etc.
                vol_c2 = time_data[t].get('c2', 0.0) 
                
                output_lines.append(f"{t}\t{vol_c1:.6f}\t{vol_c2:.6f}")

            # Write to output file
            output_filename = f"mass-{item}"
            output_filepath = os.path.join(mass_dir_path, output_filename)
            
            with open(output_filepath, 'w') as f:
                # Optional: Header
                # f.write("Time\tVol_C1\tVol_C2\n") 
                f.write('\n'.join(output_lines))
            
            print(f"Saved: {output_filepath}")

if __name__ == "__main__":
    process_directories()