import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def main():
    # Base directory containing subdirectories (e.g., "examples")
    base_path = "examples/64/"
    pattern = "0.889-64-*"
    # Pattern matching subdirectories of the form "0.889-64-xx.x"
    subdir_pattern = os.path.join(base_path, pattern)
    subdirectories = glob.glob(subdir_pattern)

    # Regex to extract time from filenames "phi_data-XXX.dat"
    time_regex = re.compile(r"phi_data-(?:.*-)?([\d.]+)\.dat")
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

        elongation_map = {}

        for fpath in data_files:
            # Extract time from filename
            fname = os.path.basename(fpath)
            print(fname)
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
            dataphi = np.loadtxt(fpath)
            print(fpath)
            rhofname = s_replaced = fpath.replace("phi", "rho")
            datarho = np.loadtxt(rhofname)

            # The first column is the x-axis
            x = dataphi[:, 0]
            y = dataphi[:, 1]   # y-values in the second column
            phi = dataphi[:, 2] # concentration in the third column
            rho = datarho[:, 2] # concentration in the third column


            ylen=6
            xlen=4
            interval = 0.02
            start_y = 1
            nsteps = int(ylen / interval)
            xnsteps = int(xlen / interval)
            start_point = int(nsteps * (start_y / ylen))
            
            avg_x_for_y = [round(x * interval - 2, 2) for x in range(int(xnsteps) + 1)] 
            y_list = [round(x * interval, 2) for x in range(int(nsteps) + 1)]   
            bool_list = [False] * len(y_list)
            
            full_matrix = [len(avg_x_for_y) * [0] for i in range(len(y_list))]
            full_matrix_counter =  [len(avg_x_for_y) * [0] for i in range(len(y_list))]
            full_matrix_avg =  [len(avg_x_for_y) * [0] for i in range(len(y_list))]
            
            sum_list = [0] * len(y_list)
            counter_list = [0] * len(y_list)
            
            # Find the y value for which phi_value goes below 0.x
            threshold = 0.8
            y_threshold = None


            for i in range(len(x)):
                if x[i] < 2 and x[i] > -2:
                    closest_value = min(y_list, key=lambda c: abs(c - y[i]))
                    if phi[i] > threshold:
                        bool_list[y_list.index(closest_value)] = True
                        
                    closx_val = min(avg_x_for_y, key=lambda c: abs(c - x[i]))
                    xind = avg_x_for_y.index(closx_val)
                    yind = y_list.index(closest_value)
                    sum_list[yind] += phi[i]
                    counter_list[yind] += 1
                    full_matrix[yind][xind] += phi[i]
                    full_matrix_counter[yind][xind] += 1
                    
                    
            np.set_printoptions(threshold=np.inf)
            # full_matrix_avg = np.divide(full_matrix, full_matrix_counter, out=np.zeros_like(full_matrix), where=full_matrix_counter!=0)


            curvature_list = []

            for i in range(len(full_matrix_avg)):
                avg_x_val = 0
                for j in range(len(full_matrix_avg[i])):
                    if (full_matrix_counter[i][j] != 0):
                        full_matrix_avg[i][j] = full_matrix[i][j] / full_matrix_counter[i][j]
                    avg_x_val += j * full_matrix_avg[i][j]
                if (sum(full_matrix_avg[i]) > 0):
                    avg_x_val = avg_x_val / sum(full_matrix_avg[i])
                    avg_x_val = avg_x_val * interval - 2
                else:
                    avg_x_val = 0
                curvature_list.append(avg_x_val)
                                
                    # if diff[i] < 0.01:
                    #     diff_values.append(np.nan)
                    # else:
                    #     diff_values.append(diff[i])
            avg_phi_list = []
            for (i, (sum_val, counter_val)) in enumerate(zip(sum_list, counter_list)):
                if counter_val != 0:
                    tt = sum_val / counter_val
                    avg_phi_list.append(tt)
                else:
                    avg_phi_list.append(0)
            
            curved_length = start_y
            
            for i in range(start_point, len(bool_list)):
                distance = np.sqrt((curvature_list[i] - curvature_list[i-1]) ** 2 + (y_list[i] - y_list[i-1]) ** 2)
                # print("distance is: ", distance)
                curved_length += distance
                if avg_phi_list[i] < 0.08:
                # if bool_list[i] == False and bool_list[i-1] == True:
                    y_threshold = y_list[i]
                    break
                # if (avg_phi_list[i] > avg_phi_list[i-2] + 0.1):
                #     y_threshold = y_list[i]
                #     break
            # print(y_threshold)
            print("y threshold is: ", y_threshold)
            print("curved length is: ", curved_length)
        
            curved_length = curved_length - 1.75
            if curved_length < 0:
                curved_length = 0

            y_threshold = y_threshold - 1.75
            if y_threshold < 0:
                y_threshold = 0            
        

            # Store in the dictionary
            elongation_map[time_val] = y_threshold
        
        if not elongation_map:
            continue

        # Sort times for plotting
        sorted_times = sorted(elongation_map.keys())
        sorted_concentrations = [elongation_map[t] for t in sorted_times]

        # Plot the line for the current subdirectory
        plt.plot(sorted_times, sorted_concentrations, marker='o', linestyle='-', label=label)


    plt.xlabel("Time")
    plt.ylabel("Total Order Parameter Concentration")
    plt.title("Total Concentration vs. Time")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
