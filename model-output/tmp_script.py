import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def main():
    # Base directory containing subdirectories (e.g., "examples")
    base_path = "examples/"
    pattern = "0.889-64-*"
    # Pattern matching subdirectories of the form "0.889-64-xx.x"
    subdir_pattern = os.path.join(base_path, pattern)
    subdirectories = glob.glob(subdir_pattern)

    # Regex to extract time from filenames "phi_data-XXX.dat"
    time_regex = re.compile(r"phi_data-(?:.*-)?([\d.]+)\.dat")
    labels = []
    sigmas = []
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
            
     

        labels.append(float(label))
        sigma = float(label) * np.sqrt((2)*0.0002) / 24
        sigmas.append(sigma)

    labels = sorted(labels)
    sigmas = sorted(sigmas)
    for i in range(len(labels)):
        print(labels[i], "  ", sigmas[i])


main()