import numpy as np
from scipy.signal import argrelextrema
import os
import glob
import re
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import math
import matplotlib.pyplot as plt


# Use glob to find all .dat files in 'data' folder that match the pattern
files = glob.glob('data/phi_data-*.dat')

# Regex pattern to capture two integer parameters between hyphens
pattern = re.compile(r'phi_data-(\d+)-(\d+)\.dat$')

# Prepare a list or dictionary to store results
gamma_phis = []
gamma_rhos = []
minima = []



def solve_quadratic(a, b, c):
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c
    
    # For real solutions (discriminant >= 0)
    if discriminant > 0:
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)
        return (root1, root2)
    elif discriminant == 0:
        # One real root (double root)
        root = -b / (2*a)
        return (root, root)
    else:
        # If the discriminant is negative, provide complex solutions
        real_part = -b / (2*a)
        imaginary_part = math.sqrt(-discriminant) / (2*a)
        root1 = complex(real_part, imaginary_part)
        root2 = complex(real_part, -imaginary_part)
        return (root1, root2)



def get_data():
    for fpath in files:
        # Extract filename without path
        fname = os.path.basename(fpath)
        
        # Use the regex to capture the two numbers in the filename
        match = pattern.match(fname)
        if not match:
            # Skip if the file name does not match the expected pattern
            continue

        # Extract the two parameters (convert to integer)
        param1, param2 = map(int, match.groups())
        
        dataphi = np.loadtxt(fpath)
        dataphi = dataphi[dataphi[:, 0].argsort()]
        # The first column is the x-axis
        x = dataphi[:, 0]
        y = dataphi[:, 1]


        # 2) Find indices of local minima
        # argrelextrema(y, np.less) finds all indices i
        # such that y[i] < y[i-1] and y[i] < y[i+1] (local minima)
        minima_indices = argrelextrema(y, np.less_equal)
        
        # print(fpath, minima_indices)

        # argrelextrema returns a tuple of arrays, so we extract the first element:
        minima_indices = minima_indices[0]
        
        hit = False
        curmin = 1.0

        for idx in minima_indices:
            val = y[idx]
            if (val < 0.995 and val > 0.1):
                curmin = val
                hit=True
                break

        if param1 < 2:# and param2 < 16:
            minima.append(curmin)
            print(param1, param2, curmin)
                
            gamma_phis.append(param1)
            gamma_rhos.append(param2)

        # 3) Print results
        # for idx in minima_indices:
        #     print(f"Local minimum at index={idx}: x={x[idx]}, y={y[idx]}")
            
            
def PlotPhase():  
    unique_x = np.unique(gamma_phis)
    unique_y = np.unique(gamma_rhos)

    # Create matrices for plotting
    grid_x, grid_y = np.meshgrid(unique_x, unique_y)
    grid_order_param = np.full(grid_x.shape, np.nan)

    # Fill the grid with the order parameter values
    for i in range(len(gamma_phis)):
        x_idx = np.where(unique_x == gamma_phis[i])[0][0]
        y_idx = np.where(unique_y == gamma_rhos[i])[0][0]
        grid_order_param[y_idx, x_idx] = minima[i]

    # Plotting the phase diagram
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.viridis
    # norm = mcolors.Normalize(vmin=np.nanmin(0), vmax=np.nanmax(100))
    plt.pcolormesh(grid_x, grid_y, grid_order_param, cmap=cmap, shading='auto')#, norm=norm)
    plt.colorbar(label='Elongation')
    plt.xlabel('gamma liquid')
    plt.ylabel('gamma liquid-solid')
    plt.title('Phase Diagram')
    plt.show()


def PlotLine():


    # Ensure both lists are paired and sorted by gamma_rhos
    sorted_pairs = sorted(zip(gamma_rhos, minima))

    # Unzip the sorted pairs back into two lists
    sorted_gamma_rhos, sorted_minima = zip(*sorted_pairs)

    # Convert tuples back to lists
    sorted_gamma_rhos = list(sorted_gamma_rhos)
    sorted_minima = list(sorted_minima)

    Lvals = []

    for g in sorted_gamma_rhos:
        eps=0.001
        gphi = 1

        gamma_p = np.sqrt((8/9)*gphi*eps)
        gamma_r = np.sqrt((8/9)*g*eps)
        # make quadratic:
        # qa = 9*gamma_p**2
        # qb = - gamma_r**2
        # qc = gamma_r**2 / 2. + 0*gamma_p**2

        # r1, r2 = solve_quadratic(qa, qb, qc)
        # newL=0
        # # print(r1, r2)
        # if (r1.imag != 0 or r2.imag != 0):
        #     newL = 1
        # else:
        #     r1real = r1.real
        #     r2real = r2.real
        #     print(r1real, r2real)
        #     if (r2real < r1real):
        #         newL = r2real
        #     else:  
        #         newL = r1real   
        # newL = (3*gamma_p**2 + ((8./9.)*g*eps/2))/((8./9.)*g*eps - (8./9.)*eps)
        # newL = (4*gamma_p**2 + ((8./9.)*g*eps/2) )/((8./9.)*g*eps ) # float( gamma_r)/2. + 2.*gamma_p) / (gamma_r )
        newL = 4*(gphi)/g + 0.5
        if newL > 1:
            Lvals.append(math.nan)
        else:
            Lvals.append(2*newL-1)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_gamma_rhos, sorted_minima, marker='o', linestyle='-', label='Line connecting unique points')
    plt.plot(sorted_gamma_rhos, Lvals, marker='x', linestyle='-', label='Line connecting unique points')

    # Customize plot
    plt.xlabel('Unique gamma_rhos')
    plt.ylabel('Unique minima')
    plt.title('Line Plot of Unique Values')
    plt.legend()

    # Show the plot
    plt.show()




get_data()

PlotLine()
