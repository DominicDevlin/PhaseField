import numpy as np
import os
import matplotlib.pyplot as plt

# Load the data from a text file
# The data.txt file should contain the provided rows of numbers
# datan= '-2-42.2-1.72'
# Get a list of all subdirectories in the 'data' directory
data_dir = 'data/'
subdirectories = [subdir for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]

data_taus = []
data_tau_rhophis = []
y_thresholds = []

# Iterate through each subdirectory
for subdir in subdirectories:
    dir = data_dir + subdir + '/'
    datan = '-' + subdir
    # Separate the numbers in datan and store them as variables
    datan_numbers = datan.split('-')[1:]
    tphi = float(datan_numbers[0])
    trhophi = float(datan_numbers[1])
    trho = float(datan_numbers[2])
    

    # Load the data from the current subdirectory
    dataphi = np.loadtxt(dir + 'phi_data' + datan + '.dat')
    datarho = np.loadtxt(dir + 'rho_data' + datan + '.dat')
    datadiff = np.loadtxt(dir + 'diff_data' + datan + '.dat')

    dataphi = dataphi[dataphi[:, 1].argsort()]
    datarho = datarho[datarho[:, 1].argsort()]
    datadiff = datadiff[datadiff[:, 1].argsort()]

    # The first column is the x-axis
    x = dataphi[:, 0]
    y = dataphi[:, 1]   # y-values in the second column
    phi = dataphi[:, 2] # concentration in the third column
    rho = datarho[:, 2] # concentration in the third column
    diff = datadiff[:, 2]

    y_values=[]
    phi_values=[]
    rho_values=[]
    diff_values = []

    for i in range(len(x)):
        if x[i] < 0.05 and x[i] > -0.05:
            y_values.append(y[i])
            phi_values.append(phi[i])
            rho_values.append(rho[i])
            if diff[i] < 0.01:
                diff_values.append(np.nan)
            else:
                diff_values.append(diff[i])

    # Find the y value for which phi_value goes below 0.5
    threshold = 0.8
    y_threshold = None

    for y_val, phi_val in zip(y_values, phi_values):
        if phi_val < threshold and y_val > 1:
            y_threshold = y_val
            break

    y_threshold = y_threshold - 2
    if y_threshold < 0:
        y_threshold = 0
    
    data_taus.append((trhophi, trho))
    data_tau_rhophis.append((float(trhophi)))
    y_thresholds.append(y_threshold)
    # print(trhophi, trho, y_threshold)

    # Print the y_threshold for the current subdirectory
    # print(f"y_threshold for {trhophi, trho}: {y_threshold}")


# constant params.
tauphi = 2.

eps=0.001
sigma12s = [0.12, 0.18, 0.24, 0.36] # <-- desired sigma_12 value

plot_x_values=[]
plot_y_values=[]

for sigma12_target in sigma12s:
    # tauphirho_values = [1, 1.5625, 2.25, 3.0625, 4, 5.0625, 6.25, 7.5625, 9, 12.25, 16, 20.25, 25, 30.25, 36, 42.25, 49]
    tauphirho_values = [1, 1.5625, 2.25, 3.0625, 4, 5.0625, 6.25, 7.5625, 9, 10.5625, 12.25, 14.0625, 16, 18.0625, 20.25, 22.5625, 25, 27.5625, 30.25, 33.0625, 36, 39.0625, 42.25, 45.5625, 49]
    taurho_values = []
    sigma13_values = []
    real_sigma13s = []

    for tauphirho in tauphirho_values:
        # ---- 1) Compute quantities that depend on tauphirho but not on taurho ----
        #
        # phiint = [ (32/9)*tauphi*eps + (4/9)*tauphirho*eps ] / [ (8/9)*tauphirho*eps ]
        # Simplifies to phiint = 4*(tauphi/tauphirho) + 1/2

        # gamma_phirho1 = sqrt( (8/9)*tauphirho*eps * phiint )
        gamma_phirho1 = np.sqrt((8./9.)*tauphirho*eps)

        # ---- 2) Solve for taurho so that sigma12 stays fixed ----
        # We want sigma12_target = gamma_phirho1 + sqrt((8/9)*taurho*eps).
        # => taurho = ( (sigma12_target - gamma_phirho1)**2 ) / ( (8/9)*eps )
        const_for_taurho = (8./9.) * eps
        # Be careful to avoid negative radicand:
        inside_sqrt = sigma12_target - gamma_phirho1
        
        if inside_sqrt < 0:
            # If gamma_phirho1 alone is already bigger than sigma12_target, 
            # then you cannot keep sigma_12 at that smaller value.
            taurho = None
            sigma12_computed = None
        else:
            taurho = (inside_sqrt**2) / const_for_taurho

            # ---- 3) Re-compute gamma_rho with the solved taurho ----
            gamma_rho = np.sqrt((8./9.)*taurho*eps)

            # Double check sigma12 is indeed ~ sigma12_target
            sigma12_computed = gamma_phirho1 + gamma_rho

        # ---- 4) Now we can compute sigma13, sigma23, etc. if needed ----
        # gamma_phi  = sqrt((8/9)*tauphi*eps)
        gamma_phi = np.sqrt((8./9.)*tauphi*eps)
        
        # gamma_phirho2 = sqrt((4/9)*tauphirho*eps)
        gamma_phirho2 = np.sqrt((4./9.)*tauphirho*eps)

        # sigma13 = gamma_phirho2 + gamma_rho + gamma_phi
        # sigma23 = gamma_phi
        sigma13 = None
        sigma23 = None
        
        if taurho is not None:
            sigma13 = gamma_phirho2 + gamma_rho + gamma_phi
            # newsig12 = gamma_phirho1 = np.sqrt((8./9.)*tauphirho*eps) + gamma_rho
            sigma13_values.append(sigma13/ sigma12_target)
            sigma23 = gamma_phi
            taurho_values.append(taurho)

    constant_list =[]
    sigma13_shorts = []
    for r in range(len(taurho_values)):
        tpr = tauphirho_values[r]
        tr = taurho_values[r]
        constant_list.append((tpr, tr))
        sigma13_shorts.append(sigma13_values[r])
    # print(constant_list, sigma13_shorts)

    plotx=[]
    ploty=[]
    for ir in range(len(data_taus)):
        for iter in range(len(constant_list)):
            # print(str(data_tau), constant_tuple)
            if (abs(data_taus[ir][0] - constant_list[iter][0]) < 0.49 and abs(data_taus[ir][1] - constant_list[iter][1]) < 0.49):
                y_threshold = y_thresholds[ir]
                plotx.append(sigma13_shorts[iter])
                ploty.append(y_threshold)
                # print(sigma13_shorts[iter], y_threshold)
                # print(f"y_threshold for {data_taus[ir]}: {y_threshold}")
                break
            
    # Sort the plot_x_values and plot_y_values based on plot_x_values
    sorted_data = sorted(zip(plotx, ploty), key=lambda x: x[0])

    # Unzip the sorted data
    sorted_plot_x_values, sorted_plot_y_values = zip(*sorted_data)
    
    plot_x_values.append(sorted_plot_x_values)
    plot_y_values.append(sorted_plot_y_values)

# Make matplotlib line plot
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Make matplotlib line plot
plt.figure()
for i in range(len(plot_x_values)):
    plt.plot(plot_x_values[i], plot_y_values[i], 'o-', color=colors[i % len(colors)], label=f'sigmaHL = {sigma12s[i]}')

plt.xlabel('sigmaHM / sigmaHL')
plt.ylabel('elongation')
plt.legend()
plt.show()

    
    
    

