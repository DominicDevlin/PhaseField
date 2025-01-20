import numpy as np
import os
import matplotlib.pyplot as plt

# Load the data from a text file
# The data.txt file should contain the provided rows of numbers
# datan= '-2-42.2-1.72'
# Get a list of all subdirectories in the 'data' directory
data_dir = 'data/lowd/'
subdirectories = [subdir for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]

tauphirho_values = [9, 16, 25, 36]


# constant params.
tauphi = 2.

eps=0.001

y_thresholds = []
tau_strings = []


# Iterate through each subdirectory
for subdir in subdirectories:
    dir = data_dir + subdir + '/'
    datan = '-' + subdir
    # Separate the numbers in datan and store them as variables
    datan_numbers = datan.split('-')[1:]
    tphi = float(datan_numbers[0])
    trhophi = float(datan_numbers[1])
    trho = float(datan_numbers[2])
    ttt = [tphi, trhophi, trho]

    


    # Load the data from the current subdirectory
    if not os.path.exists(dir + 'phi_data' + datan + '.dat') or not os.path.exists(dir + 'rho_data' + datan + '.dat') or not os.path.exists(dir + 'diff_data' + datan + '.dat'):
        print("Skipping", subdir)
        continue
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
            # if diff[i] < 0.01:
            #     diff_values.append(np.nan)
            # else:
            #     diff_values.append(diff[i])

    # Find the y value for which phi_value goes below 0.x
    threshold = 0.5
    y_threshold = None

    for y_val, phi_val in zip(y_values, phi_values):
        if phi_val < threshold and y_val > 0.5:
            y_threshold = y_val
            break
    
    if y_threshold is None:
        y_threshold = 6

    y_threshold = y_threshold - 2
    if y_threshold < 0:
        y_threshold = 0
    
    tau_strings.append(ttt)
    y_thresholds.append(y_threshold)
    # print(trhophi, trho, y_threshold)



plot_x_values=[]
plot_y_values=[]


for tpr in tauphirho_values:

    indices = []
    
    plotx=[]
    ploty=[]
    
    for i in range(len(tau_strings)):
        if tau_strings[i][1] == tpr:
            indices.append(i)
    
            
    for i in range(len(indices)):


        tphi = tau_strings[indices[i]][0]
        tphirho = tau_strings[indices[i]][1]
        trho = tau_strings[indices[i]][2]
        const1 = (5*np.sqrt(2)/12.) * eps
        const2 = (5*np.sqrt(2)/12.) * eps * 0.5
        # Be careful to avoid negative radicand:
        gammaphi = (const1 * tphi)
        gammaphirho = const2 * tphirho
        gammarho = (const2 * trho)

        sigmaHM = np.sqrt(np.sqrt(gammaphi) * np.sqrt(gammarho) + gammarho )
        sigmaHL = np.sqrt(gammaphirho)
        sigmaLM = 0

        sigratio = sigmaHM/sigmaHL
        yval = y_thresholds[indices[i]]
        plotx.append(sigratio)
        ploty.append(yval)
        
        print(trho, tphirho, yval)

    # Sort the plot_x_values and plot_y_values based on plot_x_values
    sorted_data = sorted(zip(plotx, ploty), key=lambda x: x[0])

    # Unzip the sorted data
    sorted_plot_x_values, sorted_plot_y_values = zip(*sorted_data)
    print(sorted_plot_x_values, sorted_plot_y_values)
    
    plot_x_values.append(sorted_plot_x_values)
    plot_y_values.append(sorted_plot_y_values)




# Make matplotlib line plot
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Make matplotlib line plot
# Create the figure
fig, ax = plt.subplots()

# Plot each line explicitly
for i in range(len(plot_x_values)):
    ax.plot(
        plot_x_values[i],
        plot_y_values[i],
        'o-',
        color=colors[i % len(colors)],
        label=f'sigmaHL = {tauphirho_values[i]}'
    )

# Set tick properties explicitly
ax.tick_params(
    axis='both',            # Apply to both axes
    which='both',           # Major and minor ticks
    direction='in',         # Tick direction
    labelsize=14,           # Label font size
    width=1.5,              # Tick line width
    length=5              # Tick length
)

# Set labels explicitly
ax.set_xlabel('sigmaHM / sigmaHL', fontsize=16, family='Helvetica')  # Font size and family
ax.set_ylabel('elongation', fontsize=16, family='Helvetica')

ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# Add legend
ax.legend(fontsize=12)

# Show the plot
plt.show()
    
    
    

