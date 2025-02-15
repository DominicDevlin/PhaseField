import numpy as np
import os
import matplotlib.pyplot as plt

# Load the data from a text file
# The data.txt file should contain the provided rows of numbers
# datan= '-2-42.2-1.72'
# Get a list of all subdirectories in the 'data' directory
data_dir = 'data/10-diff/'
subdirectories = [subdir for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]

tauphirho_values = [16, 36, 64, 100]


# constant params.
tauphi = 1.

eps=0.0002

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

    ylen=6
    xlen=4
    interval = 0.05
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
    print("NUMBER IS: ", subdir )
    print(avg_phi_list)
    # print(bool_list)
    
    curved_length = start_y
    
    for i in range(start_point, len(bool_list)):
        distance = np.sqrt((curvature_list[i] - curvature_list[i-1]) ** 2 + (y_list[i] - y_list[i-1]) ** 2)
        # print("distance is: ", distance)
        curved_length += distance
        if avg_phi_list[i] < 0.07:
        # if bool_list[i] == False and bool_list[i-1] == True:
            y_threshold = y_list[i]
            break
        if (avg_phi_list[i] > avg_phi_list[i-2] + 0.1):
            y_threshold = y_list[i]
            break
    # print(y_threshold)
    print("y threshold is: ", y_threshold)
    print("curved length is: ", curved_length)

    # for y_val, phi_val in zip(y_values, phi_values):
    #     if phi_val < threshold and y_val > 0.5:
    #         y_threshold = y_val
    #         break
    
    if y_threshold is None:
        y_threshold = 6

    y_threshold = y_threshold - 1.8
    if y_threshold < -0.1:
        y_threshold = -0.1
    
    curved_length = curved_length - 1.8
    if curved_length < -0.1:
        curved_length = -0.1
    
    tau_strings.append(ttt)
    y_thresholds.append(curved_length)
    # print(trhophi, trho, y_threshold)



plot_x_values=[]
plot_y_values=[]

sigmaHLvalues=[]

for tpr in tauphirho_values:

    indices = []
    
    plotx=[]
    ploty=[]
    
    for i in range(len(tau_strings)):
        if tau_strings[i][1] == tpr:
            indices.append(i)
    
            
    sigmaHL=0.
    
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
        grrp = const1* tphi + const2 * trho

        # sigmaHM = np.sqrt(np.sqrt(grrp) * np.sqrt(gammarho))
        sigmaHM = np.sqrt(2*eps) * trho / 24
        sigmaHL = np.sqrt(2*eps) * tphirho / 24
        sigmaLM = 0
        print(sigmaHM, sigmaHL)
        sigratio = sigmaHM - sigmaHL
        
        yval = y_thresholds[indices[i]]
        plotx.append(sigratio)
        ploty.append(yval)
        
        # print(trho, tphirho, yval)
    sigmaHLvalues.append(sigmaHL)
    # Sort the plot_x_values and plot_y_values based on plot_x_values
    sorted_data = sorted(zip(plotx, ploty), key=lambda x: x[0])

    # Unzip the sorted data
    sorted_plot_x_values, sorted_plot_y_values = zip(*sorted_data)
    print(sorted_plot_x_values, sorted_plot_y_values)
    
    plot_x_values.append(sorted_plot_x_values)
    plot_y_values.append(sorted_plot_y_values)


from scipy.interpolate import UnivariateSpline
# Make matplotlib line plot
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Make matplotlib line plot
# Create the figure
fig, ax = plt.subplots()


for i in range(len(plot_x_values)):
    # Convert to NumPy arrays
    x = np.array(plot_x_values[i], dtype=float)
    y = np.array(plot_y_values[i], dtype=float)
    
    # Create a finer grid for smooth interpolation
    x_new = np.linspace(x.min(), x.max(), 200)
    
    # Create the spline function (cubic spline by default)
    spline = UnivariateSpline(x, y, s=0.008)

    y_smooth = spline(x_new)
    
    currsig = f"{sigmaHLvalues[i]:.2g}"
    
    # Plot the smooth curve
    ax.plot(
        x_new,
        y_smooth,
        color=colors[i % len(colors)],
        label=f'sigmaHL = {currsig}',
        linewidth=2
    )

    # Optionally, plot the original data points
    ax.plot(x, y, 'o', color=colors[i % len(colors)], markersize=5)
    

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

ax.set_ylim([-0.1, 2.1])

# Show the plot
plt.show()
    
    
    

