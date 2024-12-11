import numpy as np
import matplotlib.pyplot as plt

# Load the data from a text file
# The data.txt file should contain the provided rows of numbers
dataphi = np.loadtxt('phi_data.dat')
dataphi = dataphi[dataphi[:, 0].argsort()]
# The first column is the x-axis
x = dataphi[:, 0]


datarho = np.loadtxt('rho_data.dat')
datarho = datarho[datarho[:, 0].argsort()]
# The first column is the x-axis
xr = datarho[:, 0]

datadiff = np.loadtxt('diff_data.dat')
datadiff = datadiff[datarho[:, 0].argsort()]
# The first column is the x-axis
xd = datadiff[:, 0]


# Suppose we know the times corresponding to each concentration column.
# For example, if you have 10 columns after x, and they represent times t0, t1, t2, ...
# Here, we just assume some times for demonstration.
# Replace these times with the actual times that your columns represent.
times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Just an example mapping

# Choose the desired time (e.g., t = 3)
desired_time = 2
# Find the column index for this time:a
# times[0] corresponds to data[:,1], times[1] -> data[:,2], etc.
time_index = times.index(desired_time) + 1  # +1 because x is in column 0

# Extract the corresponding concentration column
yphi = dataphi[:, time_index]

yrho = datarho[:, time_index]

ydiff = datadiff[:, time_index]




# Generate a figure and a primary axis
fig, ax1 = plt.subplots()

# Plot the first two lines on the primary y-axis
ax1.plot(x, yphi, linestyle='-', color='black', label='yphi')
ax1.plot(x, yrho, linestyle='-', color='red', label='yrho')
ax1.set_xlabel('X-axis Label')
ax1.set_ylabel('Primary Y-axis Label')

# Create a secondary axis that shares the x-axis
ax2 = ax1.twinx()

# Plot the third line on the secondary y-axis
ax2.plot(x, ydiff, linestyle='-', color='blue', label='ydiff')
ax2.set_ylabel('Secondary Y-axis Label')

# Optionally, you can adjust the scale of the secondary y-axis
# For example, if you want a log scale:
# ax2.set_yscale('log')

# Add legends if needed
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

plt.tight_layout()
plt.show()
