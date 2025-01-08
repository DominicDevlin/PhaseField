import numpy as np
import matplotlib.pyplot as plt

# Load the data from a text file
# The data.txt file should contain the provided rows of numbers
datan= '-2-42.2-1.72'
dir = 'data/' + datan[1:] +'/'


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
    if x[i] < 0.01 and x[i] > -0.01:
        y_values.append(y[i])
        phi_values.append(phi[i])
        rho_values.append(rho[i])
        if diff[i] < 0.01:
            diff_values.append(np.nan)
        else:
            diff_values.append(diff[i])




# Generate a figure and a primary axis
fig, ax1 = plt.subplots()

# Plot the first two lines on the primary y-axis
ax1.plot(y_values, phi_values, linestyle='-', color='black', label='yphi')
ax1.plot(y_values, rho_values, linestyle='-', color='red', label='yrho')

ax2 = ax1.twinx()
# Plot the third line on the secondary y-axis
ax2.plot(y_values, diff_values, linestyle='-', color='blue', label='ydiff')
ax2.set_ylabel('Secondary Y-axis Label')


ax1.set_xlabel('X-axis Label')
ax1.set_ylabel('Primary Y-axis Label')


plt.tight_layout()
plt.show()
