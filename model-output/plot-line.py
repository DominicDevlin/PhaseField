import numpy as np
import matplotlib.pyplot as plt

# Load the data from a text file
# The data.txt file should contain the provided rows of numbers
datan= '-2-25-10.9'
dir = 'data/diffhi/' + datan[1:]
if (dir != ''):
    dir = dir + '/'


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
well_values = []

for i in range(len(x)):
    if x[i] < 0.02 and x[i] > -0.02:
        y_values.append(y[i])
        phi_values.append(phi[i])
        rho_values.append(rho[i])
        if diff[i] < 0.01:
            diff_values.append(np.nan)
        else:
            diff_values.append(diff[i])
        val =  phi[i]
        well = val**2*(1-val)**2
        well_values.append(well)


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
fig.set_size_inches(12, 4)

plt.xlim(0, 5)

plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(y_values, well_values, linestyle='-', color='black', label='well')
fig2.set_size_inches(12, 4)
plt.xlim(0, 5)
fig2.tight_layout()
plt.show()
