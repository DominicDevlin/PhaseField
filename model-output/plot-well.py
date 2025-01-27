import numpy as np
import matplotlib.pyplot as plt

# Load the data from a text file
# The data.txt file should contain the provided rows of numbers
datan= '-2'
dir = 'data/1-36-36/'#  + datan[1:]
if (dir != ''):
    dir = dir + ''


dataphi = np.loadtxt(dir + 'phi_data' + datan + '.dat')

datarho = np.loadtxt(dir + 'rho_data' + datan + '.dat')


dataphi = dataphi[dataphi[:, 0].argsort()]
datarho = datarho[datarho[:, 0].argsort()]

# The first column is the x-axis
x = dataphi[:, 0]
y = dataphi[:, 1]   # y-values in the second column
phi = dataphi[:, 2] # concentration in the third column
rho = datarho[:, 2] # concentration in the third column



x_values=[]
phi_values=[]
rho_values=[]
diff_values = []
well_values = []

for i in range(len(x)):
    if y[i] < 1.02 and y[i] > 0.98:
        x_values.append(x[i])
        phi_values.append(phi[i])
        rho_values.append(rho[i])
        val =  phi[i]
        well = val**2*(1-val)**2
        well_values.append(well)


# Generate a figure and a primary axis
fig, ax1 = plt.subplots()

# Plot the first two lines on the primary y-axis
ax1.plot(x_values, phi_values, linestyle='-', color='black', label='yphi')
ax1.plot(x_values, rho_values, linestyle='-', color='red', label='yrho')


ax1.set_xlabel('X-axis Label')
ax1.set_ylabel('Primary Y-axis Label')
fig.set_size_inches(12, 4)

plt.xlim(-2, 2)

plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(x_values, well_values, linestyle='-', color='black', label='well')
fig2.set_size_inches(12, 4)
plt.xlim(-2, 2)
fig2.tight_layout()
plt.show()
