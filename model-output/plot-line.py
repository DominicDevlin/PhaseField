import numpy as np
import matplotlib.pyplot as plt

# Load the data from a text file
# The data.txt file should contain the provided rows of numbers
dataphi = np.loadtxt('phi_data.dat')
dataphi = dataphi[dataphi[:, 1].argsort()]
# The first column is the x-axis
x = dataphi[:, 0]
y = dataphi[:, 1]   # y-values in the second column
phi = dataphi[:, 2] # concentration in the third column

y_values=[]
phi_values=[]

for i in range(len(x)):
    if x[i] < 0.01 and x[i] > -0.01:
        y_values.append(y[i])
        phi_values.append(phi[i])




# Generate a figure and a primary axis
fig, ax1 = plt.subplots()

# Plot the first two lines on the primary y-axis
ax1.plot(y_values, phi_values, linestyle='-', color='black', label='yphi')
ax1.set_xlabel('X-axis Label')
ax1.set_ylabel('Primary Y-axis Label')


plt.tight_layout()
plt.show()
