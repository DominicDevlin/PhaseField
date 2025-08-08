import numpy as np
import matplotlib.pyplot as plt

# Load the data from a text file
# The data.txt file should contain the provided rows of numbers
datan= '0.5'
dir = 'data/init-condition/'#  + datan[1:]
if (dir != ''):
    dir = dir + ''


data1 = np.loadtxt(dir + 'c1-' + datan + '.dat')

data2 = np.loadtxt(dir + 'c2-' + datan + '.dat')

# datadiff = np.loadtxt(dir + 'diff_data' + datan + '.dat')

data1 = data1[data1[:, 1].argsort()]
data2 = data2[data2[:, 1].argsort()]
# data3 = datadiff[datadiff[:, 1].argsort()]

# The first column is the x-axis
x = data1[:, 0]
y = data1[:, 1]   # y-values in the second column
c1 = data1[:, 2] # concentration in the third column
c2 = data2[:, 2] # concentration in the third column
diff = c1**2 * c2**2



y_values=[]
c1_values=[]
c2_values=[]
c3_values=[]
diff_values = []
well_values = []

for i in range(len(x)):
    if x[i] < 0.02 and x[i] > -0.02:
        y_values.append(y[i])
        c1_values.append(c1[i])
        c2_values.append(c2[i])
        c3_values.append(1-c1[i]-c2[i])
        diff_values.append(diff[i])
        val =  c1[i]
        well = val**2*(1-val)**2
        well_values.append(well)


# Generate a figure and a primary axis
fig, ax1 = plt.subplots()

# Plot the first two lines on the primary y-axis
ax1.plot(y_values, c1_values, linestyle='-', color='red', label='yc1', linewidth=3)
ax1.plot(y_values, c2_values, linestyle='-', color='lightgreen', label='yc2', linewidth=3)
ax1.plot(y_values, c3_values, linestyle='-', color='blue', label='yc2', linewidth=3)

ax2 = ax1.twinx()
# Plot the third line on the secondary y-axis
ax2.plot(y_values, diff_values, linestyle='--', color='black', label='ydiff')
ax2.set_ylabel('Secondary Y-axis Label')


ax1.set_xlabel('X-axis Label')
ax1.set_ylabel('Primary Y-axis Label')
fig.set_size_inches(12, 4)

plt.xlim(0, 3)

plt.tight_layout()
plt.show()

# fig2, ax2 = plt.subplots()
# ax2.plot(y_values, well_values, linestyle='-', color='black', label='well')
# fig2.set_size_inches(12, 4)
# plt.xlim(0, 5)
# fig2.tight_layout()
# plt.show()
