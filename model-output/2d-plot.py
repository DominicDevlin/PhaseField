import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
# ---------------------------------------------------------------
# 1. Load the data
#    Assume the data file has three columns: x, y, concentration
#    Replace 'data.csv' with your filename or filepath.
#    If your data is whitespace-separated, use 'delim_whitespace=True'.
# ---------------------------------------------------------------

number = '2-25-2'
prepend = 'data/4diff/' + number + '/'
time = '2-25-2'


df = pd.read_csv(prepend + 'phi_data' + '-' + time + '.dat', header=None, names=['x', 'y', 'conc'], delimiter='\t')
df2 = pd.read_csv(prepend + 'rho_data' + '-' + time + '.dat', header=None, names=['x', 'y', 'conc'], delimiter='\t')

# ---------------------------------------------------------------
# 2. Convert DataFrame columns to NumPy arrays
# ---------------------------------------------------------------
x = df['x'].values 
y = df['y'].values
c = df['conc'].values* (df2['conc'].values * 2 - 1)

# ---------------------------------------------------------------
# 3. Create triangulation for contour plotting
# ---------------------------------------------------------------
triang = Triangulation(x, y)

# ---------------------------------------------------------------
# 4. Plot using tricontourf
# ---------------------------------------------------------------
plt.figure(figsize=(8, 12))

# Optionally, specify the contour levels
# E.g., 50 equally spaced intervals between min and max of c
levels = np.linspace(-1.03, 1.03, 40)

bwr_cmap = get_cmap('bwr')
blue_to_white_cmap = LinearSegmentedColormap.from_list(
    "blue_to_white", bwr_cmap(np.linspace(0, 1, 256))
)

contour_f = plt.tricontourf(triang, c, levels=levels, cmap='bwr')
plt.colorbar(contour_f, label="Concentration")
# Set equal aspect ratio for x and y axes
plt.gca().set_aspect('equal', adjustable='box')

# ---------------------------------------------------------------
# 5. Label axes and add a title
# ---------------------------------------------------------------
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('2D Concentration Distribution')

# ---------------------------------------------------------------
# 6. Show the plot
# ---------------------------------------------------------------
plt.tight_layout()
plt.show()
