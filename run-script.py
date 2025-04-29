import subprocess
import sys
import numpy as np
      

index = 0

if (len(sys.argv) > 1):
    index = int(sys.argv[1])
    print("INDEX IS: ", index)
    # prepend = "xvfb-run -a "

# this needs to be 2 instead of 1 otherwise phi falls too fast. 
sigma12 = 30


pairs = []
results = []

sigma13_values =  [10, 20, 30, 40, 50]
sigma23_values = [10, 20, 30, 40, 50]
for tpr in sigma13_values:
    for tr in sigma23_values:
        pairs.append((tpr, tr))

sigma13, sigma23 = pairs[index]



sigma12_str = f"{sigma12:.3g}"
sigma13_str = f"{sigma13:.3g}"
sigma23_str = f"{sigma23:.3g}"


print("totals: ", len(pairs))
print("tau phi:", sigma12_str)
print("tau rhophi:", sigma13_str)
print("tau rho:", sigma12_str)

# Build the command as a list of strings
command = ["FreeFem++", "three-field.edp", sigma12_str, sigma13_str, sigma12_str]

print(f"Running: {' '.join(command)}")
# Execute the command
subprocess.run(command)
