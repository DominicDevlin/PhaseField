import subprocess
import sys
import numpy as np
      

index = 0

if (len(sys.argv) > 1):
    index = int(sys.argv[1])
    print("INDEX IS: ", index)
    # prepend = "xvfb-run -a "


sigma13 = 3


pairs = []
results = []

sigma12_values =  [3, 6, 9, 12, 15]
sigma23_values = [3, 6, 9, 12, 15]
for tpr in sigma12_values:
    for tr in sigma23_values:
        pairs.append((tpr, tr))

sigma12, sigma23 = pairs[index]



sigma12_str = f"{sigma12:.3g}"
sigma13_str = f"{sigma13:.3g}"
sigma23_str = f"{sigma23:.3g}"


print("totals: ", len(pairs))
print("sig12:", sigma12_str)
print("sig13:", sigma13_str)
print("sig23:", sigma23_str)

# Build the command as a list of strings
command = ["FreeFem++", "three-field.edp", sigma12_str, sigma13_str, sigma12_str]

print(f"Running: {' '.join(command)}")
# Execute the command
subprocess.run(command)
