import subprocess
import sys
import numpy as np
import sympy as sp


index = 12

if (len(sys.argv) > 1):
    index = int(sys.argv[1])
    print("INDEX IS: ", index)
    # prepend = "xvfb-run -a "


output = []
results = []
eps = 0.02

gamma12_values =  [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]# 0.12, 0.15]
gamma23_values = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]

gamma13 = 0.06


for gamma12 in gamma12_values:
    for gamma23 in gamma23_values:
        sig12 = 3 * gamma12 / eps
        sig13 = 3 * gamma13 / eps
        sig23 = 3 * gamma23 / eps
        output.append((sig12, sig13, sig23))
        print(sig12, sig13, sig23)

sigma12, sigma13, sigma23 = output[index]



sigma12_str = f"{sigma12:.4g}"
sigma13_str = f"{sigma13:.4g}"
sigma23_str = f"{sigma23:.4g}"
eps_str = f"{eps:.4g}"


print("totals: ", len(output))
print("sig12:", sigma12_str)
print("sig13:", sigma13_str)
print("sig23:", sigma23_str)

# Build the command as a list of strings
command = ["FreeFem++", "three-field.edp", sigma12_str, sigma13_str, sigma23_str, eps_str]

print(f"Running: {' '.join(command)}")
# Execute the command
# subprocess.run(command)
