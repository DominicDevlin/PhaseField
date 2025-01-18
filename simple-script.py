import subprocess
import sys
import numpy as np
      

index = 0

if (len(sys.argv) > 1):
    index = int(sys.argv[1])
    print("INDEX IS: ", index)
    # prepend = "xvfb-run -a "


tauphi = 1 


pairs = []
results = []

tauphirho_values =  [4, 9, 16, 25, 36, 49]
for tpr in tauphirho_values:
    
    quadratic_scaling = np.linspace(0.2, 1, 10)**2
    taurho_values = (2) * quadratic_scaling * tpr
    taurho_values = list(taurho_values)
    print(taurho_values)
    for tr in taurho_values:
        pairs.append((tpr, tr))

# Select the pair at the given index
# (Make sure index is valid for pairs; you may want to add a check if needed)
taurhophi, taurho = pairs[index]

tauphi_str = f"{tauphi:.3g}"
taurhophi_str = f"{taurhophi:.3g}"
taurho_str = f"{taurho:.3g}"


print("totals: ", len(pairs))
print("tau phi:", tauphi_str)
print("tau rhophi:", taurhophi_str)
print("tau rho:", taurho_str)

# Build the command as a list of strings
command = ["FreeFem++", "working.edp", tauphi_str, taurhophi_str, taurho_str]

print(f"Running: {' '.join(command)}")
# Execute the command
subprocess.run(command)
