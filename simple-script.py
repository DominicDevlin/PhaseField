import subprocess
import sys
import numpy as np
      

index = 0

if (len(sys.argv) > 1):
    index = int(sys.argv[1])
    print("INDEX IS: ", index)
    # prepend = "xvfb-run -a "

# this needs to be 2 instead of 1 otherwise phi falls too fast. 
tauphi = 2 


pairs = []
results = []

tauphirho_values =  [9, 25, 49, 81]
for tpr in tauphirho_values:
    
    quadratic_scaling = (np.linspace(0.2, 1, 20)**2)*30-15
    taurho_values = quadratic_scaling + tpr*0.83
    taurho_values = list(taurho_values)
    taurho_values = [tr for tr in taurho_values if tr >= 0]
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
# subprocess.run(command)
