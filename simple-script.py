import subprocess
import sys
import numpy as np
      

index = 0

if (len(sys.argv) > 1):
    index = int(sys.argv[1])
    print("INDEX IS: ", index)
    # prepend = "xvfb-run -a "

# constant params.
tauphi = 2.

eps=0.001
tauphi = 1  # <-- desired sigma_12 value


pairs = []
results = []

tauphirho_values =  [1, 2, 4, 9, 16]
for tr in tauphirho_values:
    
    taurho_values = np.linspace(0.2, 2, 10) * tr
    taurho_values = list(taurho_values)
        
    for tpr in taurho_values:
        pairs.append((tpr, tr))
        print(tpr, tr)

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