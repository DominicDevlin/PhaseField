import subprocess
import sys



        

index = 0

if (len(sys.argv) > 1):
    index = int(sys.argv[1])
    print("INDEX IS: ", index)
    # prepend = "xvfb-run -a "


gamma_phi = 2.
gamma_rhophi = 1.
gamma_rho = 1.

# gamma_phi_vals should be [1, 2, 3, 4, 5]
gamma_rhophi_vals = list(range(2, 21, 4))
# gamma_rho_vals should be between 1 and 20 (inclusive)
gamma_rho_vals = list(range(2, 21, 4))

# Create all pairs of (gamma_phi, gamma_rho)
pairs = []
for r1 in gamma_rhophi_vals:
    for r2 in gamma_rho_vals:
        pairs.append((r1, r2))

# Select the pair at the given index
# (Make sure index is valid for pairs; you may want to add a check if needed)
gamma_rhophi, gamma_rho = pairs[index]

print("gamma_phi:", gamma_phi)
print("gamma_rho:", gamma_rhophi)
print("gamma_rho:", gamma_rho)

# Build the command as a list of strings
command = ["FreeFem++", "working.edp", str(gamma_phi), str(gamma_rhophi), str(gamma_rho)]

print(f"Running: {' '.join(command)}")
# Execute the command
subprocess.run(command)