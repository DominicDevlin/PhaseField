import subprocess

# Define the range for the two parameters, e.g., 1 to 10.
startphi, endphi = 1, 1
statrho, endrho = 1, 40

for i in range(startphi, endphi+1):
    for j in range(statrho, endrho + 1):
        # Build the command as a list of strings
        command = ["FreeFem++", "oned-working.edp", str(i), str(j)]
        
        print(f"Running: {' '.join(command)}")
        # Execute the command
        subprocess.run(command)
