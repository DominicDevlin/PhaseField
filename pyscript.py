import subprocess

# Define the range for the two parameters, e.g., 1 to 10.
start, end = 1, 20

for i in range(start, end + 1):
    for j in range(start, end + 1):
        # Build the command as a list of strings
        command = ["FreeFem++", "oned-working.edp", str(i), str(j)]
        
        print(f"Running: {' '.join(command)}")
        # Execute the command
        subprocess.run(command)
