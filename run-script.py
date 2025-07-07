import subprocess
import sys
import numpy as np
import sympy as sp

def get_unknowns(gamma12, gamma13, gamma23, eps):
    # Define unknowns
    sig12, sig13, sig23 = sp.symbols('sig12 sig13 sig23', positive=True, real=True)


    # Define the three equations:
    eq1 = eps/6 * sp.sqrt(2*sig12*(4*sig12 + sig13 + sig23)) - gamma12
    eq2 = eps/6 * sp.sqrt(2*sig13*(4*sig13 + sig12 + sig23)) - gamma13
    eq3 = eps/6 * sp.sqrt(2*sig23*(4*sig23 + sig12 + sig13)) - gamma23

    # Provide an initial guess for [sig12, sig13, sig23]:
    initial_guess = [5.0, 5.0, 5.0]

    # Solve numerically
    solution = sp.nsolve([eq1, eq2, eq3], [sig12, sig13, sig23], initial_guess)

    # Unpack and display
    sig12_sol, sig13_sol, sig23_sol = solution
    # print(f"sig12 = {sig12_sol:.6f}")
    # print(f"sig13 = {sig13_sol:.6f}")
    # print(f"sig23 = {sig23_sol:.6f}")
    return sig12_sol, sig13_sol, sig23_sol


index = 0

if (len(sys.argv) > 1):
    index = int(sys.argv[1])
    print("INDEX IS: ", index)
    # prepend = "xvfb-run -a "


output = []
results = []
eps = 0.015

gamma12_values =  [0.02, 0.04, 0.06, 0.08, 0.1]
gamma23_values = [0.02, 0.04, 0.06, 0.08, 0.1]

gamma13 = 0.03



for gamma12 in gamma12_values:
    for gamma23 in gamma23_values:
        sig12, sig13, sig23 = get_unknowns(gamma12, gamma13, gamma23, eps)
        output.append((sig12, sig13, sig23))
        # print(sig12, sig13, sig12)

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
subprocess.run(command)
