import subprocess
import sys
import numpy as np
      

index = 0

if (len(sys.argv) > 1):
    index = int(sys.argv[1])
    print("INDEX IS: ", index)
    # prepend = "xvfb-run -a "

# constant params.
tauphi = 8.

eps=0.001
sigma12s = [0.12, 0.18, 0.24, 0.3, 0.36]  # <-- desired sigma_12 value


pairs = []
results = []


for sigma12_target in sigma12s:
    # tauphirho_values = [1, 1.5625, 2.25, 3.0625, 4, 5.0625, 6.25, 7.5625, 9, 12.25, 16, 20.25, 25, 30.25, 36, 42.25, 49]
    # tauphirho_values = [1, 1.5625, 2.25, 3.0625, 4, 5.0625, 6.25, 7.5625, 9, 10.5625, 12.25, 14.0625, 16, 18.0625, 20.25, 22.5625, 25, 27.5625, 30.25, 33.0625, 36, 39.0625, 42.25, 45.5625, 49]
    tauphirho_values = [1, 1.5625, 2.25, 3.0625, 4, 5.0625, 6.25, 7.5625, 9, 10.5625, 12.25, 14.0625, 16, 18.0625, 20.25, 22.5625, 25, 27.5625, 30.25, 33.0625, 36, 39.0625, 42.25, 45.5625, 49, 52.5625, 56.25, 60.0625, 64, 68.0625, 72.25, 76.5625, 81, 85.5625, 90.25, 95.0625, 100]

    taurho_values = []

    for tauphirho in tauphirho_values:
        # ---- 1) Compute quantities that depend on tauphirho but not on taurho ----
        #
        # phiint = [ (32/9)*tauphi*eps + (4/9)*tauphirho*eps ] / [ (8/9)*tauphirho*eps ]
        # Simplifies to phiint = 4*(tauphi/tauphirho) + 1/2
        phiint = 4.0*(tauphi/tauphirho) + 0.5
        if (phiint > 1):
            phiint = 1
        # gamma_phirho1 = sqrt( (8/9)*tauphirho*eps * phiint )
        gamma_phirho1 = np.sqrt((8./9.)*tauphirho*eps)


        # ---- 2) Solve for taurho so that sigma12 stays fixed ----
        # We want sigma12_target = gamma_phirho1 + sqrt((8/9)*taurho*eps).
        # => taurho = ( (sigma12_target - gamma_phirho1)**2 ) / ( (8/9)*eps )
        const_for_taurho = (8./9.) * eps
        # Be careful to avoid negative radicand:
        inside_sqrt = sigma12_target - gamma_phirho1
        
        if inside_sqrt < 0:
            # If gamma_phirho1 alone is already bigger than sigma12_target, 
            # then you cannot keep sigma_12 at that smaller value.
            taurho = None
            sigma12_computed = None
            msg = "Impossible to keep sigma12 at {} because gamma_phirho1 already exceeds it!".format(sigma12_target)
            print(msg)
        else:
            taurho = (inside_sqrt**2) / const_for_taurho

            # ---- 3) Re-compute gamma_rho with the solved taurho ----
            gamma_rho = np.sqrt((8./9.)*taurho*eps)

            # Double check sigma12 is indeed ~ sigma12_target
            sigma12_computed = gamma_phirho1 + gamma_rho

        # ---- 4) Now we can compute sigma13, sigma23, etc. if needed ----
        # gamma_phi  = sqrt((8/9)*tauphi*eps)
        gamma_phi = np.sqrt((8./9.)*tauphi*eps)
        
        # gamma_phirho2 = sqrt((4/9)*tauphirho*eps)
        gamma_phirho2 = np.sqrt((4./9.)*tauphirho*eps)
        
        # print(gamma_phirho1, gamma_phirho2)

        # sigma13 = gamma_phirho2 + gamma_rho + gamma_phi
        # sigma23 = gamma_phi
        sigma13 = None
        sigma23 = None
        
        if taurho is not None:
            sigma13 = gamma_phirho2 + gamma_rho + gamma_phi
            sigma12 = gamma_phirho1 + gamma_rho
            # print(sigma12, sigma13, gamma_phirho1, gamma_phirho2)
            
            sigma23 = gamma_phi
            taurho_values.append(taurho)

        # ---- 5) Store or print results ----
        results.append({
            'tauphirho'   : tauphirho,
            'taurho'      : taurho,
            'sigma12_fix' : sigma12_computed,
            'sigma13'     : sigma13,
            'sigma23'     : sigma23,
            'phiint'      : phiint,
        })
    # Create all pairs of (gamma_phi, gamma_rho)
    for r in range(len(taurho_values)):
        pairs.append((tauphirho_values[r], taurho_values[r]))


# Print the table of results
print("   tauphirho     taurho         sigma12     sigma13     sigma23      phiint")
for res in results:
    print(
        f"{res['tauphirho']:>10} "
        f"{res['taurho'] if res['taurho'] is not None else 'N/A':>10} "
        f"{res['sigma12_fix'] if res['sigma12_fix'] else 'N/A':>10} "
        f"{res['sigma13'] if res['sigma13'] else 'N/A':>10} "
        f"{res['sigma23'] if res['sigma23'] else 'N/A':>10}"
        f"{res['phiint'] if res['phiint'] else 'N/A':>10}"
    )

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