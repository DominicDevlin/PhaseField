import numpy as np
from scipy.signal import argrelextrema
import os
import glob
import re
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import math
import matplotlib.pyplot as plt



eps=0.001
tauphi=2.0
sigma12_target = 0.2  # <-- your desired sigma_12 value


tauphirho_values = [1, 4, 9, 16, 25, 36, 49, 64]  # examples


results = []


for tauphirho in tauphirho_values:

    # ---- 1) Compute quantities that depend on tauphirho but not on taurho ----
    #
    # phiint = [ (32/9)*tauphi*eps + (4/9)*tauphirho*eps ] / [ (8/9)*tauphirho*eps ]
    # Simplifies to phiint = 4*(tauphi/tauphirho) + 1/2
    phiint = 4.0*(tauphi/tauphirho) + 0.5

    # gamma_phirho1 = sqrt( (8/9)*tauphirho*eps * phiint )
    gamma_phirho1 = np.sqrt((8./9.)*tauphirho*eps*phiint)

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

    # sigma13 = gamma_phirho2 + gamma_rho + gamma_phi
    # sigma23 = gamma_phi
    sigma13 = None
    sigma23 = None
    
    if taurho is not None:
        sigma13 = gamma_phirho2 + gamma_rho + gamma_phi
        sigma23 = gamma_phi

    # ---- 5) Store or print results ----
    results.append({
        'tauphirho'   : tauphirho,
        'taurho'      : taurho,
        'sigma12_fix' : sigma12_computed,
        'sigma13'     : sigma13,
        'sigma23'     : sigma23,
    })

# Print the table of results
print("   tauphirho     taurho         sigma12     sigma13     sigma23")
for res in results:
    print(
        f"{res['tauphirho']:>10} "
        f"{res['taurho'] if res['taurho'] is not None else 'N/A':>10} "
        f"{res['sigma12_fix'] if res['sigma12_fix'] else 'N/A':>10} "
        f"{res['sigma13'] if res['sigma13'] else 'N/A':>10} "
        f"{res['sigma23'] if res['sigma23'] else 'N/A':>10}"
    )