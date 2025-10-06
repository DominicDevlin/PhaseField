#!/usr/bin/env python3


main_directory = 'data/equilibrium-long'
choose_time = "1000"

import argparse
import math
import sys
from typing import List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap

import os
import re
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ----------------------------
# IO + interpolation utilities
# ----------------------------

def load_field(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path}: expected 3 columns [x,y,c], got shape {arr.shape}")
    return arr[:, 0], arr[:, 1], arr[:, 2]

def build_linear_interpolator(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    tri = mtri.Triangulation(x, y)
    return mtri.LinearTriInterpolator(tri, z)

def common_bbox(x1, y1, x2, y2, margin=0.02):
    bbox1 = (x1.min(), x1.max(), y1.min(), y1.max())
    bbox2 = (x2.min(), x2.max(), y2.min(), y2.max())
    xmin = max(bbox1[0], bbox2[0])
    xmax = min(bbox1[1], bbox2[1])
    ymin = max(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    if not (xmax > xmin and ymax > ymin):
        raise ValueError("Input point sets have disjoint bounding boxes; no overlap.")
    dx = xmax - xmin; dy = ymax - ymin
    return xmin + margin*dx, xmax - margin*dx, ymin + margin*dy, ymax - margin*dy

# ----------------------------
# Contours + intersections
# ----------------------------

def contour_paths_from_grid(X, Y, Z, level=0.0) -> List[np.ndarray]:
    """Return list of Nx2 polylines (level-set contours)."""
    fig = plt.figure()
    try:
        cs = plt.contour(X, Y, Z, levels=[level])
        lines = []
        if cs.collections:
            for coll in cs.collections:
                for path in coll.get_paths():
                    v = path.vertices
                    if v.shape[0] >= 2:
                        lines.append(v.copy())
        return lines
    finally:
        plt.close(fig)

def seg_intersection(p, r, q, s):
    """Segment intersection p->r with q->s."""
    p = np.asarray(p); r = np.asarray(r); q = np.asarray(q); s = np.asarray(s)
    u = r - p; v = s - q; w = p - q
    denom = u[0]*v[1] - u[1]*v[0]
    eps = 1e-12 * (np.linalg.norm(u) + np.linalg.norm(v) + 1.0)
    if abs(denom) < eps:
        return False, None
    t = (v[0]*w[1] - v[1]*w[0]) / denom
    uparam = (u[0]*w[1] - u[1]*w[0]) / denom
    if -1e-10 <= t <= 1+1e-10 and -1e-10 <= uparam <= 1+1e-10:
        return True, p + t*u
    return False, None

def pairwise_intersections(linesA: List[np.ndarray], linesB: List[np.ndarray]) -> List[np.ndarray]:
    pts = []
    for a in linesA:
        for i in range(a.shape[0] - 1):
            p = a[i]; r = a[i+1]
            for b in linesB:
                for j in range(b.shape[0] - 1):
                    q = b[j]; s = b[j+1]
                    hit, pt = seg_intersection(p, r, q, s)
                    if hit:
                        pts.append(pt)
    return pts

def cluster_points(pts: List[np.ndarray], eps: float) -> List[np.ndarray]:
    centers = []
    groups = []
    for p in pts:
        assigned = False
        for gi, c in enumerate(centers):
            if np.linalg.norm(p - c) <= eps:
                groups[gi].append(p)
                centers[gi] = np.mean(groups[gi], axis=0)
                assigned = True
                break
        if not assigned:
            centers.append(p.copy())
            groups.append([p])
    return [np.mean(g, axis=0) for g in groups]

# ----------------------------
# Gradients + angles
# ----------------------------

def grad_from_interpolant(f, x, y, h):
    """Central difference gradient of scalar interpolant f at (x,y)."""
    def val(xx, yy):
        vv = f(xx, yy)
        if np.ma.is_masked(vv):
            raise ValueError("Point outside interpolation hull.")
        return float(vv)
    fx = (val(x + h, y) - val(x - h, y)) / (2*h)
    fy = (val(x, y + h) - val(x, y - h)) / (2*h)
    return np.array([fx, fy], dtype=float)

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def angle_between(u, v):
    cu = unit(u); cv = unit(v)
    dot = np.clip(np.dot(cu, cv), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


# ----------------------------
# Main
# ----------------------------

def get_angle(c1path, c2path):


    # ap = argparse.ArgumentParser()
    # ap.add_argument("--c1", default=defaultf1, help="path to c1 data file")
    # ap.add_argument("--c2", default=defaultf2, help="path to c2 data file")
    # ap.add_argument("--nx", type=int, default=500, help="grid resolution in x (y auto)")
    # ap.add_argument("--plot", default=None, help="optional output figure path (PNG/SVG)")
    # ap.add_argument("--phi_tol", type=float, default=5e-3, help="|phi31| tolerance for triple-point filtering")
    # ap.add_argument("--report-small-angles", action="store_true",
    #                 help="also print the three small crossing angles that sum to 180°")
    # args = ap.parse_args()

    # Load scattered fields
    x1, y1, c1v = load_field(c1path)
    x2, y2, c2v = load_field(c2path)

    f1 = build_linear_interpolator(x1, y1, c1v)
    f2 = build_linear_interpolator(x2, y2, c2v)

    xmin, xmax, ymin, ymax = common_bbox(x1, y1, x2, y2, margin=0.02)
    nx = 100#max(256, args.nx)
    ny = 100#max(256, int(round(nx * (ymax - ymin) / (xmax - xmin))))
    print(nx, ny)
    X = np.linspace(xmin, xmax, nx)
    Y = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(X, Y, indexing='xy')

    C1 = f1(XX, YY)
    C2 = f2(XX, YY)
    mask = np.ma.getmaskarray(C1) | np.ma.getmaskarray(C2)
    C1 = np.ma.array(C1, mask=mask)
    C2 = np.ma.array(C2, mask=mask)
    C3 = np.ma.array(1.0 - C1 - C2, mask=mask)

    # Pairwise interface fields (zero level sets)
    PHI12 = np.ma.array(C1 - C2, mask=mask)
    PHI23 = np.ma.array(C2 - C3, mask=mask)
    PHI31 = np.ma.array(C3 - C1, mask=mask)

    # Extract zero-contours
    lines12 = contour_paths_from_grid(XX, YY, PHI12.filled(np.nan), level=0.0)
    lines23 = contour_paths_from_grid(XX, YY, PHI23.filled(np.nan), level=0.0)
    lines31 = contour_paths_from_grid(XX, YY, PHI31.filled(np.nan), level=0.0)

    if len(lines12) == 0 or len(lines23) == 0:
        print("No interfaces found (phi12 or phi23 has no zero-level contours in the overlap).")
        return 180

    # Intersections of phi12=0 and phi23=0 -> triple-point candidates
    raw_pts = pairwise_intersections(lines12, lines23)

    # Filter: require |phi31| ~ 0 at candidate
    def phi31_val(x, y) -> float:
        # nearest grid sample if available; otherwise direct eval
        iy = int(np.clip(np.argmin(np.abs(Y - y)), 0, ny-1))
        ix = int(np.clip(np.argmin(np.abs(X - x)), 0, nx-1))
        v = PHI31[iy, ix]
        if not np.ma.is_masked(v):
            return float(v)
        vv = (1.0 - f1(x, y) - f2(x, y)) - f1(x, y)  # = c3 - c1
        return float(vv) if not np.ma.is_masked(vv) else np.inf

    candidates = [p for p in raw_pts]# if abs(phi31_val(p[0], p[1])) <= args.phi_tol]
    if not candidates:
        print("Intersections found, but none satisfy |phi31|<=tol near zero; try increasing --phi_tol.")
        return 0
        sys.exit(0)

    # Cluster nearby intersections -> at most two triple points
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    grid_step = 0.5 * (dx + dy)
    centers = cluster_points(candidates, eps=2.0*grid_step)

    # Finite-difference step for gradients
    h = 2.0 * grid_step

    results = []
    for (xt, yt) in centers:
        g1 = grad_from_interpolant(f1, xt, yt, h)
        g2 = grad_from_interpolant(f2, xt, yt, h)
        g3 = -(g1 + g2)  # since c3 = 1 - c1 - c2

        # small crossing angles αk between bounding interfaces for phase k
        a1 = angle_between(g1 - g2, g1 - g3)  # (1–2) vs (1–3)
        a2 = angle_between(g2 - g3, g2 - g1)  # (2–3) vs (2–1)
        a3 = angle_between(g3 - g1, g3 - g2)  # (3–1) vs (3–2)

        # interior contact angles θk (supplements)
        th1 = 180.0 - a1
        th2 = 180.0 - a2
        th3 = 180.0 - a3

        results.append({
            "triple_point": (xt, yt),
            "angles_deg": (th1, th2, th3),
            "small_deg": (a1, a2, a3),
            "gradients": (g1, g2, g3)
        })
    lowest_angle = 360
    # ---- Printout ----
    print("\nDetected triple junctions (interior angles shown; sum ~360°):\n")
    for i, res in enumerate(results, 1):
        (xt, yt) = res["triple_point"]
        th1, th2, th3 = res["angles_deg"]
        a1, a2, a3 = res["small_deg"]

        print(f"#{i}: at (x={xt:.6g}, y={yt:.6g})")
        print(f"    θ1 (inside phase-1): {th1:8.3f}°")
        print(f"    θ2 (inside phase-2): {th2:8.3f}°")
        print(f"    θ3 (inside phase-3): {th3:8.3f}°")
        print(f"    angle sum (should be ~360): {(th1+th2+th3):.3f}°")
        print()
        if th2 < lowest_angle:
            lowest_angle=th2
        if th1 > 165 and th2 > 165:
            return 180
        if th2 > 140 and th3 > 140:
            return 0
    return th2




  
    

def generate_phase_diagram(root_directory, fixed_gamma13=6, vmax=2.5):
    """
    Main function to process data and plot the phase diagram overlaid on a
    theoretical wetting regime field.
    
    Args:
        root_directory (str): Path to the main data directory.
        fixed_gamma13 (float): The value of gamma13 to use for the theoretical background.
    """
    # --- 1. Data Processing (Same as before) ---
    dir_pattern = re.compile(r'^(\d+)-(\d+)-(\d+)$')
    results = []
    print(f"Scanning root directory: {root_directory}")
    for dir_name in os.listdir(root_directory):
        full_path = os.path.join(root_directory, dir_name)
        print("FULL PATH IS: ", full_path)
        if os.path.isdir(full_path):
            match = dir_pattern.match(dir_name)
            if match:
                gamma12, fixed_gamma13, gamma23 = map(int, match.groups()) # We get g13 but ignore it for the axes
                gamma12 = (2/3) * gamma12 / 100
                fixed_gamma13 = (2/3) * fixed_gamma13 / 100
                gamma23 = (2/3) * gamma23 / 100
                print(f"Processing directory: {dir_name} -> (g12={gamma12}, g23={gamma23})")
                c1_file = full_path + "/c1" + "-" + choose_time + ".dat"
                c2_file = full_path + "/c2" + "-" + choose_time + ".dat"
                print(c1_file, c2_file)
                if c1_file:
                    contact_angle = get_angle(c1_file, c2_file)
                    if not np.isnan(contact_angle):
                        results.append((gamma12, gamma23, contact_angle))
                    else:
                        print(f"  -> No valid data point (c1>0.5) found for (g12={gamma12}, g23={gamma23}).")

    if not results:
        print("\nNo data could be processed. Exiting.")
        return

    data_points = np.array(results)
    g12_vals = data_points[:, 0]
    g23_vals = data_points[:, 1]
    y_color_vals = data_points[:, 2]

    # --- 2. Create the Theoretical Wetting Regime Field ---
    print("\nCreating theoretical wetting background...")
    
    # Define the boundaries of our plot based on the data, with some padding
    g12_min, g12_max = g12_vals.min() - 0.01, g12_vals.max() + 0.01
    g23_min, g23_max = g23_vals.min() - 0.01, g23_vals.max() + 0.01
    
    # Create a grid of points
    g12_grid, g23_grid = np.meshgrid(
        np.linspace(g12_min, g12_max, 500),
        np.linspace(g23_min, g23_max, 500)
    )

    # Define regime conditions based on the triangle inequality
    # We assign an integer to each regime.
    # 0: Partial Wetting (Young's Regime) - The default
    # 1: Droplet Dewets from Solid (g12 > g13 + g23)
    # 2: Droplet Wets Solid (g13 > g12 + g23)
    # 3: Solid Wets Droplet (g23 > g12 + g13)
    regime_grid = np.zeros_like(g12_grid)
    regime_grid[g12_grid > fixed_gamma13 + g23_grid] = 1
    regime_grid[fixed_gamma13 > g12_grid + g23_grid] = 2
    regime_grid[g23_grid > g12_grid + fixed_gamma13] = 3

    # Define discrete colors for the regimes
    # Using light, distinct colors for the background
    regime_colors = ['#d3d3d3', '#ffcccc', '#cce5ff', '#d4edda'] # Grey, Red, Blue, Green
    cmap_regimes = mcolors.ListedColormap(regime_colors)
    
    # --- 3. Plotting ---
    print("Generating combined phase diagram plot...")
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 9))

    # Plot the background field first
    im = ax.imshow(regime_grid, origin='lower', 
                   extent=[g12_min, g12_max, g23_min, g23_max],
                   cmap=cmap_regimes)

    # Plot the simulation data points on top
    scatter = ax.scatter(g12_vals, g23_vals, c=y_color_vals, 
                         cmap='viridis', s=1000, edgecolors='k', zorder=10, vmin=0, vmax=vmax)

    # --- 4. Legends and Labels ---
    
    # Colorbar for the scatter plot data
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Max y-value (for c1 > 0.5)', fontsize=12, weight='bold')

    # Custom legend for the background regimes
    legend_labels = {
        0: 'Partial Wetting / Dewetting',
        1: 'Droplet Dewets from Solid ($\gamma_{12} > \gamma_{13} + \gamma_{23}$)',
        2: 'Droplet Wets Solid ($\gamma_{13} > \gamma_{12} + \gamma_{23}$)',
        3: 'Solid Wets Droplet ($\gamma_{23} > \gamma_{12} + \gamma_{13}$)'
    }
    patches = [mpatches.Patch(color=regime_colors[i], label=label) for i, label in legend_labels.items()]
    # ax.legend(handles=patches, bbox_to_anchor=(1.05, 0.6), loc='upper left',
    #           title='Theoretical Wetting Regimes', fontsize=10)

    # Set axis labels and title
    ax.set_xlabel('$\gamma_{12}$ (Surface Tension 1-2)', fontsize=14, weight='bold')
    ax.set_ylabel('$\gamma_{23}$ (Surface Tension 2-3)', fontsize=14, weight='bold')
    ax.set_title(f'Phase Diagram of Wetting Behavior (for $\gamma_{13} = {fixed_gamma13}$)', fontsize=16, weight='bold')
    
    ax.set_xlim(g12_min, g12_max)
    ax.set_ylim(g23_min, g23_max)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.set_aspect('equal', adjustable='box')

    plt.xticks(np.arange(0.02, 0.16, 0.02)) 
    plt.yticks(np.arange(0.02, 0.16, 0.02)) 

    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout to make space for legend
    #plt.savefig('phase_diagram_with_background.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # --- Option 1: Point the script to your actual data directory ---
    # Replace this with the path to your folder containing '3-6-3', '9-6-3', etc.
    
    

    # The second argument is the fixed gamma13 value used for the background.
    # This should match the value used in your simulations.
    generate_phase_diagram(main_directory, fixed_gamma13=9, vmax=180)
