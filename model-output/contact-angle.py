#!/usr/bin/env python3
"""
Estimate triple-junction contact angles for a three-phase field from scattered data.

Inputs:
  - c1.dat : columns [x, y, c1]
  - c2.dat : columns [x, y, c2]
  (c3 is taken as 1 - c1 - c2)

Outputs:
  - Prints triple point coordinates and contact angles (deg) for phases 1, 2, 3.
  - Optional PNG figure showing interfaces and triple points.

Usage:
  python triple_contact_angle.py --c1 c1.dat --c2 c2.dat --plot out.png
"""

import argparse
import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # for headless contour extraction and optional plot
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from typing import List, Tuple

# ----------------------------
# IO + interpolation utilities
# ----------------------------

def load_field(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path}: expected 3+ columns [x,y,c], got shape {arr.shape}")
    return arr[:, 0], arr[:, 1], arr[:, 2]

def build_linear_interpolator(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Returns a function f(X, Y) that evaluates z at arbitrary points
    using piecewise-linear interpolation on a Delaunay triangulation.
    Output is a masked array; masked=True outside convex hull.
    """
    tri = mtri.Triangulation(x, y)
    interp = mtri.LinearTriInterpolator(tri, z)
    return interp

def common_bbox(x1, y1, x2, y2, margin=0.0):
    bbox1 = (x1.min(), x1.max(), y1.min(), y1.max())
    bbox2 = (x2.min(), x2.max(), y2.min(), y2.max())
    xmin = max(bbox1[0], bbox2[0])
    xmax = min(bbox1[1], bbox2[1])
    ymin = max(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    if not (xmax > xmin and ymax > ymin):
        raise ValueError("Input point sets have disjoint bounding boxes; no overlap.")
    dx = xmax - xmin
    dy = ymax - ymin
    return xmin + margin*dx, xmax - margin*dx, ymin + margin*dy, ymax - margin*dy

# ----------------------------
# Contours + intersections
# ----------------------------

def contour_paths_from_grid(X, Y, Z, level=0.0) -> List[np.ndarray]:
    """Return list of Nx2 polylines (level-set contours) for Z at 'level'."""
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
    """
    Segment intersection for p->r and q->s (each 2D).
    Returns (hit:bool, point:np.ndarray).
    Robust to near-parallel with a small eps.
    """
    p = np.asarray(p); r = np.asarray(r); q = np.asarray(q); s = np.asarray(s)
    u = r - p
    v = s - q
    w = p - q
    denom = u[0]*v[1] - u[1]*v[0]
    eps = 1e-12 * (np.linalg.norm(u) + np.linalg.norm(v) + 1.0)
    if abs(denom) < eps:
        return False, None  # parallel or nearly so
    t = (v[0]*w[1] - v[1]*w[0]) / denom
    uparam = (u[0]*w[1] - u[1]*w[0]) / denom
    if -1e-10 <= t <= 1+1e-10 and -1e-10 <= uparam <= 1+1e-10:
        pt = p + t*u
        return True, pt
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
    """
    Greedy clustering with radius eps; returns list of cluster centers (means).
    """
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
    """
    Central difference gradient of scalar interpolant f at (x,y)
    using step h (adapted to domain scale).
    Returns 2-vector; masked points raise ValueError.
    """
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
    if n == 0:
        return v
    return v / n

def angle_between(u, v):
    cu = unit(u); cv = unit(v)
    dot = np.clip(np.dot(cu, cv), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))

# ----------------------------
# Main pipeline
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", default="data/equilibrium/9-9-9/c1-300.dat", help="path to c1 data file")
    ap.add_argument("--c2", default="data/equilibrium/9-9-9/c2-300.dat", help="path to c2 data file")
    ap.add_argument("--nx", type=int, default=500, help="grid resolution in x (y auto)")
    ap.add_argument("--eps", type=float, default=None, help="clustering radius; default = 2 grid steps")
    ap.add_argument("--plot", default=None, help="optional path to save plot (PNG/SVG)")
    ap.add_argument("--phi_tol", type=float, default=1e-1, help="|phi31| tolerance to accept a triple point")
    args = ap.parse_args()

    # Load
    x1, y1, c1v = load_field(args.c1)
    x2, y2, c2v = load_field(args.c2)

    # Interpolators
    f1 = build_linear_interpolator(x1, y1, c1v)
    f2 = build_linear_interpolator(x2, y2, c2v)

    # Grid over common bbox (inside both)
    xmin, xmax, ymin, ymax = common_bbox(x1, y1, x2, y2, margin=0.02)
    nx = max(64, args.nx)
    ny = max(64, int(round(nx * (ymax - ymin) / (xmax - xmin))))
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

    # Extract zero-contours as polylines
    lines12 = contour_paths_from_grid(XX, YY, PHI12.filled(np.nan), level=0.0)
    lines23 = contour_paths_from_grid(XX, YY, PHI23.filled(np.nan), level=0.0)
    lines31 = contour_paths_from_grid(XX, YY, PHI31.filled(np.nan), level=0.0)

    if len(lines12) == 0 or len(lines23) == 0:
        print("No interfaces found (phi12 or phi23 has no zero-level contours in the overlap).")
        sys.exit(0)

    # Intersections of phi12=0 and phi23=0 are triple-point candidates
    raw_pts = pairwise_intersections(lines12, lines23)
    if not raw_pts:
        print("No triple points detected (no intersections of phi12 and phi23).")
        sys.exit(0)
    print(raw_pts)

    # Filter by requiring phi31≈0 at the intersection
    def phi31_val(x, y) -> float:
        v = PHI31[np.argmin(np.abs(Y - y)), np.argmin(np.abs(X - x))]  # coarse lookup
        if np.ma.is_masked(v):
            # fallback to direct eval via interpolants
            vv = (1.0 - f1(x, y) - f2(x, y)) - f1(x, y)
            if np.ma.is_masked(vv):
                return np.inf
            return float(vv)
        return float(v)

    candidates = [p for p in raw_pts]# if abs(phi31_val(p[0], p[1])) <= args.phi_tol]
    if not candidates:
        print("Intersections found, but none satisfy |phi31|<=tol near zero; try increasing --phi_tol.")
        sys.exit(0)

    # Cluster intersections (there are often two triple points)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    grid_step = 0.5 * (dx + dy)
    eps = args.eps if args.eps is not None else 2.0 * grid_step
    centers = cluster_points(candidates, eps=eps)

    # Compute contact angles at each triple point
    h = 2.0 * grid_step  # finite-difference step for gradients
    results = []

    for (xt, yt) in centers:
        # gradients of c1,c2,c3
        g1 = grad_from_interpolant(f1, xt, yt, h)
        g2 = grad_from_interpolant(f2, xt, yt, h)
        # c3 = 1 - c1 - c2 -> grad c3 = -(g1 + g2)
        g3 = -(g1 + g2)

        # phase-1 angle between (1-2) and (1-3): ∇(c1-c2) vs ∇(c1-c3)
        th1 = angle_between(g1 - g2, g1 - g3)
        # phase-2 angle between (2-3) and (2-1): ∇(c2-c3) vs ∇(c2-c1)
        th2 = angle_between(g2 - g3, g2 - g1)
        # phase-3 angle between (3-1) and (3-2): ∇(c3-c1) vs ∇(c3-c2)
        th3 = angle_between(g3 - g1, g3 - g2)

        s = th1 + th2 + th3
        results.append({
            "triple_point": (xt, yt),
            "angles_deg": (th1, th2, th3),
            "sum_check_deg": s
        })

    # Print results
    print("\nDetected triple junctions (up to small numerical tolerance):\n")
    for i, res in enumerate(results, 1):
        (xt, yt) = res["triple_point"]
        th1, th2, th3 = res["angles_deg"]
        print(f"#{i}: at (x={xt:.6g}, y={yt:.6g})")
        print(f"    θ1 (inside phase-1, between interfaces 1–2 & 1–3): {th1:8.3f}°")
        print(f"    θ2 (inside phase-2, between interfaces 2–3 & 2–1): {th2:8.3f}°")
        print(f"    θ3 (inside phase-3, between interfaces 3–1 & 3–2): {th3:8.3f}°")
        print(f"    angle sum (sanity ~360°): {res['sum_check_deg']:.3f}°\n")

    # Optional plot
    if args.plot:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        # plot zero contours
        for lines, lab in [(lines12, r"$\phi_{12}=0$"),
                           (lines23, r"$\phi_{23}=0$"),
                           (lines31, r"$\phi_{31}=0$")]:
            for k, ln in enumerate(lines):
                ax.plot(ln[:,0], ln[:,1], lw=1.25, label=lab if k==0 else None)
        # triple points
        for i, res in enumerate(results, 1):
            xt, yt = res["triple_point"]
            ax.plot([xt],[yt], marker="o", ms=5)
            th1, th2, th3 = res["angles_deg"]
            ax.text(xt, yt, f"  T{i}", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=8)
        ax.set_title("Three-phase interfaces and triple point(s)")
        fig.tight_layout()
        fig.savefig(args.plot, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to: {args.plot}")

if __name__ == "__main__":
    main()
