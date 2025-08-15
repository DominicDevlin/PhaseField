#!/usr/bin/env python3
"""
Compute and visualize interior contact angles at three-phase triple junctions.

Inputs
------
c1.dat : columns [x, y, c1]
c2.dat : columns [x, y, c2]
(c3 := 1 - c1 - c2)

Outputs
-------
- Console printout of triple point(s) and interior angles (θ1, θ2, θ3) that sum ~360°.
- Optional figure with:
    * light background showing which phase dominates (argmax(c1,c2,c3)),
    * interfaces (phi12=0, phi23=0, phi31=0),
    * triple point markers,
    * angle labels placed inside each phase wedge.

Usage
-----
python triple_contact_angles.py --c1 c1.dat --c2 c2.dat --plot triples.png --nx 600
"""

import argparse
import math
import sys
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", default="data/equilibrium/21-9-15/c1-500.dat", help="path to c1 data file")
    ap.add_argument("--c2", default="data/equilibrium/21-9-15/c2-500.dat", help="path to c2 data file")
    ap.add_argument("--nx", type=int, default=500, help="grid resolution in x (y auto)")
    ap.add_argument("--plot", default=None, help="optional output figure path (PNG/SVG)")
    ap.add_argument("--phi_tol", type=float, default=5e-3, help="|phi31| tolerance for triple-point filtering")
    ap.add_argument("--report-small-angles", action="store_true",
                    help="also print the three small crossing angles that sum to 180°")
    args = ap.parse_args()

    # Load scattered fields
    x1, y1, c1v = load_field(args.c1)
    x2, y2, c2v = load_field(args.c2)

    f1 = build_linear_interpolator(x1, y1, c1v)
    f2 = build_linear_interpolator(x2, y2, c2v)

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

    # Extract zero-contours
    lines12 = contour_paths_from_grid(XX, YY, PHI12.filled(np.nan), level=0.0)
    lines23 = contour_paths_from_grid(XX, YY, PHI23.filled(np.nan), level=0.0)
    lines31 = contour_paths_from_grid(XX, YY, PHI31.filled(np.nan), level=0.0)

    if len(lines12) == 0 or len(lines23) == 0:
        print("No interfaces found (phi12 or phi23 has no zero-level contours in the overlap).")
        sys.exit(0)

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
        if args.report_small_angles:
            print(f"    α (small crossing angles): {a1:.3f}°, {a2:.3f}°, {a3:.3f}° (sum ~180)")
        print()
        
    # ---- Angles plot (grouped bars per triple point) ----
    if results:
        # Collect angles and labels
        th = np.array([res["angles_deg"] for res in results])   # shape: (n, 3)
        tp_labels = [
            f"#{i}\n({res['triple_point'][0]:.3g}, {res['triple_point'][1]:.3g})"
            for i, res in enumerate(results, 1)
        ]
        n = th.shape[0]

        # Figure size scales with number of triple points
        fig_w = max(6.5, 1.4 * n + 3.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=140)

        x = np.arange(n)
        width = 0.25

        r1 = ax.bar(x - width, th[:, 0], width=width, label=r"$\theta_1$")
        r2 = ax.bar(x,         th[:, 1], width=width, label=r"$\theta_2$")
        r3 = ax.bar(x + width, th[:, 2], width=width, label=r"$\theta_3$")

        ax.set_xticks(x)
        ax.set_xticklabels(tp_labels)
        ax.set_ylabel("Interior angle (degrees)")
        ax.set_title("Triple-junction interior contact angles")
        ax.set_ylim(0, 360)
        ax.legend(ncol=3)

        # Reference line at 360° (θ1+θ2+θ3 should be ~360)
        ax.axhline(360, linestyle="--", linewidth=1.0)

        # Annotate sums and per-bar values for quick sanity checks
        sums = th.sum(axis=1)
        for xi, s in zip(x, sums):
            ax.text(xi, 355, f"Σ={s:.1f}°", ha="center", va="bottom", fontsize=9)

        def _annotate_bars(rects):
            for r in rects:
                h = r.get_height()
                ax.text(r.get_x() + r.get_width()/2.0, h + 2, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=8)
        _annotate_bars(r1); _annotate_bars(r2); _annotate_bars(r3)

        fig.tight_layout()
        out_path = args.plot if args.plot else "triple_angles.png"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved angle plot to {out_path}")

if __name__ == "__main__":
    main()