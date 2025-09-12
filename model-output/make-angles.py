# This script computes Neumann contact angles for three-phase junctions from
# interfacial tensions (γ12, γ23, γ31), and renders images with three lines
# intersecting at a point so that the angles between them equal the contact angles.
#
# It produces three PNGs (one per tension set) and shows them inline.
#
# Math: Build the "force triangle" with side lengths γ12, γ23, γ31.
# Let α_i be the triangle's internal angle opposite side γ_jk.
# The contact angle inside phase i is θ_i = 180° - α_i.
#
# Conventions here:
#   i=1 ↔ opposite γ23
#   i=2 ↔ opposite γ31
#   i=3 ↔ opposite γ12

import math
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def neumann_contact_angles(g12, g23, g31):
    """Compute (θ1, θ2, θ3) in degrees given (γ12, γ23, γ31)."""
    # Helper: clamp for numerical safety
    def clamp(x, lo=-1.0, hi=1.0): 
        return max(lo, min(hi, x))
    
    # Internal angles (α1 opposite g23, α2 opposite g31, α3 opposite g12)
    # Law of cosines on triangle with sides (g12, g23, g31).
    # α1 opposite side g23:
    cos_a1 = clamp((g12**2 + g31**2 - g23**2) / (2.0 * g12 * g31))
    cos_a2 = clamp((g12**2 + g23**2 - g31**2) / (2.0 * g12 * g23))
    cos_a3 = clamp((g23**2 + g31**2 - g12**2) / (2.0 * g23 * g31))
    a1 = math.degrees(math.acos(cos_a1))
    a2 = math.degrees(math.acos(cos_a2))
    a3 = math.degrees(math.acos(cos_a3))
    # Contact angles:
    t1 = 180.0 - a1
    t2 = 180.0 - a2
    t3 = 180.0 - a3
    # Normalize minor numerical drift so they sum to 360 exactly
    s = t1 + t2 + t3
    t1 *= 360.0/s
    t2 *= 360.0/s
    t3 *= 360.0/s
    return (t1, t2, t3)

def plot_triple_angles(contact_angles_deg, fname=None, title=None):
    """Draw three lines through the origin whose separations equal the given contact angles.
    contact_angles_deg: iterable of three angles (θ1, θ2, θ3) in degrees, summing to 360.
    Order is used around the junction in counterclockwise direction.
    """
    θ1, θ2, θ3 = contact_angles_deg
    # Place line orientations so the sectors are [0, θ1], [θ1, θ1+θ2], [θ1+θ2, 360]
    # Lines at cumulative edges:
    φ0 = 0.0
    φ1 = θ1
    φ2 = θ1 + θ2
    line_angles = [φ0, φ1, φ2]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    L = 1.1  # half-length of line segments

    # Draw three lines as centered segments
    for φ in line_angles:
        rad = math.radians(φ)
        dx, dy = math.cos(rad), math.sin(rad)
        # segment from -L*(dx,dy) to +L*(dx,dy)
        ax.plot([-L*dx, L*dx], [-L*dy, L*dy], linewidth=2)

    # Draw wedges for visualizing angles (small radius)
    r = 0.65
    starts = [0.0, θ1, θ1 + θ2]
    spans  = [θ1,  θ2,  θ3]
    labels = [f"θ₁ = {θ1:.2f}°", f"θ₂ = {θ2:.2f}°", f"θ₃ = {θ3:.2f}°"]

    for start, span, label in zip(starts, spans, labels):
        # Wedge(center, r, theta1, theta2): angles in degrees CCW from +x
        wedge = Wedge((0, 0), r, start, start + span, alpha=0.15, linewidth=1)
        ax.add_patch(wedge)
        # Place label at middle angle
        mid = math.radians(start + span / 2.0)
        tx, ty = 0.45 * math.cos(mid), 0.45 * math.sin(mid)
        ax.text(tx, ty, label, ha="center", va="center", fontsize=9)

    # Origin marker
    ax.plot(0, 0, marker="o", markersize=4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    if fname:
        plt.tight_layout()
        fig.savefig(fname, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# Define the three requested cases (γ12, γ23, γ31)
cases = {
    "(i) γ = (9, 9, 9)":   (9.0, 9.0, 9.0),
    "(ii) γ = (9, 9, 15)": (9.0, 9.0, 15.0),
    "(iii) γ = (15, 9, 9)": (15.0, 9.0, 9.0),
}

outputs = []

for label, (g12, g23, g31) in cases.items():
    θ = neumann_contact_angles(g12, g23, g31)
    # Build filename
    safe = label.replace(" ", "_").replace("γ", "gamma").replace("(", "").replace(")", "").replace(",", "_")
    fname = f"triple_point_angles_{safe}.png"
    outputs.append((label, θ, fname))
    plot_triple_angles(
        θ, 
        fname=fname,
        title=f"{label}\nContact angles (θ₁, θ₂, θ₃) = ({θ[0]:.2f}°, {θ[1]:.2f}°, {θ[2]:.2f}°)"
    )

# Print a concise table of results and saved file paths
for label, θ, fname in outputs:
    print(f"{label}: θ = ({θ[0]:.6f}°, {θ[1]:.6f}°, {θ[2]:.6f}°) -> saved: {fname}")
