import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

def plot_ternary_legend():
    # --- 1. Define Colors (Okabe-Ito) ---
    # These must match your main script exactly
    c1_color = np.array(to_rgb('#E69F00')) # Orange (Top)
    c2_color = np.array(to_rgb('#009E73')) # Bluish Green (Right)
    c3_color = np.array(to_rgb('#56B4E9')) # Sky Blue (Left)

    # --- 2. Geometry Setup (Equilateral Triangle) ---
    # Vertices: (x, y)
    # Top (c1=1)
    v1 = np.array([0.0, np.sqrt(3)/2]) 
    # Right (c2=1)
    v2 = np.array([0.5, 0.0])          
    # Left (c3=1)
    v3 = np.array([-0.5, 0.0])         

    # --- 3. Create Grid ---
    resolution = 500
    x = np.linspace(-0.6, 0.6, resolution)
    y = np.linspace(-0.1, 1.0, resolution)
    gx, gy = np.meshgrid(x, y)

    # --- 4. Convert Cartesian (x,y) to Barycentric (c1,c2,c3) ---
    # This solves the linear system for every pixel to find weights
    # formula derived from: P = c1*v1 + c2*v2 + c3*v3, with c1+c2+c3=1
    
    # c1 is simply the percentage of height
    c1 = gy / (np.sqrt(3)/2)
    
    # c2 is derived based on x position relative to slope
    c2 = (gx + 0.5 * (1 - c1)) 
    
    # c3 is the remainder
    c3 = 1.0 - c1 - c2

    # --- 5. Create Mask for Triangle Shape ---
    # A point is inside if all barycentric coordinates are between 0 and 1
    mask = (c1 >= 0) & (c1 <= 1) & \
           (c2 >= 0) & (c2 <= 1) & \
           (c3 >= 0) & (c3 <= 1)

    # --- 6. Blend Colors ---
    # Initialize image with white (1,1,1) + Alpha channel
    img = np.ones((resolution, resolution, 4)) 
    
    # Stack weights for broadcasting: shape (N, M, 3)
    weights = np.dstack((c1, c2, c3))
    
    # Calculate weighted color sum
    # shape (3, 3) -> c1*Orange + c2*Green + c3*Blue
    # We transpose colors to shape (3, 3) to matrix multiply or just sum manually
    mixed_rgb = (c1[..., None] * c1_color + 
                 c2[..., None] * c2_color + 
                 c3[..., None] * c3_color)
    
    # Clip to avoid numeric errors
    mixed_rgb = np.clip(mixed_rgb, 0, 1)

    # Apply to image where mask is True
    img[..., :3] = np.where(mask[..., None], mixed_rgb, 1.0) # White background
    img[..., 3]  = np.where(mask, 1.0, 0.0) # Transparent background outside triangle

    # --- 7. Plotting ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img, extent=[-0.6, 0.6, -0.1, 1.0], origin='lower', interpolation='bilinear')

    # Add Labels
    offset = 0.05
    ax.text(v1[0], v1[1] + offset, 'Phase 1\n(Orange)', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(v2[0] + offset/2, v2[1], 'Phase 2\n(Bluish Green)', ha='left', va='center', fontsize=12, fontweight='bold')
    ax.text(v3[0] - offset/2, v3[1], 'Phase 3\n(Sky Blue)', ha='right', va='center', fontsize=12, fontweight='bold')

    # Clean up axes
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off') # Remove box and ticks

    plt.show()

if __name__ == "__main__":
    plot_ternary_legend()