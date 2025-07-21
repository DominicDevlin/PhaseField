import numpy as np
import matplotlib.pyplot as plt

def plot_colored_triangle():
    """
    Plots a triangle with smoothly blended colors from its corners.
    Red (c1), Green (c2), and Blue (c3) are assigned to the corners,
    and the colors are interpolated towards the center.
    """
    
    # 1. Define the triangle corners and their corresponding labels
    # We use an equilateral triangle for a visually pleasing result.
    corners = {
        'c1': np.array([0.5, np.sqrt(3)/2]), # Top corner
        'c2': np.array([0.0, 0.0]),          # Bottom-left corner
        'c3': np.array([1.0, 0.0])           # Bottom-right corner
    }
    
    # Assign RGB colors to each corner label
    # c1 -> Red, c2 -> Green, c3 -> Blue
    corner_colors = {
        'c1': np.array([1, 0, 0]),  # Red
        'c2': np.array([0, 1, 0]),  # Green
        'c3': np.array([0, 0, 1])   # Blue
    }
    
    # 2. Set up the grid for the plot
    # The resolution determines the smoothness of the gradient.
    resolution = 500
    # Create a grid of x, y coordinates that spans the triangle's bounds.
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, np.sqrt(3)/2
    
    # np.meshgrid creates coordinate matrices from coordinate vectors.
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Flatten the grid to a list of points (x, y)
    points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # 3. Calculate Barycentric Coordinates
    # For any point P, P = w1*c1 + w2*c2 + w3*c3, where w1+w2+w3=1.
    # The weights (w1, w2, w3) are the barycentric coordinates.
    
    # We use a vectorized formula for efficiency.
    # Let c1=(x1,y1), c2=(x2,y2), c3=(x3,y3)
    x1, y1 = corners['c1']
    x2, y2 = corners['c2']
    x3, y3 = corners['c3']
    
    # This denominator is related to twice the area of the triangle.
    den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    
    # Calculate the weights for each point in the grid
    x, y = points[:, 0], points[:, 1]
    w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / den
    w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / den
    w3 = 1 - w1 - w2

    # 4. Determine colors and create the image
    # The RGB color at each point is a blend of the corner colors,
    # weighted by the barycentric coordinates.
    # Since our corner colors are (1,0,0), (0,1,0), and (0,0,1), the
    # blended color is simply (w1, w2, w3).
    colors = np.vstack([w1, w2, w3]).T

    # 5. Create a mask to hide colors outside the triangle
    # A point is inside the triangle if all its barycentric weights are >= 0.
    # (The w1+w2+w3=1 condition ensures they are also all <= 1).
    mask = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
    
    # Create an RGBA image (Red, Green, Blue, Alpha)
    # Initialize with a transparent background (alpha=0)
    rgba_image = np.zeros((len(points), 4))
    
    # Set the RGB values from our calculated colors
    rgba_image[:, :3] = colors
    # Set the alpha channel to 1 (opaque) only for points inside the triangle
    rgba_image[mask, 3] = 1.0
    
    # Reshape the flat list of RGBA values back into a 2D image
    image = rgba_image.reshape(resolution, resolution, 4)

    # 6. Plot the final image and legend
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Use imshow to display the generated image.
    # The 'extent' parameter maps the image pixels to the data coordinates.
    # 'origin="lower"' places the (0,0) index of the array at the bottom-left.
    ax.imshow(image, origin='lower', extent=[x_min, x_max, y_min, y_max])
    
    # Create a legend
    # We plot invisible points at the corner locations just to create legend handles.
    for name, pos in corners.items():
        ax.scatter(pos[0], pos[1], c=corner_colors[name], s=0, label=name)

    ax.legend(title="Corners", loc="best", fontsize=12)
    
    # Clean up the plot
    ax.set_title("Barycentric Color Gradient in a Triangle", fontsize=16)
    ax.set_aspect('equal', adjustable='box') # Ensure the triangle is not distorted
    ax.axis('off') # Hide the x and y axes for a cleaner look
    
    plt.tight_layout()
    plt.show()

# Run the function
if __name__ == "__main__":
    plot_colored_triangle()