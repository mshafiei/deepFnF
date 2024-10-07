import numpy as np
import cv2

def create_random_image(width, height):
    """Generate a random image of given width and height."""
    random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return random_data

def tile_images(images, output_width):
    # Get the widths and heights of the images
    widths = [img.shape[1] for img in images]
    heights = [img.shape[0] for img in images]

    # Initialize variables for the new image
    max_height = max(heights)  # Maximum height of images in a row
    total_height = 0           # Total height of the resulting image
    new_image = np.zeros((max_height, output_width, 3), dtype=np.uint8)

    x_offset = 0  # Current x position
    y_offset = 0  # Current y position

    for img in images:
        img_height, img_width = img.shape[:2]

        # Check if the image fits in the current row
        if x_offset + img_width > output_width:
            # Move to the next row
            y_offset += max_height  # Move down by the max height of the previous row
            x_offset = 0            # Reset x offset
            max_height = img_height  # Update max height for the new row
            # Expand new_image to add space for the new row
            new_image = np.vstack([new_image, np.zeros((max_height, output_width, 3), dtype=np.uint8)])

        # Paste the image into the new canvas
        new_image[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = img
        
        # Update offsets
        x_offset += img_width  # Move the x offset to the right

    return new_image

# Example usage:
# Generate a list of random images with varying sizes
random_images = [create_random_image(np.random.randint(100, 301), np.random.randint(100, 301)) for _ in range(6)]

output_width = 2000  # Set the desired width of the final image

# Tile the images and get the resulting image
resulting_image = tile_images(random_images, output_width)

# Save the resulting image using OpenCV
cv2.imwrite('tiled_image.jpg', resulting_image)