import numpy as np
import cv2
def composite_centered_numpy(small_image, large_image, x, y):
    # Get dimensions of both images
    large_height, large_width, _ = large_image.shape
    small_height, small_width, _ = small_image.shape

    # Calculate the top-left corner where the large image will be placed
    top_left_x = x - large_width // 2
    top_left_y = y - large_height // 2

    # Determine the region in the small image where the large image will be pasted
    x_start = max(0, top_left_x)
    y_start = max(0, top_left_y)
    x_end = min(small_width, top_left_x + large_width)
    y_end = min(small_height, top_left_y + large_height)

    # Determine the region of the large image that fits within the small image
    large_x_start = max(0, -top_left_x)
    large_y_start = max(0, -top_left_y)
    large_x_end = large_x_start + (x_end - x_start)
    large_y_end = large_y_start + (y_end - y_start)

    # Paste the large image into the small image
    small_image[y_start:y_end, x_start:x_end] = large_image[large_y_start:large_y_end, large_x_start:large_x_end]

    return small_image

# Example usage:
small_image = np.zeros((300, 400, 3), dtype=np.uint8)  # Small image of size 300x400
large_image = np.ones((800, 800, 3), dtype=np.uint8) * 255  # Large white image of size 200x200

# Call the function to composite the large image onto the small one at (150, 200)
output_image = composite_centered_numpy(small_image, large_image, 600, 600)

cv2.imwrite('./image.png',(output_image).astype(np.uint8))