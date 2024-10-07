import tensorflow as tf

def split_image_into_blocks(image, num_blocks_h, num_blocks_w, block_size_h, block_size_w):
    """
    Splits an image into blocks with symmetrical zero padding.
    
    Parameters:
    - image: 4D Tensor of shape [1, height, width, channels] (e.g., a single image tensor).
    - num_blocks_h: The number of blocks along the height (vertical axis).
    - num_blocks_w: The number of blocks along the width (horizontal axis).
    - block_size_h: The height of each block.
    - block_size_w: The width of each block.
    
    Returns:
    - blocks_reshaped: Tensor of shape [num_blocks_h, num_blocks_w, block_size_h, block_size_w, channels].
    """
    # Get the height and width of the image
    image_height = image.shape[1]
    image_width = image.shape[2]

    # Calculate the required padding for height and width to fit into the specified blocks
    total_pad_h = max(0, num_blocks_h * block_size_h - image_height)
    total_pad_w = max(0, num_blocks_w * block_size_w - image_width)

    # Calculate top, bottom, left, and right padding
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    # Pad the image with zeros (top, bottom, left, right)
    padded_image = tf.pad(image, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    # Extract blocks from the padded image
    blocks = tf.image.extract_patches(
        images=padded_image,
        sizes=[1, block_size_h, block_size_w, 1],
        strides=[1, block_size_h, block_size_w, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    # Reshape the patches to [num_blocks_h, num_blocks_w, block_size_h, block_size_w, channels]
    blocks_reshaped = tf.reshape(blocks, [num_blocks_h, num_blocks_w, block_size_h, block_size_w, image.shape[-1]])

    return blocks_reshaped

# Example usage:
image = tf.random.uniform(shape=[1, 60, 60, 3], minval=0, maxval=255, dtype=tf.float32)
num_blocks_h = 2  # Number of blocks in the vertical axis
num_blocks_w = 2  # Number of blocks in the horizontal axis
block_size_h = 32 # Height of each block
block_size_w = 32 # Width of each block

blocks = split_image_into_blocks(image, num_blocks_h, num_blocks_w, block_size_h, block_size_w)
print("Blocks shape:", blocks.shape)  # Should output [num_blocks_h, num_blocks_w, block_size_h, block_size_w, channels]

# Checking the first block's first channel
print("Block[0,0] for channel 0:\n", blocks[0, 0, :, :, 0])