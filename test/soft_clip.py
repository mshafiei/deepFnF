import tensorflow as tf

def smooth_strict_clip(x, clip_min, clip_max, smoothing=1.0):
    """
    Smoothly clips values to the range [clip_min, clip_max] in a differentiable way
    with strict enforcement of boundaries.

    Parameters:
    - x: The input tensor.
    - clip_min: The lower boundary for clipping.
    - clip_max: The upper boundary for clipping.
    - smoothing: Controls the smoothness of the transition (higher = smoother).

    Returns:
    - A tensor with values strictly clipped between clip_min and clip_max.
    """
    # Scale x to the range [-1, 1] before applying tanh
    scale = 2.0 / (clip_max - clip_min)
    offset = (clip_max + clip_min) / 2.0
    x_scaled = (x - offset) * scale

    # Apply tanh for smooth, strict clipping and scale back to [clip_min, clip_max]
    clipped = tf.tanh(x_scaled / smoothing)
    output = (clipped / scale) + offset
    
    return output

# Example usage
x = tf.constant([-10.0, -2.0, 0.0, 2.0, 10.0])
clip_min = -1.0
clip_max = 1.0

smooth_clipped_x = smooth_strict_clip(x, clip_min, clip_max)
print(smooth_clipped_x)