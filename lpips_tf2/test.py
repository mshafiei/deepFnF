import os
import numpy as np
import tensorflow as tf
from PIL import Image

from models.lpips_tensorflow import learned_perceptual_metric_model


def load_image(fn):
    image = Image.open(fn)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = tf.constant(image, dtype=tf.dtypes.float32)
    return image

def convert_tf1_to_tf2(checkpoint_path, output_prefix):
    """Converts a TF1 checkpoint to TF2.

    To load the converted checkpoint, you must build a dictionary that maps
    variable names to variable objects.
    ```
    ckpt = tf.train.Checkpoint(vars={name: variable})  
    ckpt.restore(converted_ckpt_path)

    ```

    Args:
        checkpoint_path: Path to the TF1 checkpoint.
        output_prefix: Path prefix to the converted checkpoint.

    Returns:
        Path to the converted checkpoint.
    """
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
        vars[key] = tf.Variable(reader.get_tensor(key))
    return tf.train.Checkpoint(vars=vars).save(output_prefix), vars

image_size = 64
model_dir = './models'
vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
v3_vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported_v3.h5')
v3_lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported_v3.h5')
mdlvgg = tf.train.load_checkpoint(vgg_ckpt_fn)
mdllin = tf.train.load_checkpoint(lin_ckpt_fn)
# mdlvgg.save_weights(v3_vgg_ckpt_fn)

res, vgg_weights = convert_tf1_to_tf2(vgg_ckpt_fn, 'v3')
res, lin_weights = convert_tf1_to_tf2(lin_ckpt_fn, 'v3')

lpips = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn, vgg_weights, lin_weights)

# official pytorch model metric value
# ex_ref.png <-> ex_p0.png: 0.569
# ex_ref.png <-> ex_p1.png: 0.422
image_fn1 = '/home/mohammad/Downloads/fft_combine/blurred.png'
image_fn2 = '/home/mohammad/Downloads/fft_combine/flash.png'
image_fn3 = '/home/mohammad/Downloads/fft_combine/blurred.png'

# images should be RGB normalized to [0.0, 255.0]
image1 = load_image(image_fn1)[:,:64,:64]
image2 = load_image(image_fn2)[:,:64,:64]
image3 = load_image(image_fn3)[:,:64,:64]

batch_ref = tf.concat([image1, image1], axis=0)
batch_inp = tf.concat([image2, image3], axis=0)
metric = lpips([batch_ref, batch_inp])
print(f'ref shape: {batch_ref.shape}')
print(f'inp shape: {batch_inp.shape}')
print(f'lpips metric shape: {metric.shape}')
print(f'ref <-> p0: {metric[0]:.3f}')
print(f'ref <-> p1: {metric[1]:.3f}')
