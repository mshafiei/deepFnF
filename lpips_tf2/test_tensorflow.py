import os
import numpy as np
import tensorflow as tf
from PIL import Image

from models_tensorflow.lpips_tensorflow import learned_perceptual_metric_model


def load_image(fn):
    image = Image.open(fn)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)

    image = tf.constant(image, dtype=tf.dtypes.float32)
    return image


image_size = 64
model_dir = './models'
vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported.data-00000-of-00002.h5')
lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported.data-00000-of-00002.h5')
lpips = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)

# official pytorch model value:
# Distance: ex_ref.png <-> ex_p0.png = 0.569
# Distance: ex_ref.png <-> ex_p1.png = 0.422
image_fn1 = './imgs/ex_ref.png'
image_fn2 = './imgs/ex_p0.png'
# image_fn2 = './imgs/ex_p1.png'

image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Distance: {:.3f}'.format(dist01))