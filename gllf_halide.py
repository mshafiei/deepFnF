import tensorflow as tf
from guided_local_laplacian_color_local_alpha_Mullapudi2016 import guided_local_laplacian_color_local_alpha_Mullapudi2016 as guided_local_laplacian_color
from guided_local_laplacian_color_neural_local_alpha_Mullapudi2016 import guided_local_laplacian_color_neural_local_alpha_Mullapudi2016 as guided_local_laplacian_color
from guided_local_laplacian_color_1d_local_alpha_Mullapudi2016 import guided_local_laplacian_color_1d_local_alpha_Mullapudi2016 as guided_local_laplacian_color_1d
import numpy as np
import timeit
def halide_gllf(denoised_np, flash_np, levels, alpha, beta,sigma):
    h, w, c = flash_np.shape
    flash_np = tf.transpose(flash_np, [2,0,1])
    denoised_np = tf.transpose(denoised_np, [2,0,1])
    aw = 2
    ah = 2
    alpha = np.ones([ah, aw], dtype=np.float32) * np.array(alpha).astype(np.float32)
    llf_out = np.empty([3, h, w], dtype=np.float32)
    guided_local_laplacian_color(flash_np, denoised_np, levels, alpha, beta, sigma, aw, ah, w, h, llf_out)
    return tf.transpose(llf_out, [1,2,0])[None,...]

def halide_neural_gllf(denoised_np, flash_np, image_weights, range_weights, w_i, sigma_i, levels, alpha, beta,sigma):
    h, w, c = flash_np.shape
    flash_np = tf.transpose(flash_np, [2,0,1])
    denoised_np = tf.transpose(denoised_np, [2,0,1])
    inpt = tf.stack((denoised_np,flash_np),axis=0)
    aw = 2
    ah = 2
    # alpha = np.ones([ah, aw, 2], dtype=np.float32) * np.array(alpha).astype(np.float32)
    # alpha[0,:,:] *= 0
    llf_out = np.empty([3, h, w], dtype=np.float32)
    # beta = np.array([1,0],dtype=np.float32)
    # sigma = np.array([1,0],dtype=np.float32)
    img_ct = len(inpt)
    fn = lambda: guided_local_laplacian_color(inpt, levels, image_weights.numpy().transpose(5,4,3,2,1,0), range_weights.numpy().transpose(2,1,0), w_i.numpy().transpose(1,0), sigma_i.numpy().transpose(1,0), img_ct, aw, ah, w, h, llf_out)
    t = timeit.Timer(fn, setup=fn)
    avg_time_sec = t.timeit(number=3) / 3
    print('guided_local_laplacian_color takes ', avg_time_sec)
    return tf.transpose(llf_out, [1,2,0])[None,...]

#     *********
# (3, 448, 448) <dtype: 'float32'> (3, 448, 448) <dtype: 'float32'> (3, 448, 448) float32
# 4 [[1. 1.]
#  [1. 1.]] 1.0 1.0 2 2 448 448
# *********

# (3, 408, 408) <dtype: 'float32'> (3, 408, 408) <dtype: 'float32'> (3, 408, 408) float32
# 4 tf.Tensor(
# [[-1. -1.]
#  [-1. -1.]], shape=(2, 2), dtype=float32) 1.0 1.0 2 2 408 408