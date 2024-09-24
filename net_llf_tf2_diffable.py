from collections import OrderedDict
from guided_local_laplacian import guided_local_laplacian
from guided_local_laplacian_color import guided_local_laplacian_color
from guided_local_laplacian_color_grad import guided_local_laplacian_color_grad
from net import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time

@tf.numpy_function(Tout=tf.float32)
def call_llf(flash, denoised, alpha, levels, beta):
    flash_np = flash
    denoised_np = denoised
    c, h, w, = flash_np.shape
    combined = np.empty([3, h, w], dtype=denoised_np.dtype)
    intensity_max = max(flash_np.max(), denoised_np.max())
    intensity_min = min(flash_np.min(), denoised_np.min())
    # denoised_np = (denoised_np - intensity_min) / (intensity_max - intensity_min)
    # flash_np = (flash_np - intensity_min) / (intensity_max - intensity_min)
    start = time.time_ns()
    guided_local_laplacian_color(flash_np, denoised_np, levels, alpha, beta, combined)
    # tf.print('llf_time ', (time.time_ns() - start)/1000000)
    combined = combined.transpose(1,2,0)[None,...]
    # combined = combined * (intensity_max - intensity_min) + intensity_min
    return combined

@tf.numpy_function(Tout=tf.float32)
def call_dllf(flash, denoised, alpha, levels, beta):
    flash_np = flash
    denoised_np = denoised
    c, h, w, = flash_np.shape
    deriv = np.empty([3, h, w], dtype=denoised_np.dtype)
    intensity_max = max(flash_np.max(), denoised_np.max())
    intensity_min = min(flash_np.min(), denoised_np.min())
    # denoised_np = (denoised_np - intensity_min) / (intensity_max - intensity_min)
    # flash_np = (flash_np - intensity_min) / (intensity_max - intensity_min)
    start = time.time_ns()
    guided_local_laplacian_color_grad(flash_np, denoised_np, levels, alpha, beta, deriv)
    # tf.print('dllf_time ', (time.time_ns() - start)/1000000)
    deriv = deriv.transpose(1,2,0)[None,...]
    # deriv = deriv * (intensity_max - intensity_min) + intensity_min
    return deriv

class Net(OriginalNet):
    def __init__(self, llf_beta, llf_levels, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1, lmbda=1,IMSZ=448):
        self.weights = {}
        self.activations = OrderedDict()
        self.num_basis = num_basis
        self.ksz = ksz
        self.burst_length = burst_length
        self.channels_count_factor = channels_count_factor
        self.channel_count = lambda x: max(1, int(x * self.channels_count_factor))
        self.lmbda = lmbda
        self.IMSZ = IMSZ
        self.beta = llf_beta
        self.levels = llf_levels

    def deepfnfDenoising(self, inp, alpha):
        denoised = super().forward(inp)
        flash = inp[:, :, :, 3:6] * alpha / ut.FLASH_STRENGTH
        return denoised, flash
    
    @tf.custom_gradient
    def llf(self, flash, denoised, alpha):
        
        combined = call_llf(tf.transpose(flash[0,...],(2,0,1)),
            tf.transpose(denoised[0,...],(2,0,1)),
            alpha, self.levels, self.beta)

        def grad_fn(upstream):
            # assert len(upstream) == 3
            deriv = call_dllf(tf.transpose(flash[0,...],(2,0,1)),
            tf.transpose(denoised[0,...],(2,0,1)),
            alpha, self.levels, self.beta)
            
            return (upstream, upstream, tf.reduce_sum(upstream * deriv))
        
        return combined, grad_fn