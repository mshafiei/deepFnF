from collections import OrderedDict
from guided_local_laplacian_color import guided_local_laplacian_color
from net import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time

class Net(OriginalNet):
    def __init__(self, alpha, beta, levels, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1, lmbda=1,IMSZ=448):
        self.weights = {}
        self.activations = OrderedDict()
        self.num_basis = num_basis
        self.ksz = ksz
        self.burst_length = burst_length
        self.channels_count_factor = channels_count_factor
        self.channel_count = lambda x: max(1, int(x * self.channels_count_factor))
        self.lmbda = lmbda
        self.IMSZ = IMSZ
        self.alpha = alpha
        self.beta = beta
        self.levels = levels

    def deepfnfDenoising(self, inp, alpha):
        denoised = super().forward(inp)
        flash = inp[:, :, :, 3:6] * alpha / ut.FLASH_STRENGTH
        return denoised, flash
    
    def forward(self, inp, alpha):
        denoised, _ = self.deepfnfDenoising(inp, alpha)
        return denoised

    
    def llf(self, denoised, flash):
        flash_np = np.ascontiguousarray(np.array(flash[0,...]).squeeze().transpose(2,0,1))
        denoised_np = np.ascontiguousarray(np.array(denoised[0,...]).squeeze().transpose(2,0,1))
        c, h, w, = flash_np.shape
        combined = np.empty([3, h, w], dtype=denoised_np.dtype)
        intensity_max = max(flash_np.max(), denoised_np.max())
        intensity_min = min(flash_np.min(), denoised_np.min())
        denoised_np = (denoised_np - intensity_min) / (intensity_max - intensity_min)
        flash_np = (flash_np - intensity_min) / (intensity_max - intensity_min)
        # guided_local_laplacian(flash_np, denoised_np, self.levels, self.alpha / (self.levels - 1), self.beta, combined)
        start = time.time_ns()
        guided_local_laplacian_color(flash_np, denoised_np, self.levels, self.alpha / (self.levels - 1), self.beta, combined)
        tf.print('llf_time ', (time.time_ns() - start)/1000000)
        combined = tf.convert_to_tensor(combined.transpose(1,2,0))[None,...]
        combined = combined * (intensity_max - intensity_min) + intensity_min

        return combined