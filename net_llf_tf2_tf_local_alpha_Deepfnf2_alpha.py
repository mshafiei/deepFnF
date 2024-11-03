from collections import OrderedDict
from deepfnf import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time

from gllf import gllf, _resize
from easydict import EasyDict as dotmap

class Net(OriginalNet):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        
    def create_basis(self):
        '''Predict image-specific basis'''
        assert (self.ksz == 15 or self.ksz == 7 or self.ksz == 3)
        bottleneck = self.activations['bottleneck']
        out = tf.reduce_mean(bottleneck, axis=[1, 2], keepdims=True)  # 1x1
        out = self.kernel_up_block(
            out, self.channel_count(512,), self.activations['skip_down5'], 'k_up1')  # 2x2
        out = self.kernel_up_block(
            out, self.channel_count(256,), self.activations['skip_down4'], 'k_up2')  # 4x4
        if(self.ksz >= 7):
            out = self.kernel_up_block(
                out, self.channel_count(256,), self.activations['skip_down3'], 'k_up3')  # 8x8
        if(self.ksz == 15):
            out = self.kernel_up_block(
                out, self.channel_count(128), self.activations['skip_down2'], 'k_up4')  # 16x16
        out = self.conv('k_conv', out, self.channel_count(128), ksz=2, stride=1, pad='VALID')
        out = self.conv('k_output_1', out, self.channel_count(128))
        out = self.conv('k_output_2', out, 3 * 2 * self.burst_length * self.num_basis, relu=False)
        out = tf.reshape(
            out, [-1, self.ksz * self.ksz * 3 * 2 * self.burst_length, self.num_basis])
        self.basis = tf.transpose(out, [0, 2, 1])

    def combine(self):
        '''Combine coeffs and basis to get a per-pixel kernel'''
        imsp = self.imsp
        coeffs = tf.reshape(
            self.coeffs, [-1, imsp[1] * imsp[2], self.num_basis])
        self.kernels = tf.matmul(
            coeffs,
            self.basis
        )  # (h * w) x (ksz * ksz * 3 * 2)
        self.kernels = tf.reshape(
            self.kernels, [-1, imsp[1], imsp[2], self.ksz * self.ksz * 3 * self.burst_length, 2])
        self.activations['decoding'] = self.kernels

    def filter_flash_ambient(self, inp):
        self.predict_coeff(inp)
        self.create_basis()
        self.combine()

        filtered_images = tfu.apply_filtering(
            inp[..., :6], self.kernels[..., 0], framewise_op=True)

        # "Bilinearly upsample kernels + filtering"
        # is equivalent to
        # "filter the image with a bilinear kernel + dilated filter the image
        # with the original kernel".
        # This will save more memory.
        smoothed_images = tfu.bilinear_filter(inp[..., :6], ksz=7)
        smoothed_images = tfu.apply_dilated_filtering(
            smoothed_images, self.kernels[..., 1], dilation=4, framewise_op=True)
        filtered_images = filtered_images + smoothed_images
        denoised = (filtered_images[...,:3] + filtered_images[...,3:]) * self.scale

        return denoised

    @tf.function
    def forward(self, inputs):
        outputs = dotmap()
        denoised_flash = self.filter_flash_ambient(inputs.net_ft_input)
        
        outputs.output = denoised_flash
        return outputs