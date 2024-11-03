from collections import OrderedDict
from deepfnf import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time

from gllf import gllf, gllf_diffable, _resize
from easydict import EasyDict as edict

from net_llf_tf2_tf_local_alpha_Deepfnf2_alpha import Net as deepfnf2

class Net(deepfnf2):
    def __init__(self, IMSZ=448,llf_beta=1,llf_levels=3, **kargs):
        self.llf_beta=llf_beta
        self.llf_levels=llf_levels
        self.IMSZ = IMSZ
        super().__init__(**kargs)
        self.llf = lambda input, guide, input_pyramid, guide_pyramid, alpha_h, alpha_i: gllf(input, guide, input_pyramid, guide_pyramid, self.llf_levels, self.llf_levels, alpha_h, alpha_i, IMSZ=self.IMSZ, beta=self.llf_beta, sigma=1)
        # self.llf = lambda input, guide, alpha_h, alpha_i: gllf_diffable(input, guide, self.llf_levels, self.llf_levels, alpha_h, alpha_i, IMSZ=self.IMSZ, beta=self.llf_beta, sigma=1)
        
    def predict_coeff(self, inp):
        '''Predict per-pixel coefficient vector given the input'''
        self.imsp = tf.shape(inp)

        out, skips = self.encode(inp)
        out = self.decode(out, skips)
        out = self.conv('output', out, self.num_basis + 6, relu=False)
        self.coeffs_pre_soft = out
        self.coeffs = out[..., :self.num_basis]
        self.llf_alpha_i = out[..., -6:-3]
        self.llf_alpha_h = out[..., -3:]
        self.activations['output'] = self.coeffs
    
    def filter_flash_ambient(self, inp):
        self.predict_coeff(inp)
        self.create_basis()
        self.combine()

        # input_ambient = inp[..., :3]
        # input_flash = inp[..., 3:6]
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
        filtered_ambient = filtered_images[...,:3]
        filtered_flash = filtered_images[...,3:]
        return self.llf(filtered_ambient, filtered_flash, filtered_ambient, filtered_flash, self.llf_alpha_h, self.llf_alpha_i), filtered_ambient, filtered_flash
        # return self.llf(filtered_ambient, filtered_flash, self.llf_alpha_h, self.llf_alpha_i)

    @tf.function
    def forward(self, inputs):
        outputs = edict()
        denoised_flash, filtered_ambient, filtered_flash = self.filter_flash_ambient(inputs.net_ft_input)
        
        outputs.output = denoised_flash
        outputs.alpha_map_i = self.llf_alpha_i
        outputs.alpha_map_h = self.llf_alpha_h
        outputs.llf_input = filtered_ambient
        outputs.llf_guide = filtered_flash
        return outputs