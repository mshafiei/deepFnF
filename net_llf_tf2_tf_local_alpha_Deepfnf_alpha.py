from collections import OrderedDict
from net import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time
from net_llf_tf2_tf_local_alpha_diffable import Net as NetAlpha

from gllf import gllf, _resize
from easydict import EasyDict as dotmap

class Net(NetAlpha):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        assert self.alpha_height == self.alpha_width
        assert self.IMSZ % self.alpha_width == 0
        self.alpha_is_scalar = self.alpha_width == 1
        if(not self.alpha_is_scalar):
            self.layers_count = np.floor(5.0 - np.log2(self.IMSZ // self.alpha_width))
        
        self.llf = lambda input, guide, input_pyramid, guide_pyramid, alpha_: gllf(input, guide, input_pyramid, guide_pyramid, self.levels, self.levels, alpha_, IMSZ=self.IMSZ, beta=self.beta, sigma=1)
        
    def filter_flash(self, inp):
        self.predict_coeff(inp)
        self.create_basis()
        self.combine()

        filtered_flash = tfu.apply_filtering(
            inp[:, :, :, 3:6], self.kernels[..., 0])

        # "Bilinearly upsample kernels + filtering"
        # is equivalent to
        # "filter the image with a bilinear kernel + dilated filter the image
        # with the original kernel".
        # This will save more memory.
        smoothed_flash = tfu.bilinear_filter(inp[:, :, :, 3:6], ksz=7)
        smoothed_flash = tfu.apply_dilated_filtering(
            smoothed_flash, self.kernels[..., 1], dilation=4)
        filtered_flash = filtered_flash + smoothed_flash
        denoised = filtered_flash * self.scale

        return denoised

    def unet(self, inp):
        '''Predict per-pixel coefficient vector given the input'''
        self.imsp = tf.shape(inp)

        out, skips = self.encode(inp,'unet')
        out = self.decode(out, skips,'unet')
        out = self.conv('unet_output', out, 3, relu=False)
        return out

    @tf.function
    def forward(self, inputs_dict):
        inputs = dotmap(inputs_dict)
        outputs = dotmap()
        flash = inputs.noisy_flash_scaled
        denoised = inputs.deepfnf_scaled
        color_matrix = inputs.color_matrix
        adapt_matrix = inputs.adapt_matrix
        denoised_flash = self.filter_flash(inputs.net_ft_input)
        
        denoised_flash_scaled = tfu.camera_to_rgb(
            denoised_flash, color_matrix, adapt_matrix)
        # denoised_flash_scaled = self.unet(inputs.net_ft_input)
        # # denoised_flash_scaled = flash
        # llf_alpha = self.unet(inp)
        # h, w = denoised.shape[1:3]
        # ah, aw = llf_alpha.shape[1:3]
        # if(h > ah):
        #     llf_alpha = _resize(llf_alpha, (w, h))
        # else:
        #     llf_alpha = llf_alpha
        # return denoised_flash_scaled, denoised, denoised_flash_scaled, flash
        # return self.llf(denoised, denoised_flash_scaled, 1), denoised, denoised_flash_scaled, self.llf_alpha
        outputs.gllf_out = self.llf(denoised, denoised_flash_scaled, denoised, flash, self.llf_alpha)
        outputs.llf_input = denoised
        outputs.llf_guide = denoised_flash_scaled
        outputs.llf_alpha = self.llf_alpha
        return dict(outputs)
        # return self.llf(denoised, denoised_flash_scaled, denoised, denoised_flash_scaled, 1), denoised, denoised_flash_scaled, denoised_flash_scaled