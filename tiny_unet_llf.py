from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu
from easydict import EasyDict as edict
from gllf import gllf_diffable_1d
from tiny_unet import Net as tiny_unet


class Net(tiny_unet):
    def __init__(self, llf_levels, IMSZ=448, unet_output_size=3, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1):
        super().__init__(unet_output_size=unet_output_size, num_basis=num_basis, ksz=ksz, burst_length=burst_length, channels_count_factor=channels_count_factor)
        self.llf_levels = llf_levels
        # self.gllf = lambda diffable_imgs, diffable_alphas: halide_gllf(diffable_imgs[0][0], diffable_imgs[1][0], llf_levels, diffable_alphas, 1., 1.)[0]
        self.gllf = lambda diffable_imgs, diffable_alphas: gllf_diffable_1d(diffable_imgs, diffable_alphas, llf_levels, llf_levels, betas=[1.], sigmas=[1.], IMSZ=IMSZ)


    def forward(self, inp):
        output = edict()
        diffable_alphas = super().forward(inp)
        tensor_shapes = diffable_alphas.output.shape
        # source_images = tf.reshape(inp.net_ft_input[...,:6],tensor_shapes[:-1] + [3, 2])
        # source_images = tf.transpose(source_images, (4,0,1,2,3))
        # diffable_alphas = tf.reshape(diffable_alphas.output, tensor_shapes[:-1] + [3, 2])

        
        source_images = [tf.clip_by_value(inp.noisy_ambient_scaled,0,1000)]
        # diffable_alphas = tf.transpose(diffable_alphas, (4,0,1,2,3))
        # diffable_alphas = tf.reduce_mean(diffable_alphas)
        alphas = [diffable_alphas.output]
        output.output = self.gllf(source_images, alphas)
        output.alpha_map_i = diffable_alphas.output
        return output
        # output.output = tf.reduce_sum(diffable_alphas,axis=-1)
        # return output