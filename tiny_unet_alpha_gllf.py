from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu
from easydict import EasyDict as edict
from gllf import gllf_diffable_1d
from tiny_unet_adjustable import Net as tiny_unet
from gllf_halide import halide_gllf

class Net(tiny_unet):
    def __init__(self, llf_levels, downsample_ct, IMSZ=448, unet_output_size=3, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1):
        super().__init__(downsample_ct=downsample_ct, unet_output_size=unet_output_size, num_basis=num_basis, ksz=ksz, burst_length=burst_length, channels_count_factor=channels_count_factor)
        self.llf_levels = llf_levels
        # self.gllf = lambda diffable_imgs, diffable_alphas: halide_gllf(diffable_imgs[0][0], diffable_imgs[1][0], llf_levels, diffable_alphas, 1., 1.)[0]
        self.gllf = lambda diffable_imgs, diffable_alphas: gllf_diffable_1d(diffable_imgs, diffable_alphas, llf_levels, llf_levels, betas=[1,0], sigmas=[1,0], IMSZ=IMSZ)


    def forward_scalars(self, inp):
        if('alpha_i' in self.weights.keys() and 'alpha_j' in self.weights.keys()):
            alpha_i = self.weights['alpha_i']
            alpha_j = self.weights['alpha_j']
        else:
            # w_i = tf.Variable(tf.random.uniform([1], minval=-1, maxval=1, dtype=tf.float32)*0.1)
            # w_j = tf.Variable(tf.random.uniform([1], minval=-1, maxval=1, dtype=tf.float32)*0.1)
            w_i = tf.Variable(0.)
            w_j = tf.Variable(0.)
            self.weights['alpha_i'] = w_i
            self.weights['alpha_j'] = w_j
            alpha_i = w_i
            alpha_j = w_j
        
        output = edict()
        source_images = [tf.clip_by_value(inp.noisy_ambient_scaled,clip_value_min=0,clip_value_max=1000),tf.clip_by_value(inp.noisy_flash_scaled,clip_value_min=0,clip_value_max=1000)]
        # source_images = tf.stack(,axis=0)
        # source_images = tf.clip_by_value(source_images,clip_value_min=0,clip_value_max=1000)
        alphas = [alpha_i, alpha_j]
        output.output = self.gllf(source_images, alphas)
        return output

    
    def forward_encode(self, inp):
        output = edict()
        diffable_alphas = super().encode_scalar(inp.net_ft_input)
        diffable_alphas = tf.reduce_mean(diffable_alphas)
        source_images = tf.stack((inp.noisy_ambient_scaled,inp.noisy_flash_scaled),axis=0)
        alphas = [0,0]
        output.output = self.gllf(source_images, alphas)
        return output

    def forward(self, inp):
        output = edict()
        diffable_alphas = super().forward(inp)
        tensor_shapes = diffable_alphas.output.shape
        diffable_alphas = tf.reshape(diffable_alphas.output, tensor_shapes[:-1] + [3, 2])

        source_images = [tf.clip_by_value(inp.noisy_ambient_scaled,clip_value_min=0,clip_value_max=1000),tf.clip_by_value(inp.noisy_flash_scaled,clip_value_min=0,clip_value_max=1000)]
        alphas = [diffable_alphas[...,0],diffable_alphas[...,1]]
        output.output = self.gllf(source_images, alphas)
        output.alpha_map_i = alphas[0]
        output.alpha_map_h = alphas[1]
        return output
