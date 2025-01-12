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
    
    def forward_tmp(self, inp):
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

    def forward(self, inp):
        output = edict()
        diffable_alphas = super().encode_scalar(inp.net_ft_input) #1, 2, 2, 1
        diffable_alphas = tf.reduce_mean(diffable_alphas,axis=2)[0,:,0]
        source_images = [tf.clip_by_value(inp.noisy_ambient_scaled,clip_value_min=0,clip_value_max=1000),tf.clip_by_value(inp.noisy_flash_scaled,clip_value_min=0,clip_value_max=1000)]
        # alphas = [0, 0]
        output.output = self.gllf(source_images, diffable_alphas)
        return output

    def forward(self, inp):
        _, h, w, _ = inp.net_ft_input.shape
        input = inp.net_ft_input
        #downsample
        if(self.downsample_ct > 0):
            input = tf.image.resize(input,(h//(2**self.downsample_ct), w//(2**self.downsample_ct)))
        out, _ = self.encode(input)

        out, _ = self.down_block(out, self.channel_count(512), 'down6')
        out, _ = self.down_block(out, self.channel_count(256), 'down7')
        out, _ = self.down_block(out, self.channel_count(128), 'down8')
        out = self.conv('bottleneck_4', out, self.channel_count(64),relu=False)
        out = self.conv('bottleneck_5', out, self.channel_count(32),relu=False)
        out = self.conv('bottleneck_6', out, self.channel_count(16),relu=False)
        out = self.conv('bottleneck_7', out, self.channel_count(8),relu=False)
        out = self.conv('bottleneck_8', out, self.channel_count(4),relu=False)
        diffable_alphas = self.conv('bottleneck_9', out, self.channel_count(2),relu=False, activation_name='bottleneck') #1, 2, 2, 1
        diffable_alphas = tf.reduce_mean(diffable_alphas,axis=2)[0,:,0]
        
        output = edict()
        source_images = [tf.clip_by_value(inp.noisy_ambient_scaled,clip_value_min=0,clip_value_max=1000),tf.clip_by_value(inp.noisy_flash_scaled,clip_value_min=0,clip_value_max=1000)]
        output.output = self.gllf(source_images, diffable_alphas)
        return output
