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
    
    def dense(self, inp, outch, name):
        inch = inp.get_shape().as_list()[-1] # B, H, W, C

        wnm = name + "_w"
        if wnm in self.weights.keys():
            W = self.weights[wnm]['W']
        else:
            sq = np.sqrt(2.0 / np.float32(inch*outch))
            W = tf.Variable(tf.random.uniform([inch, outch], minval=-sq, maxval=sq, dtype=tf.float32))
            self.weights[wnm] = W

        wnm = name + "_b"
        if wnm in self.weights.keys():
            b = self.weights[wnm]['b']
        else:
            sq = np.sqrt(2.0 / np.float32(inch*outch))
            b = tf.Variable(tf.zeros([outch]))
            self.weights[wnm] = b

        x = tf.matmul(x, W) + b
        return tf.nn.relu(x)

    def remapping_mlps(self, inp, name):
        '''Wrapper of mlp'''

        hidden_units1 = 128   # Units in the first hidden layer
        hidden_units2 = 64    # Units in the second hidden layer
        output_classes = 1   # Number of output classes (0-9 digits)

        d1  = self.dense(inp,hidden_units1,name + 'dense1')
        d2  = self.dense(d1,hidden_units2 ,name + 'dense2')
        out = self.dense(d2,output_classes,name + 'dense3')
        return tf.nn.softmax(out)

    def encode(self, out, pfx=''):
        out = self.conv(pfx + 'inp', out, self.channel_count(64))

        out, d1 = self.down_block(out, self.channel_count(64  ), pfx + 'down1')
        out, d2 = self.down_block(out, self.channel_count(128 ), pfx + 'down2')
        out, d3 = self.down_block(out, self.channel_count(256 ), pfx + 'down3')
        out, d4 = self.down_block(out, self.channel_count(512 ), pfx + 'down4')
        out, d5 = self.down_block(out, self.channel_count(1024), pfx + 'down5')

        out_bottleneck1 = self.conv(pfx + 'bottleneck_1', out, self.channel_count(1024))
        mlp_weights = self.conv(pfx + 'mlp_weights', out_bottleneck1, 1024)
        mlp_weights = tf.reshape(mlp_weights,-1)
        
        hidden_units1 = 64   # Units in the first hidden layer
        hidden_units2 = 64    # Units in the second hidden layer
        output_classes = 1   # Number of output classes (0-9 digits)
        self.weights['dense1_w'] = mlp_weights[:hidden_units1]
        last_idx = hidden_units1
        self.weights['dense1_b'] = mlp_weights[last_idx:last_idx+hidden_units1]
        last_idx = hidden_units1+hidden_units1
        self.weights['dense2_w'] = mlp_weights[last_idx:last_idx+hidden_units1*hidden_units2]
        
        64+64*128+128*64+64
        # mlp_weights
        out = self.conv(pfx + 'bottleneck_2', out_bottleneck1, self.channel_count(1024),
                        activation_name=pfx + 'bottleneck')
        return out, [d1, d2, d3, d4, d5]

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
        return self.llf(filtered_ambient, filtered_flash, filtered_ambient, filtered_flash, self.llf_alpha_h, self.llf_alpha_i)
        # return self.llf(filtered_ambient, filtered_flash, self.llf_alpha_h, self.llf_alpha_i)

    @tf.function
    def forward(self, inputs):
        outputs = edict()
        denoised_flash = self.filter_flash_ambient(inputs.net_ft_input)
        
        outputs.output = denoised_flash
        return outputs