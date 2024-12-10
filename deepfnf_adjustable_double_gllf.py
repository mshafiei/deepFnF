from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu
from easydict import EasyDict as edict
from tiny_unet_adjustable import Net as tiny_unet
from easydict import EasyDict as edict
from gllf import gllf_diffable_1d

class Net(tiny_unet):
    def __init__(self, llf_levels, IMSZ, deepfnf_upscaling=False,source_images=1, downsample_ct=0, unet_output_size=6, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1):
        super().__init__(downsample_ct=downsample_ct, unet_output_size=unet_output_size, num_basis=num_basis, ksz=ksz, burst_length=burst_length, channels_count_factor=channels_count_factor)
        self.deepfnf_upscaling=deepfnf_upscaling
        self.kernel_channels=1
        if(self.deepfnf_upscaling):
            self.kernel_channels = 2
        else:
            self.kernel_channels = 1
        self.source_images=source_images
        self.gllf = lambda diffable_imgs, diffable_alphas: gllf_diffable_1d(diffable_imgs, diffable_alphas, llf_levels, llf_levels, betas=[1,0], sigmas=[1,0], IMSZ=IMSZ)


    def kernel_up_block(self, out, nch, pfx=''):
        '''
        Upsampling block, including:
            one upsampling with bilinear resizing,
            one conv layer,
            skip connection (with gloal average pooling)
            two more conv layers
        Args:
            out: output from previous layer
            skip: output from the layer that is skip connected to this block
            nch: number of channels for the block
            pfx: prefix of names for layers in this block
        Return:
            out: output of this block
        '''
        shape = tf.shape(out)
        out = tf.image.resize(out, 2 * shape[1:3])
        out = self.conv(pfx + '_1', out, nch, ksz=3, stride=1)

        #no skip connection
        # # resize the skip connection
        # skip = tf.reduce_mean(skip, axis=[1, 2], keepdims=True)
        # skip = tf.tile(skip, [1, 2 * shape[1], 2 * shape[2], 1])
        # out = tf.concat([out, skip], axis=-1)

        out = self.conv(pfx + '_2', out, nch, ksz=3, stride=1)
        out = self.conv(pfx + '_3', out, nch, ksz=3,
                        stride=1, activation_name=pfx)
        return out


    def decode(self, out, skips, pfx=''):
        out = self.up_block(out, self.channel_count(512), skips.d5, pfx + 'up1')
        if(self.downsample_ct <= 3):
            out = self.up_block(out, self.channel_count(256), skips.d4, pfx + 'up2')
        if(self.downsample_ct <= 2):
            out = self.up_block(out, self.channel_count(128), skips.d3, pfx + 'up3')
        if(self.downsample_ct <= 1):
            out = self.up_block(out, self.channel_count(64 ), skips.d2, pfx + 'up4')
        if(self.downsample_ct <= 0):
            out = self.up_block(out, self.channel_count(64 ), skips.d1, pfx + 'up5')

        out = self.conv(pfx + 'end_1', out, self.channel_count_end(64))
        out = self.conv(pfx + 'end_2', out, self.channel_count_end(64), activation_name=pfx + 'end')

        return out
    def lowres_unet(self, inp):
        _, h, w, _ = inp.shape
        input = inp
        #downsample
        input = tf.image.resize(input,(h//(2**self.downsample_ct), w//(2**self.downsample_ct)))
        out, skips = self.encode(input)
        out = self.decode(out, skips)
        #upsample
        output = tf.image.resize(out,(h, w))
        return output
    def create_basis(self):
        '''Predict image-specific basis'''
        assert self.ksz == 15
        bottleneck = self.activations['bottleneck']
        out = tf.reduce_mean(bottleneck, axis=[1, 2], keepdims=True)  # 1x1
        out = self.kernel_up_block(
            out, self.channel_count(512,),'k_up1')  # 2x2
        out = self.kernel_up_block(
            out, self.channel_count(256,), 'k_up2')  # 4x4
        out = self.kernel_up_block(
            out, self.channel_count(256,), 'k_up3')  # 8x8
        out = self.kernel_up_block(
            out, self.channel_count(128), 'k_up4')  # 16x16
        out = self.conv('k_conv', out, self.channel_count(128), ksz=2, stride=1, pad='VALID')
        out = self.conv('k_output_1', out, self.channel_count(128))
        #one  self.burst_length is for multiplication w/ basis and another is for the kernel
        out = self.conv('k_output_2', out, 3 * self.kernel_channels * self.burst_length * self.burst_length * self.num_basis, relu=False)
        out = tf.reshape(
            out, [-1, self.ksz * self.ksz * 3 * self.kernel_channels * self.burst_length, self.num_basis * self.burst_length])
        self.basis = tf.transpose(out, [0, 2, 1])

    def predict_coeff(self, inp):
        '''Predict per-pixel coefficient vector given the input'''
        self.imsp = tf.shape(inp)

        
        out = self.lowres_unet(inp)
        #  * 2 because of alpha
        out = self.conv('output', out, self.num_basis * self.burst_length + 3*self.burst_length * 2, relu=False)
        self.coeffs_pre_soft = out
        self.coeffs = out[..., :self.num_basis * self.burst_length]
        self.scale = out[..., -3*self.burst_length*2: -3*self.burst_length]
        self.alphas = out[..., -3*self.burst_length:]
        self.activations['output'] = self.coeffs

    def combine(self):
        '''Combine coeffs and basis to get a per-pixel kernel'''
        imsp = self.imsp
        coeffs = tf.reshape(
            self.coeffs, [-1, imsp[1] * imsp[2], self.num_basis*self.burst_length])
        self.kernels = tf.matmul(
            coeffs,
            self.basis
        )  # (h * w) x (ksz * ksz * 3 * 2)
        self.kernels = tf.reshape(
            self.kernels, [-1, imsp[1], imsp[2], self.ksz * self.ksz * 3 * self.burst_length, self.kernel_channels])
        self.activations['decoding'] = self.kernels

    def forward(self, inp):
        inp_is_edict = type(inp) == edict
        if(inp_is_edict):
            noflash_wb_fn = inp.noflash_wb_fn
            flash_wb_fn = inp.flash_wb_fn
            inp = inp.net_ft_input
            
        # ambient_max = tf.reduce_max(inp[...,:3])
        # flash_max = tf.reduce_max(inp[...,3:6])
        # scale_ratio = flash_max/ambient_max
        # scale_ratio = 1/0.0848
        self.predict_coeff(inp)
        self.create_basis()
        self.combine()

        filtered_images = tfu.apply_filtering(
            inp[:, :, :, :6], self.kernels[..., 0], framewise_op = True)
        
        # "Bilinearly upsample kernels + filtering"
        # is equivalent to
        # "filter the image with a bilinear kernel + dilated filter the image
        # with the original kernel".
        # This will save more memory.
        # smoothed_ambient = tfu.bilinear_filter(inp[:, :, :, :3], ksz=7)
        # smoothed_ambient = tfu.apply_dilated_filtering(
        #     smoothed_ambient, self.kernels[..., 1], dilation=4)
        # filtered_ambient = filtered_ambient + smoothed_ambient
        filtered_ambient = (filtered_images[...,:3] * self.scale[...,:3])# * scale_ratio
        filtered_flash = (filtered_images[...,3:] * self.scale[...,3:])

        # filtered_ambient = noflash_wb_fn(filtered_ambient)
        # filtered_flash = flash_wb_fn(filtered_flash)
        self.alpha_i = self.alphas[...,-6:-3]
        self.alpha_h = self.alphas[...,-3:]
        
        source_images = [tf.clip_by_value(filtered_ambient,0,1000),tf.clip_by_value(filtered_flash,0,1000)]
        alphas = [self.alpha_i, self.alpha_h]

        
        if(inp_is_edict):
            output=edict()
            output.output = self.gllf(source_images, alphas)
            output.alpha_map_i = self.alpha_i
            output.alpha_map_h = self.alpha_h
            return output
        else:
            return filtered_ambient + filtered_flash