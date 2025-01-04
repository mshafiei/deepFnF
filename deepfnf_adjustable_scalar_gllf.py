from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu
from easydict import EasyDict as edict
from tiny_unet_adjustable import Net as tiny_unet
from easydict import EasyDict as edict
# from gllf import gllf_diffable_1d
from gllf.gllf_layer import *
class Net(gllf_layer_radial):
    def __init__(self, yuv_gllf, alphas, betas, sigmas, llf_intensity_levels, llf_levels, thresholds, IMSZ, llf_remap_function, basis_ct, input_images, img_ct=None,deepfnf_upscaling=False,source_images=1, downsample_ct=0, num_basis=90, ksz=15, burst_length=2, unet_output_size=6, **kwargs):
        super().__init__(llf_levels, llf_intensity_levels, img_ct=len(input_images), basis_ct=basis_ct, llf_remap_function=llf_remap_function, yuv_gllf=yuv_gllf, alphas=alphas, betas=betas, sigmas=sigmas, thresholds=thresholds, downsample_ct=downsample_ct, IMSZ=IMSZ, unet_output_size=unet_output_size, **kwargs)
        self.deepfnf_upscaling=deepfnf_upscaling
        self.kernel_channels=1
        self.num_basis = num_basis
        self.burst_length = burst_length
        self.ksz = ksz
        self.input_images = input_images
        if(self.deepfnf_upscaling):
            self.kernel_channels = 2
        else:
            self.kernel_channels = 1
        self.source_images=source_images
        self.llf_levels = llf_levels
        self.llf_intensity_levels = llf_intensity_levels
        self.IMSZ = IMSZ

        self.coeff_count = self.num_basis * self.burst_length
        self.scale_count = 3*self.burst_length*4

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
        out, skips = self.encode(inp)
        out = self.resize_decode(out, skips, self.imsp[1], self.imsp[2])
        #  * 2 because of alpha
        out = self.conv('output', out, self.coeff_count+self.scale_count, relu=False)
        self.coeffs_pre_soft = out
        self.coeffs = out[..., :self.coeff_count]
        self.scale = out[..., self.coeff_count: self.coeff_count+self.scale_count]
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
    
    def visualize(self, inp):
        return self.forward(inp, visualize=True)

    @tf.function
    def forward(self, inp, visualize=False):
        inp_is_edict = type(inp) == edict
        if(inp_is_edict):
            # noflash_wb_fn = inp.noflash_wb_fn
            # flash_wb_fn = inp.flash_wb_fn
            alpha = inp.alpha
            color_matrix = inp.color_matrix
            adapt_matrix = inp.adapt_matrix
            inp = inp.net_ft_input
            
        # ambient_max = tf.reduce_max(inp[...,:3])
        # flash_max = tf.reduce_max(inp[...,3:6])
        # scale_ratio = flash_max/ambient_max
        # scale_ratio = 1/0.0848
        self.imsp = tf.shape(inp)
        lowres_input = self.downsample(inp)
        self.predict_coeff(lowres_input)
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
        filtered_ambient = (filtered_images[...,:3])# * scale_ratio
        filtered_flash = (filtered_images[...,3:])
        
        filtered_ambient_scaled = tfu.camera_to_rgb(
            filtered_ambient / alpha, color_matrix, adapt_matrix, do_gamma_correct=False)
        
        filtered_flash_scaled = tfu.camera_to_rgb(
            filtered_flash, color_matrix, adapt_matrix, do_gamma_correct=False)

        if(np.sum(self.scalar_alphas_options) == 0 and np.sum(self.scalar_betas_options) == 0 and np.sum(self.scalar_sigmas_options) == 0):
            bpn_out=tfu.gamma_correct(filtered_ambient_scaled + filtered_flash_scaled)
            return edict(output=bpn_out)

        ambient_scaled = tfu.camera_to_rgb(
            inp[:, :, :, :3] / alpha, color_matrix, adapt_matrix, do_gamma_correct=False)
        flash_scaled = tfu.camera_to_rgb(
            inp[:, :, :, 3:6], color_matrix, adapt_matrix, do_gamma_correct=False)
        
        # source_images = [filtered_ambient_scaled, filtered_flash_scaled, ambient_scaled, flash_scaled]
        source_images = []
        if('bpn_ambient' in self.input_images):
            source_images.append(filtered_ambient_scaled)
        
        if('bpn_flash' in self.input_images):
            source_images.append(filtered_flash_scaled)
        
        if('noisy_ambient' in self.input_images):
            source_images.append(ambient_scaled)
        
        if('noisy_flash' in self.input_images):
            source_images.append(flash_scaled)


        if(inp_is_edict):
            output=edict()
            if(visualize):
                visualization = self.gllf(source_images, self.bottleneck, reconstruct_gllf_pyramids=visualize)
                for i in range(len(visualization)):
                    visualization[i].image = tfu.gamma_correct(visualization[i].image)
                return visualization
            else:
                output.output = self.gllf(source_images, self.bottleneck, reconstruct_gllf_pyramids=visualize)
                output.output = tfu.gamma_correct(output.output)
                return output
        else:
            return filtered_ambient + filtered_flash