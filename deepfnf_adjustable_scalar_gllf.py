from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu
from easydict import EasyDict as edict
from tiny_unet_adjustable import Net as tiny_unet
from easydict import EasyDict as edict
from gllf import gllf_diffable_1d

class Net(tiny_unet):
    def __init__(self, yuv_gllf, gllf_scalar, alphas, betas, sigmas, llf_intensity_levels, llf_levels, IMSZ, deepfnf_upscaling=False,source_images=1, downsample_ct=0, unet_output_size=6, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1):
        super().__init__(downsample_ct=downsample_ct, unet_output_size=unet_output_size, num_basis=num_basis, ksz=ksz, burst_length=burst_length, channels_count_factor=channels_count_factor)
        self.deepfnf_upscaling=deepfnf_upscaling
        self.kernel_channels=1
        if(self.deepfnf_upscaling):
            self.kernel_channels = 2
        else:
            self.kernel_channels = 1
        self.yuv_gllf = yuv_gllf
        self.gllf_scalar = gllf_scalar
        self.source_images=source_images
        self.scalar_alphas_options = alphas
        self.scalar_betas_options = betas
        self.scalar_sigmas_options = sigmas
        self.llf_levels = llf_levels
        self.llf_intensity_levels = llf_intensity_levels
        self.IMSZ = IMSZ
        
        self.coeff_count = self.num_basis * self.burst_length
        self.scale_count = 3*self.burst_length*4
        self.alpha_count = 3*4*3

    def gllf(self, diffable_imgs, diffable_alphas, betas, sigmas, thresholds):
        imgs = []
        for im in diffable_imgs:
            if(self.yuv_gllf):
                imgs.append(tfu.rgb_to_yuv(im))
            else:
                imgs.append(im)
        output = gllf_diffable_1d(imgs, diffable_alphas, self.llf_levels, self.llf_intensity_levels, thresholds=thresholds, betas=betas, sigmas=sigmas, min_intensity=0.0, max_intensity=1.0, IMSZ=self.IMSZ)
        if(self.yuv_gllf):
            return tfu.yuv_to_rgb(output)
        else:
            return output

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
        pfx='scalars'
        out_scalars, _ = self.down_block(out, self.channel_count(512), pfx + 'alpha_down5')
        out_scalars, _ = self.down_block(out_scalars, self.channel_count(256), pfx + 'alpha_down6')
        out_scalars = self.conv(pfx + 'alpha_bottleneck_1', out_scalars, self.channel_count(128),relu=False)
        out_scalars = self.conv(pfx + 'alpha_bottleneck_2', out_scalars, self.channel_count(64),relu=False)
        out_scalars = self.conv(pfx + 'alpha_bottleneck_3', out_scalars, self.channel_count(32),relu=False)
        out_scalars = self.conv(pfx + 'alpha_bottleneck_4', out_scalars, self.channel_count(16),relu=False)
        out_scalars = self.conv(pfx + 'alpha_bottleneck_5', out_scalars, self.channel_count(8),relu=False)
        out_scalars = self.conv(pfx + 'alpha_bottleneck_6', out_scalars, self.channel_count(4),relu=False)
        out_scalars = self.conv(pfx + 'alpha_bottleneck_7', out_scalars, self.channel_count(2),relu=False, activation_name=pfx + 'bottleneck')
        self.scalar_alphas = tf.reshape(out_scalars,(1,16))
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
        out = self.conv('output', out, self.coeff_count+self.scale_count+self.alpha_count, relu=False)
        self.coeffs_pre_soft = out
        self.coeffs = out[..., :self.coeff_count]
        self.scale = out[..., self.coeff_count: self.coeff_count+self.scale_count]
        self.alphas = out[..., self.coeff_count+self.scale_count:self.coeff_count+self.scale_count+self.alpha_count]
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
            alpha = inp.alpha
            color_matrix = inp.color_matrix
            adapt_matrix = inp.adapt_matrix
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
        
        # ambient_scaled = tf.clip_by_value(ambient_scaled,0,1)
        # flash_scaled = tf.clip_by_value(flash_scaled,0,1)
        
        # source_images = [tf.clip_by_value(filtered_ambient_scaled,0,1000),tf.clip_by_value(filtered_flash_scaled,0,1000)]
        source_images = [filtered_ambient_scaled, filtered_flash_scaled, ambient_scaled, flash_scaled]
        alphas = [self.scalar_alphas_options[0], self.scalar_alphas_options[1], self.scalar_alphas_options[2], self.scalar_alphas_options[3]]
        betas = [self.scalar_betas_options[0], self.scalar_betas_options[1], self.scalar_betas_options[2], self.scalar_betas_options[3]]
        sigmas = [self.scalar_sigmas_options[0], self.scalar_sigmas_options[1], self.scalar_sigmas_options[2], self.scalar_sigmas_options[3]]
        thresholds = [None, None, 0.1, None]
        # if(self.gllf_scalar):
        #     alphas = [self.scalar_alphas_options[0] * self.scalar_alphas[0,0], self.scalar_alphas_options[1] * self.scalar_alphas[0,1], self.scalar_alphas_options[2] * self.scalar_alphas[0,2],  self.scalar_alphas_options[3] * self.scalar_alphas[0,3]]
        #     betas =  [1,                                                       self.scalar_betas_options[1]  * self.scalar_alphas[0,5], self.scalar_betas_options[2]  * self.scalar_alphas[0,6],  self.scalar_betas_options[3]  * self.scalar_alphas[0,7]]
        #     sigmas = [1,                                                       self.scalar_sigmas_options[1] * self.scalar_alphas[0,9], self.scalar_sigmas_options[2] * self.scalar_alphas[0,10], self.scalar_sigmas_options[3] * self.scalar_alphas[0,11]]
        # else:
        #     alphas = [self.scalar_alphas_options[0] * self.alphas[...,:3], self.scalar_alphas_options[1] * self.alphas[...,3:6],   self.scalar_alphas_options[2] * self.alphas[...,6:9],    self.scalar_alphas_options[3] * self.alphas[...,9:12]]
        #     betas =  [1,                                                   self.scalar_betas_options[1]  * self.alphas[...,12:15], self.scalar_betas_options[2]  * self.alphas[...,15:18],  self.scalar_betas_options[3]  * self.alphas[...,18:21]]
        #     sigmas = [1,                                                   self.scalar_sigmas_options[1] * self.alphas[...,21:24], self.scalar_sigmas_options[2] * self.alphas[...,24:27],  self.scalar_sigmas_options[3] * self.alphas[...,27:30]]
            
        if(inp_is_edict):
            output=edict()
            output.output = self.gllf(source_images, alphas, betas, sigmas, thresholds)
            output.output = tfu.gamma_correct(output.output)
            return output
        else:
            return filtered_ambient + filtered_flash