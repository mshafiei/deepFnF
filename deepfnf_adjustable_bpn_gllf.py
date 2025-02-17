import tensorflow as tf

import utils.tf_utils as tfu
from easydict import EasyDict as edict
from gllf.gllf_layer import *

class Net(gllf_layer_radial):
    def __init__(self, input_images, downsample_ct=0, unet_output_size=6, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1,**kwargs):
        super().__init__(downsample_ct=downsample_ct,unet_output_size=unet_output_size, channels_count_factor=channels_count_factor,**kwargs)
        self.downsample_ct = downsample_ct
        self.num_basis = num_basis
        self.ksz = ksz
        self.burst_length = burst_length
        self.input_images = input_images
        self.img_ct = len(self.input_images)
        if('noisy_ambient' in self.input_images and (not ('noisy_flash' in self.input_images)) and (not ('deep_denoised' in self.input_images))):
            self.sources_mode = 1
        elif('noisy_ambient' in self.input_images and 'noisy_flash' in self.input_images and not('deep_denoised' in self.input_images)):
            self.sources_mode = 2
        elif('noisy_ambient' in self.input_images and 'noisy_flash' in self.input_images and 'deep_denoised' in self.input_images):
            self.sources_mode = 3
        else:
            print('Could not interprete the input images')
            exit(0)


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
        out = self.conv('k_output_2', out, 3 * 2 * self.num_basis, relu=False)
        out = tf.reshape(
            out, [-1, self.ksz * self.ksz * 3 * 2, self.num_basis])
        self.basis = tf.transpose(out, [0, 2, 1])

    def predict_coeff(self, inp):
        '''Predict per-pixel coefficient vector given the input'''
        self.imsp = tf.shape(inp)

        
        out = self.lowres_unet(inp)
        # out, skips = self.encode(inp)
        # out = self.decode(out, skips)
        out = self.conv('output', out, self.num_basis + 6, relu=False)
        self.coeffs_pre_soft = out
        self.coeffs = out[..., :self.num_basis]
        self.scale = out[..., -3:]
        self.activations['output'] = self.coeffs

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
            self.kernels, [-1, imsp[1], imsp[2], self.ksz * self.ksz * 3, 2])
        self.activations['decoding'] = self.kernels

    def visualize(self, inpt):
        return self.forward(inpt, visualize=True)
    
    @tf.function
    def forward(self, inp, visualize=False):
        inp_is_edict = type(inp) == edict
        if(inp_is_edict):
            alpha = inp.alpha
            color_matrix = inp.color_matrix
            adapt_matrix = inp.adapt_matrix
            inp = inp.net_ft_input

        self.predict_coeff(inp)
        self.create_basis()
        self.combine()

        filtered_ambient = tfu.apply_filtering(
            inp[:, :, :, :3], self.kernels[..., 0])

        #add the flash and the noisy image
        ambient_scaled = tfu.camera_to_rgb(
            inp[:, :, :, :3] / alpha, color_matrix, adapt_matrix, do_gamma_correct=False)
        flash_scaled = tfu.camera_to_rgb(
            inp[:, :, :, 3:6], color_matrix, adapt_matrix, do_gamma_correct=False)
        
        source_images = []
        if(self.sources_mode >=1):
            source_images.append(ambient_scaled)
        
        if(self.sources_mode >=2):
            source_images.append(flash_scaled)

        if(self.sources_mode >=3):
            filtered_ambient = tfu.camera_to_rgb(
                filtered_ambient / alpha, color_matrix, adapt_matrix, do_gamma_correct=False)
            source_images.append(filtered_ambient)
            image_weights = self.decode_per_layer(self.bottleneck, self.skips, self.max_levels, "decode_per_layer_")
            self.image_weights = [image_weights.d4, image_weights.d3, image_weights.d2, image_weights.d1]

        if(visualize):
            visualization = self.gllf(source_images, self.bottleneck, reconstruct_gllf_pyramids=visualize)
            for i in range(len(visualization)):
                visualization[i].image = tfu.gamma_correct(visualization[i].image)
            return visualization
        else:
            output=edict()
            output.output = self.gllf(source_images, self.bottleneck, reconstruct_gllf_pyramids=visualize)
            output.output = tfu.gamma_correct(output.output)
            return output