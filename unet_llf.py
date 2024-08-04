from collections import OrderedDict
from guided_local_laplacian_color import guided_local_laplacian_color
import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu
import utils.utils as ut
import time

class Net:
    def __init__(self,alpha, beta, levels, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1, lmbda=1):
        self.weights = {}
        self.activations = OrderedDict()
        self.num_basis = num_basis
        self.ksz = ksz
        self.burst_length = burst_length
        self.channels_count_factor = channels_count_factor
        self.channel_count = lambda x: max(1, int(x * self.channels_count_factor))
        self.alpha = alpha
        self.beta = beta
        self.levels = levels
        self.lmbda = lmbda

    def conv(
            self, name, inp, outch, ksz=3,
            stride=1, relu=True, pad='SAME', activation_name=None):
        '''Wrapper of conv'''
        inch = inp.get_shape().as_list()[-1]
        ksz = [ksz, ksz, inch, outch]

        wnm = name + "_w"
        if wnm in self.weights.keys():
            w = self.weights[wnm]
        else:
            sq = np.sqrt(3.0 / np.float32(ksz[0] * ksz[1] * ksz[2]))
            w = tf.Variable(tf.random.uniform(
                ksz, minval=-sq, maxval=sq, dtype=tf.float32))
            self.weights[wnm] = w

        out = tf.nn.conv2d(inp, w, [1, stride, stride, 1], pad)

        bnm = name + "_b"
        if bnm in self.weights.keys():
            b = self.weights[bnm]
        else:
            b = tf.Variable(tf.constant(0, shape=[ksz[-1]], dtype=tf.float32))
            self.weights[bnm] = b
        out = out + b

        if relu:
            out = tf.nn.relu(out)

        if activation_name is not None:
            self.activations[activation_name] = out

        return out

    def down_block(self, out, nch, pfx=''):
        '''
        Downsampling block, including two conv layers w/ one maxpooling layer
        Args:
            out: output from previous layer
            nch: number of channels for the block
            pfx: prefix of names for layers in this block
        Return:
            down: output of the block after downsampling
            out: output of the block right before downsampling
        '''
        out = self.conv(pfx + '_1', out, nch, ksz=3, stride=1)
        out = self.conv(pfx + '_2', out, nch, ksz=3, stride=1)
        down = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        self.activations[pfx] = down
        self.activations['skip_' + pfx] = out
        return down, out

    def up_block(self, out, nch, skip, pfx=''):
        '''
        Upsampling block, including:
            one upsampling with bilinear resizing,
            one conv layer,
            skip connection
            two more conv layers
        Args:
            out: output from previous layer
            skip: output from the layer that is skip connected to this block
            nch: number of channels for the block
            pfx: prefix of names for layers in this block
        Return:
            out: output of this block
        '''
        out = tf.image.resize(out, 2 * tf.shape(out)[1:3])
        out = self.conv(pfx + '_1', out, nch, ksz=3, stride=1)

        out = tf.concat([out, skip], axis=-1)

        out = self.conv(pfx + '_2', out, nch, ksz=3, stride=1)
        out = self.conv(pfx + '_3', out, nch, ksz=3,
                        stride=1, activation_name=pfx)
        return out

    def kernel_up_block(self, out, nch, skip, pfx=''):
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

        # resize the skip connection
        skip = tf.reduce_mean(skip, axis=[1, 2], keepdims=True)
        skip = tf.tile(skip, [1, 2 * shape[1], 2 * shape[2], 1])
        out = tf.concat([out, skip], axis=-1)

        out = self.conv(pfx + '_2', out, nch, ksz=3, stride=1)
        out = self.conv(pfx + '_3', out, nch, ksz=3,
                        stride=1, activation_name=pfx)
        return out


    def encode(self, out, pfx=''):
        out = self.conv(pfx + 'inp', out, self.channel_count(64))

        out, d1 = self.down_block(out, self.channel_count(64  ), pfx + 'down1')
        out, d2 = self.down_block(out, self.channel_count(128 ), pfx + 'down2')
        out, d3 = self.down_block(out, self.channel_count(256 ), pfx + 'down3')
        out, d4 = self.down_block(out, self.channel_count(512 ), pfx + 'down4')
        out, d5 = self.down_block(out, self.channel_count(1024), pfx + 'down5')

        out = self.conv(pfx + 'bottleneck_1', out, self.channel_count(1024))
        out = self.conv(pfx + 'bottleneck_2', out, self.channel_count(1024),
                        activation_name=pfx + 'bottleneck')
        return out, [d1, d2, d3, d4, d5]

    def decode(self, out, skips, pfx=''):
        d1, d2, d3, d4, d5 = skips
        out = self.up_block(out, self.channel_count(512), d5, pfx + 'up1')
        out = self.up_block(out, self.channel_count(256), d4, pfx + 'up2')
        out = self.up_block(out, self.channel_count(128), d3, pfx + 'up3')
        out = self.up_block(out, self.channel_count(64 ), d2, pfx + 'up4')
        out = self.up_block(out, self.channel_count(64 ), d1, pfx + 'up5')

        out = self.conv(pfx + 'end_1', out, self.channel_count(3))
        out = self.conv(pfx + 'end_2', out, self.channel_count(3), activation_name=pfx + 'end')

        return out

    def forward(self, inp, alpha):
        '''Predict per-pixel coefficient vector given the input'''
        self.imsp = tf.shape(inp)

        out, skips = self.encode(inp)
        out = self.decode(out, skips)
        denoised = self.conv('output', out, 3, relu=False)
        flash = inp[:, :, :, 3:6] * alpha / ut.FLASH_STRENGTH

        return denoised, flash

    def llf(self, denoised, flash):
        flash_np = np.ascontiguousarray(np.array(flash[0,...]).squeeze().transpose(2,0,1))
        denoised_np = np.ascontiguousarray(np.array(denoised[0,...]).squeeze().transpose(2,0,1))
        c, h, w, = flash_np.shape
        combined = np.empty([3, h, w], dtype=denoised_np.dtype)
        intensity_max = max(flash_np.max(), denoised_np.max())
        intensity_min = min(flash_np.min(), denoised_np.min())
        denoised_np = (denoised_np - intensity_min) / (intensity_max - intensity_min)
        flash_np = (flash_np - intensity_min) / (intensity_max - intensity_min)
        # guided_local_laplacian(flash_np, denoised_np, self.levels, self.alpha / (self.levels - 1), self.beta, combined)
        start = time.time_ns()
        guided_local_laplacian_color(flash_np, denoised_np, self.levels, self.alpha / (self.levels - 1), self.beta, combined)
        tf.print('llf_time ', (time.time_ns() - start)/1000000)
        combined = tf.convert_to_tensor(combined.transpose(1,2,0))[None,...]
        combined = combined * (intensity_max - intensity_min) + intensity_min

        return combined