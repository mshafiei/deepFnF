from collections import OrderedDict
from net import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time
from net_llf_tf2_tf_local_alpha_diffable import Net as NetAlpha

from gllf import gllf, _resize

class Net(NetAlpha):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        assert self.alpha_height == self.alpha_width
        assert self.IMSZ % self.alpha_width == 0
        self.alpha_is_scalar = self.alpha_width == 1
        if(not self.alpha_is_scalar):
            self.layers_count = np.floor(5.0 - np.log2(self.IMSZ // self.alpha_width))
        
        self.llf = lambda input, guide, alpha_: gllf(input, guide, self.levels, self.levels, alpha_, IMSZ=self.IMSZ, beta=self.beta, sigma=1)
        self.out_image_count = 1
        
        

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

    def decode(self, out, skips, pfx='', layers_count=5):
        d1, d2, d3, d4, d5 = skips
        
        if(layers_count <= -1):
            #8x8
            out = tf.pad(out,tf.constant([[0,0],[1,1],[1,1],[0,0]]),"REFLECT")
            out, _ = self.down_block(out, self.channel_count(256), pfx + 'down7')
        if(layers_count <= -2):
            #4x4
            out, _ = self.down_block(out, self.channel_count(128), pfx + 'down8')
        if(layers_count <= -3):
            #2x2
            out, _ = self.down_block(out, self.channel_count(128), pfx + 'down9')

        if(layers_count >= 1):
            out = self.up_block(out, self.channel_count(512), d5, pfx + 'up1')
        if(layers_count >= 2 and layers_count >= 1):
            out = self.up_block(out, self.channel_count(256), d4, pfx + 'up2')
        if(layers_count >= 3 and layers_count >= 1):
            out = self.up_block(out, self.channel_count(128), d3, pfx + 'up3')
        if(layers_count >= 4 and layers_count >= 1):
            out = self.up_block(out, self.channel_count(64 ), d2, pfx + 'up4')
        if(layers_count == 5 and layers_count >= 1):
            out = self.up_block(out, self.channel_count(64 ), d1, pfx + 'up5')

        out = self.conv(pfx + 'end_1', out, self.channel_count(64))
        out = self.conv(pfx + 'end_2', out, self.channel_count(64), activation_name=pfx + 'end')

        return out

    def unet(self, inp):
        '''Predict per-pixel coefficient vector given the input'''
        self.imsp = tf.shape(inp)

        out, skips = self.encode(inp)
        if(self.alpha_is_scalar):
            return tf.reduce_sum(out)
        else:
            out = self.decode(out, skips, layers_count=self.layers_count)
            out = self.conv('output', out, 3 * self.out_image_count, relu=False) #alpha, input, output
            outputs = []
            for i in range(self.out_image_count):
                outputs.append(out[...,3*i:3*(i+1)])
            return outputs

    def forward(self, inp, flash, denoised):
        if(self.out_image_count == 1):
            llf_alpha = self.unet(inp)[0]
            h, w = denoised.shape[0:2]
            ah, aw = llf_alpha.shape[0:2]
            if(h > ah):
                llf_alpha = _resize(llf_alpha, (w, h))
            else:
                llf_alpha = llf_alpha
            
            return self.llf(denoised, flash, llf_alpha), llf_alpha
        elif(self.out_image_count == 3):
            llf_input, llf_flash, llf_alpha = self.unet(inp)
            h, w = llf_input.shape[0:2]
            ah, aw = llf_alpha.shape[0:2]
            if(h > ah):
                llf_alpha = _resize(llf_alpha, (w, h))
            else:
                llf_alpha = llf_alpha
            
            return self.llf(llf_input, llf_flash, llf_alpha), llf_input, llf_flash, llf_alpha