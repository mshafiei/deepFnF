from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu
from tiny_unet import Net as tiny_unet
from easydict import EasyDict as edict
class Net(tiny_unet):
    def __init__(self, downsample_ct=0, unet_output_size=3, channels_count_factor=1):
        super().__init__(unet_output_size=unet_output_size, channels_count_factor=channels_count_factor)
        self.downsample_ct = downsample_ct
        self.channel_count = lambda x: max(1, int(x * self.channels_count_factor)//(2**self.downsample_ct))
        self.channel_count_end = lambda x: max(self.output_dim_size, int(x * self.channels_count_factor)//(2**self.downsample_ct))


    def encode(self, out, pfx=''):
        skips = edict()
        
        out = self.conv(pfx + 'inp', out, self.channel_count(64))
        for i in range(self.downsample_ct):
            out = self.conv(pfx + 'inp_%i'%i, out, self.channel_count(64*2**(i+1)))
            

        if(self.downsample_ct == 0):
            out, skips.d1 = self.down_block(out, self.channel_count(64  ), pfx + 'down1')
        if(self.downsample_ct <= 1):
            out, skips.d2 = self.down_block(out, self.channel_count(128 ), pfx + 'down2')
        if(self.downsample_ct <= 2):
            out, skips.d3 = self.down_block(out, self.channel_count(256 ), pfx + 'down3')
        if(self.downsample_ct <= 3):
            out, skips.d4 = self.down_block(out, self.channel_count(512 ), pfx + 'down4')
        out, skips.d5 = self.down_block(out, self.channel_count(1024), pfx + 'down5')

        out = self.conv(pfx + 'bottleneck_1', out, self.channel_count(1024))
        out = self.conv(pfx + 'bottleneck_2', out, self.channel_count(1024),
                        activation_name=pfx + 'bottleneck')
        self.bottleneck = out
        self.skips = skips
        return out, skips
    
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

        i=0
        for i in range(int(np.log2(self.channel_count(64 )) - np.ceil(np.log2(3)))):
            out = self.conv(pfx + 'end_%i'%(i), out, self.channel_count_end(64)//(2**i), relu=False)

        out = self.conv(pfx + 'end_%i'%(i+1), out, self.output_dim_size,relu=False)
        out = self.conv(pfx + 'end_%i'%(i+2), out, self.output_dim_size,relu=False)
        out = self.conv(pfx + 'end_%i'%(i+3), out, self.output_dim_size,relu=False, activation_name=pfx + 'end')

        # out = self.conv(pfx + 'end_1', out, self.channel_count_end(64))
        # out = self.conv(pfx + 'end_2', out, self.channel_count_end(64), activation_name=pfx + 'end')

        return out

    def downsample(self, inp):
        _, h, w, _ = inp.shape
        #downsample
        return tf.image.resize(inp,(h//(2**self.downsample_ct), w//(2**self.downsample_ct)))
    
    def upsample(self, inp, h, w):
        #upsample
        return tf.image.resize(inp,(h, w))
        
    def resize_encode(self, inp):
        if(self.downsample_ct != 0):
            inp = self.downsample(inp)

        out, skips = self.encode(inp)
        return out, skips

    def resize_decode(self, inp, skips, h, w):
        out = self.decode(inp, skips)
        if(self.downsample_ct != 0):
            #upsample
            return self.upsample(out, h, w)
        else:
            return out

    def lowres_unet(self, inp):
        _, h, w, _ = inp.shape
        out, skips = self.resize_encode(inp)
        return self.resize_decode(out, skips, h, w)
