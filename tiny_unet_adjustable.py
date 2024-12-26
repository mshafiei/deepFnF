from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu
from tiny_unet import Net as tiny_unet
from easydict import EasyDict as edict
class Net(tiny_unet):
    def __init__(self, downsample_ct=0, unet_output_size=3, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1):
        super().__init__(unet_output_size=unet_output_size, num_basis=num_basis, ksz=ksz, burst_length=burst_length, channels_count_factor=channels_count_factor)
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
        return out, skips
    
    def decode(self, out, skips, pfx=''):
        assert self.downsample_ct<=3
        out = self.up_block(out, self.channel_count(512), skips.d5, pfx + 'up1')
        if(self.downsample_ct <= 3):
            out = self.up_block(out, self.channel_count(256), skips.d4, pfx + 'up2')
        if(self.downsample_ct <= 2):
            out = self.up_block(out, self.channel_count(128), skips.d3, pfx + 'up3')
        if(self.downsample_ct <= 1):
            out = self.up_block(out, self.channel_count(64 ), skips.d2, pfx + 'up4')
        if(self.downsample_ct <= 0):
            out = self.up_block(out, self.channel_count(64 ), skips.d1, pfx + 'up5')

        for i in range(self.downsample_ct-1,1,-1):
            out = self.conv(pfx + 'end_conv_%i'%i, out, self.channel_count(64*i), relu=False)#128 or 64

        out = self.conv(pfx + 'end_1', out, self.channel_count_end(64), relu=False)
        out = self.conv(pfx + 'end_2', out, self.channel_count_end(32), relu=False)
        out = self.conv(pfx + 'end_3', out, self.channel_count_end(16), relu=False)
        out = self.conv(pfx + 'end_4', out, self.channel_count_end(8), relu=False)
        out = self.conv(pfx + 'end_5', out, self.output_dim_size, relu=False, activation_name=pfx + 'end')

        return out

    def forward(self, inp):
        _, h, w, _ = inp.net_ft_input.shape
        input = inp.net_ft_input
        #downsample
        if(self.downsample_ct > 0):
            input = tf.image.resize(input,(h//(2**self.downsample_ct), w//(2**self.downsample_ct)))
        out, skips = self.encode(input)
        output = self.decode(out, skips)
        if(self.downsample_ct > 0):
            output = tf.image.resize(output,(h, w))
        #upsample
        return edict(output=output)