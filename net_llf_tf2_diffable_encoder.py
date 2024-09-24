from collections import OrderedDict
from net import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time
from net_llf_tf2_diffable import Net as NetScalarAlpha

class Net(NetScalarAlpha):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        

    def encode(self, out, pfx=''):
        out = self.conv(pfx + 'inp', out, self.channel_count(64))

        out, d1 = self.down_block(out, self.channel_count(64  ), pfx + 'down1')
        out, d2 = self.down_block(out, self.channel_count(128 ), pfx + 'down2')
        out, d3 = self.down_block(out, self.channel_count(256 ), pfx + 'down3')
        out, d4 = self.down_block(out, self.channel_count(512 ), pfx + 'down4')
        out, d5 = self.down_block(out, self.channel_count(1024), pfx + 'down5')

        out = self.conv(pfx + 'bottleneck_1', out, self.channel_count(1024))
        out = self.conv(pfx + 'bottleneck_2', out, self.channel_count(1024),
                        activation_name=pfx + 'bottleneck', relu=False)
        return out, [d1, d2, d3, d4, d5]

    def unet(self, inp):
        '''Predict per-pixel coefficient vector given the input'''
        self.imsp = tf.shape(inp)

        out, _ = self.encode(inp)
        return tf.reduce_sum(out)

    def forward(self, inp, flash, denoised):
        llf_alpha = self.unet(inp) * (self.levels - 1)

        return self.llf(flash, denoised, llf_alpha)
    