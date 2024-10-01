from collections import OrderedDict
from net import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time
from net import Net as OriginalNet

class Net(OriginalNet):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def unet(self, inp):
        '''Predict per-pixel coefficient vector given the input'''
        self.imsp = tf.shape(inp)

        out, skips = self.encode(inp)
        out = self.decode(out, skips)
        out = self.conv('output', out, 3, relu=False)
        return out

    def forward(self, inp, flash, denoised):
        return self.unet(inp)
    