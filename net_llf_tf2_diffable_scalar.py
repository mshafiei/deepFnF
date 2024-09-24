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
        

    def alpha_map(self):
        wnm = 'alpha_weight'
        if wnm in self.weights.keys():
            llf_alpha = self.weights[wnm]
        else:
            # llf_alpha = tf.Variable(tf.random.uniform(
            #     [1], minval=0, maxval=1, dtype=tf.float32))
            llf_alpha = tf.Variable(0.8)
            self.weights[wnm] = llf_alpha
        return llf_alpha

    def forward(self, inp, flash, denoised):
        llf_alpha = tf.nn.sigmoid(self.alpha_map()) * (self.levels - 1)

        return self.llf(flash, denoised, llf_alpha)
    