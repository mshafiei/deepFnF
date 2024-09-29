from collections import OrderedDict
from net import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time
from net_llf_tf2_local_alpha_diffable_v2 import Net as NetAlpha

class Net(NetAlpha):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def predict_alpha(self, input):
        wnm = 'alpha_weight'
        if wnm in self.weights.keys():
            llf_alpha = self.weights[wnm]
        else:
            # llf_alpha = tf.Variable(tf.random.uniform(
            #     [self.alpha_width, self.alpha_height], minval=0, maxval=1, dtype=tf.float32))
            w = np.zeros((self.alpha_width,self.alpha_height),dtype=np.float32)
            w[0,0], w[0,1], w[0,2], w[0,3] = 0.2, 0.3, 0.4, 0.5
            w[1,0], w[1,1], w[1,2], w[1,3] = 0.6, 0.7, 0.8, 0.9
            w[2,0], w[2,1], w[2,2], w[2,3] = 0.0, 0.1, 0.2, 0.3
            w[3,0], w[3,1], w[3,2], w[3,3] = 0.4, 0.5, 0.6, 0.7
            llf_alpha = tf.Variable(tf.convert_to_tensor(w))
            self.weights[wnm] = llf_alpha
        return llf_alpha

    def forward(self, inp, flash, denoised):
        llf_alpha = self.predict_alpha(inp) * (self.levels - 1)

        return self.llf(flash, denoised, llf_alpha)
    