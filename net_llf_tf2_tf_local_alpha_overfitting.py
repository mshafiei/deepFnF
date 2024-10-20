from collections import OrderedDict
from net import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time
from net_llf_tf2_local_alpha_diffable_v2 import Net as NetAlpha
from gllf import gllf, _resize

class Net(NetAlpha):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.llf = lambda input, guide, alpha_: gllf(input, guide, self.levels, self.levels, alpha_, IMSZ=self.IMSZ, beta=self.beta, sigma=1)

    def predict_llf_inputs(self, input):
        wnm = 'alpha_weight'
        if wnm in self.weights.keys():
            llf_alpha = self.weights[wnm]
        else:
            llf_alpha = tf.Variable(tf.random.uniform(
                [self.alpha_width, self.alpha_height, 3], minval=0, maxval=0.1, dtype=tf.float32))
            self.weights[wnm] = llf_alpha
        
        wnm = 'input_weight'
        if wnm in self.weights.keys():
            llf_input = self.weights[wnm]
        else:
            llf_input = tf.Variable(tf.random.uniform(
                [self.IMSZ, self.IMSZ, 3], minval=0, maxval=1, dtype=tf.float32))
            self.weights[wnm] = llf_input

        wnm = 'flash_weight'
        if wnm in self.weights.keys():
            llf_flash = self.weights[wnm]
        else:
            llf_flash = tf.Variable(tf.random.uniform(
                [self.IMSZ, self.IMSZ, 3], minval=0, maxval=1, dtype=tf.float32))
            self.weights[wnm] = llf_flash
        
        return llf_input, llf_flash, llf_alpha

    def forward(self, inp, flash, denoised):
        llf_input, llf_flash, llf_alpha = self.predict_llf_inputs(inp)
        h, w = llf_input.shape[0:2]
        ah, aw = llf_alpha.shape[0:2]
        if(h > ah):
            llf_alpha = _resize(llf_alpha[None,...], (w, h))
        else:
            llf_alpha = llf_alpha[...,None]
        return self.llf(llf_input[None,...], llf_flash[None,...], llf_alpha), llf_input, llf_flash, llf_alpha
    