from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu


class Net:
    def __init__(self):
        pass

    def forward(self, inp, alpha):
        flash = inp[:, :, :, 3:6] * alpha

        return flash