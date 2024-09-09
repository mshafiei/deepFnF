from collections import OrderedDict
from guided_local_laplacian_color import guided_local_laplacian_color
from net import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time

class NetNoChange():
    def __init__(self,input_type='flash'):
        self.input_type = input_type

    def forward(self, inp, alpha):
        if(self.input_type == 'flash'):
            return inp[:, :, :, 3:6] * alpha / ut.FLASH_STRENGTH
        else:
            return inp[:, :, :, :3]