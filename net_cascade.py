from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu

from net import Net

class net_cascade:
    def __init__(self, num_basis=90, ksz=15, burst_length=2):
        self.weights = {}
        self.activations = OrderedDict()
        self.num_basis = num_basis
        self.ksz = ksz
        self.burst_length = burst_length
        self.net = Net(num_basis, ksz, burst_length)
    
    def forward(self, inp):
        return self.net(inp)