from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu
import utils.utils as ut
import time
from net import Net as OriginalNet

class Net(OriginalNet):
    def __init__(self, x0, k, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1, lmbda=1,IMSZ=448, laplacian_levels=5):
        super().__init__(num_basis=num_basis, ksz=ksz, burst_length=burst_length, channels_count_factor=channels_count_factor)
        self.weights = {}
        self.activations = OrderedDict()
        self.num_basis = num_basis
        self.ksz = ksz
        self.burst_length = burst_length
        self.channels_count_factor = channels_count_factor
        self.channel_count = lambda x: max(1, int(x * self.channels_count_factor))
        self.lmbda = lmbda
        self.IMSZ = IMSZ
        self.x0 = x0
        self.k = k
        self.laplacian_levels = laplacian_levels

    def deepfnfDenoising(self, inp, alpha):
        denoised = super().forward(inp)
        flash = inp[:, :, :, 3:6] * alpha / ut.FLASH_STRENGTH
        return denoised, flash

    def pyramid(self, inp, alpha):
        denoised, flash = self.deepfnfDenoising(inp, alpha)
        if(self.x0 == 0 and self.k == 0):
            return tfu.interpolated_laplacian(flash,flash,self.x0,self.k,self.laplacian_levels)
        else:
            return tfu.interpolated_laplacian(flash,denoised,self.x0,self.k,self.laplacian_levels)
        
    def forward(self, inp, alpha):
        denoised, flash = self.deepfnfDenoising(inp, alpha)
        if(self.x0 == 0 and self.k == 0):
            combined = tfu.combineFNFInLaplacian(flash,flash,self.x0,self.k,self.laplacian_levels)    
        else:
            combined = tfu.combineFNFInLaplacian(flash,denoised,self.x0,self.k,self.laplacian_levels)
        
        return combined