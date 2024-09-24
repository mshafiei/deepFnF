from collections import OrderedDict
from guided_local_laplacian_color_local_alpha_Mullapudi2016 import guided_local_laplacian_color_local_alpha_Mullapudi2016 as guided_local_laplacian_color
from gllf_color_local_alpha_grad_sparse_Mullapudi2016 import gllf_color_local_alpha_grad_sparse_Mullapudi2016 as guided_local_laplacian_color_grad
# from guided_local_laplacian_color_local_alpha_grad_Mullapudi2016 import guided_local_laplacian_color_local_alpha_grad_Mullapudi2016 as guided_local_laplacian_color_grad
from net import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time
import cv2

def derivative_tensor_size(ah, aw, imgsz):
    ratio_x =  imgsz // aw
    ratio_y =  imgsz // ah
    offset_x =  ratio_x // 2
    offset_y =  ratio_y // 2
    out_h, out_w = ratio_x + offset_x * 2, ratio_y + offset_y * 2
    return out_h, out_w, ratio_x, ratio_y, offset_x, offset_y

@tf.numpy_function(Tout=tf.float32)
def call_llf(flash, denoised, alpha, levels, beta, aw, ah, h, w):
    flash_np = flash
    denoised_np = denoised
    llf_out = np.empty([3, h, w], dtype=np.float32)
    # intensity_max = max(flash_np.max(), denoised_np.max())
    # intensity_min = min(flash_np.min(), denoised_np.min())
    # denoised_np = (denoised_np - intensity_min) / (intensity_max - intensity_min)
    # flash_np = (flash_np - intensity_min) / (intensity_max - intensity_min)
    start = time.time_ns()
    guided_local_laplacian_color(flash_np, denoised_np, levels, alpha, beta, aw, ah, w, h, llf_out)
    # tf.print('llf_time ', (time.time_ns() - start)/1000000)
    # llf_out *= (intensity_max - intensity_min) + intensity_min
    return llf_out

@tf.numpy_function(Tout=tf.float32)
def call_dllf(flash, denoised, alpha, levels, beta, ah, aw, h, w, ox, oy, dh, dw):
    flash_np = flash
    denoised_np = denoised
    dllf_out = np.empty([ah, aw, 3, dh, dw], dtype=np.float32)
    # dllf_out = np.empty([ah, aw, 3, h, w], dtype=np.float32)
    # intensity_max = max(flash_np.max(), denoised_np.max())
    # intensity_min = min(flash_np.min(), denoised_np.min())
    # denoised_np = (denoised_np - intensity_min) / (intensity_max - intensity_min)
    # flash_np = (flash_np - intensity_min) / (intensity_max - intensity_min)
    start = time.time_ns()
    guided_local_laplacian_color_grad(flash_np, denoised_np, levels, alpha / (levels - 1), beta, aw, ah, w, h, ox, oy, dllf_out)
    # guided_local_laplacian_color_grad(flash_np, denoised_np, levels, alpha / (levels - 1), beta, aw, ah, w, h, dllf_out)
    # tf.print('dllf_time ', (time.time_ns() - start)/1000000)
    # deriv = tf.convert_to_tensor(deriv.transpose(1,2,0))[None,...]
    # dllf_out *= (intensity_max - intensity_min) + intensity_min
    return dllf_out

class Net(OriginalNet):
    def __init__(self, alpha_width=8, alpha_height=8, llf_beta=1, llf_levels=2, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1, lmbda=1,IMSZ=448):
        self.weights = {}
        self.activations = OrderedDict()
        self.num_basis = num_basis
        self.ksz = ksz
        self.burst_length = burst_length
        self.channels_count_factor = channels_count_factor
        self.channel_count = lambda x: max(1, int(x * self.channels_count_factor))
        self.lmbda = lmbda
        self.IMSZ = IMSZ
        self.alpha_width = alpha_width
        self.alpha_height = alpha_height
        self.beta = llf_beta
        self.levels = llf_levels
    
        self.llf_derivative_h, self.llf_derivative_w, _, _, self.offset_x, self.offset_y = derivative_tensor_size(self.alpha_height, self.alpha_width, self.IMSZ)
        self.llf_output = np.empty([3, self.IMSZ, self.IMSZ], dtype=np.float32)
        self.dllf_output = np.empty([self.alpha_height, self.alpha_width, 3, self.llf_derivative_h, self.llf_derivative_w], dtype=np.float32)

        

    def deepfnfDenoising(self, inp, alpha):
        denoised = super().forward(inp)
        flash = inp[:, :, :, 3:6] * alpha / ut.FLASH_STRENGTH
        return denoised, flash
    
    @tf.custom_gradient
    def llf(self, flash, denoised, alpha):
        
        llf_output = call_llf(tf.transpose(flash[0,...],(2,0,1)),
            tf.transpose(denoised[0,...],(2,0,1)),
            alpha, self.levels, self.beta, self.alpha_width, self.alpha_height, self.IMSZ, self.IMSZ)

        def grad_fn(upstream):
            # assert len(upstream) == 3
            dllf_output = call_dllf(tf.transpose(flash[0,...],(2,0,1)),
            tf.transpose(denoised[0,...],(2,0,1)),
            alpha, self.levels, self.beta, self.alpha_width, self.alpha_height, self.IMSZ, self.IMSZ, self.offset_x, self.offset_y, self.llf_derivative_h, self.llf_derivative_w)
            ah, aw, block_h, block_w = self.alpha_height, self.alpha_width, self.llf_derivative_h, self.llf_derivative_w
            blocks = tfu.split_image_into_blocks(upstream, self.IMSZ, self.IMSZ, ah, aw, block_h, block_w, block_h//4, block_w//4)
            blocks = tf.transpose(blocks,(0,1,4,2,3))
            # for i in range(aw):
            #     for j in range(ah):
            #         cv2.imwrite('./deriv_out/upstream_%i_%i.exr'%(i, j), np.abs(blocks[i,j,:,:,:].numpy().transpose(1,2,0)))
            #         cv2.imwrite('./deriv_out/dllf_%i_%i.exr'%(i, j), np.abs(dllf_output[i,j,:,:,:].numpy().transpose(1,2,0)))
            alpha_grad = tf.reduce_sum(blocks * dllf_output,axis=(2,3,4))
            return (upstream, upstream, alpha_grad)
        
        return tf.transpose(llf_output[None,...],(0,2,3,1)), grad_fn

    