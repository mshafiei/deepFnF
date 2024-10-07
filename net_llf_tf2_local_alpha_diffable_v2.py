from collections import OrderedDict
from guided_local_laplacian_color_local_alpha_Mullapudi2016 import guided_local_laplacian_color_local_alpha_Mullapudi2016 as guided_local_laplacian_color
from gllf_color_local_alpha_grad_sparse_v2_Mullapudi2016 import gllf_color_local_alpha_grad_sparse_v2_Mullapudi2016 as guided_local_laplacian_color_grad
# from guided_local_laplacian_color_local_alpha_grad_Mullapudi2016 import guided_local_laplacian_color_local_alpha_grad_Mullapudi2016 as guided_local_laplacian_color_grad
from net import Net as OriginalNet

import numpy as np
import tensorflow as tf
import utils.tf_utils as tfu
import utils.utils as ut
import time
import cv2

def prepare_params(h, w, ah, aw):
    #     margin block  margin
    #____|      |      |      |
    #____|      |      |      |
    
    core_w = max(w // aw, 8)
    core_h = max(h // ah, 8)
    block_w =  max(core_w + core_w, 4)*2
    block_h =  max(core_h + core_h, 4)*2
    stride_y = h//ah
    stride_x = w//aw
    margin_y = (block_w - stride_x) // 2
    margin_x = (block_h - stride_y) // 2

    return block_h, block_w, stride_y, stride_x, margin_y, margin_x

def derivative_tensor_size(ah, aw, imgsz):
    ratio_x =  imgsz // aw
    ratio_y =  imgsz // ah
    offset_x =  ratio_x // 2
    offset_y =  ratio_y // 2
    out_h, out_w = ratio_x + offset_x * 2, ratio_y + offset_y * 2
    return out_h, out_w, ratio_x, ratio_y, offset_x, offset_y

@tf.numpy_function(Tout=tf.float32)
def call_llf(flash, denoised, alpha, levels, beta, sigma):
    flash_np = flash
    denoised_np = denoised
    ah, aw = alpha.shape
    h, w = denoised_np.shape[1:]
    llf_out = np.empty([3, h, w], dtype=np.float32)
    guided_local_laplacian_color(flash_np, denoised_np, levels, alpha, beta, sigma, aw, ah, w, h, llf_out)
    return llf_out

@tf.numpy_function(Tout=tf.float32)
def call_dllf(flash, denoised, alpha, levels, beta, block_h, block_w, stride_y, stride_x, margin_y, margin_x):
    flash_np = flash
    denoised_np = denoised
    ah, aw = alpha.shape
    h, w = denoised.shape[-2:]
    dllf_out = np.empty([ah, aw, 3, block_h, block_w], dtype=np.float32)
    guided_local_laplacian_color_grad(flash_np, denoised_np, levels, alpha / (levels - 1), beta, aw, ah, w, h, block_w, block_h, stride_y, stride_x, margin_y, margin_x, dllf_out)
    return dllf_out

class Net(OriginalNet):
    def __init__(self, llf_sigma=1, alpha_width=8, alpha_height=8, llf_beta=1, llf_levels=2, num_basis=90, ksz=15, burst_length=2, channels_count_factor=1, lmbda=1,IMSZ=448):
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
        self.sigma = llf_sigma

        IMSZ = int(2**np.ceil(np.log2(self.IMSZ)))
        alpha_height, alpha_width = int(2**np.ceil(np.log2(self.alpha_height))), int(2**np.ceil(np.log2(self.alpha_width)))

        self.image_res_h, self.image_res_w = (IMSZ - self.IMSZ)//2, (IMSZ - self.IMSZ)//2
        self.alpha_res_h, self.alpha_res_w = (alpha_height - self.alpha_height)//2, (alpha_width - self.alpha_width)//2
        self.new_alpha_height, self.new_alpha_width = alpha_height, alpha_width
        self.new_IMSZ = IMSZ

        # self.llf_derivative_h, self.llf_derivative_w, _, _, self.offset_x, self.offset_y = derivative_tensor_size(self.alpha_height, self.alpha_width, self.IMSZ)
        self.block_h, self.block_w, self.stride_y, self.stride_x, self.margin_y, self.margin_x = prepare_params(self.new_IMSZ, self.new_IMSZ, self.new_alpha_height, self.new_alpha_width)
        self.llf_output = np.empty([3, self.new_IMSZ, self.new_IMSZ], dtype=np.float32)
        self.dllf_output = np.empty([self.new_alpha_height, self.new_alpha_width, 3, self.block_h, self.block_w], dtype=np.float32)

        

    def deepfnfDenoising(self, inp, alpha):
        denoised = super().forward(inp)
        flash = inp[:, :, :, 3:6] * alpha / ut.FLASH_STRENGTH
        return denoised, flash
    
    @tf.custom_gradient
    def llf(self, flash, denoised, alpha):
        #pad flash, denoised, alpha
        h, w = flash.shape[1:3]
        ah, aw = alpha.shape

        # image_res_h, image_res_w = (self.IMSZ-h)//2, (self.IMSZ - w)//2
        # alpha_res_h, alpha_res_w = (self.alpha_height-ah)//2, (self.alpha_width - aw)//2
        image_padding = [[0,0],[self.image_res_h,self.image_res_h],[self.image_res_w,self.image_res_w],[0,0]]
        alpha_padding = [[self.alpha_res_h,self.alpha_res_h],[self.alpha_res_w,self.alpha_res_w]]
        flash = tf.pad(flash, image_padding, "SYMMETRIC")
        denoised = tf.pad(denoised, image_padding, "SYMMETRIC")
        alpha = tf.pad(alpha, alpha_padding, "SYMMETRIC")

        llf_output = call_llf(tf.transpose(flash[0,...],(2,0,1)),
            tf.transpose(denoised[0,...],(2,0,1)),
            alpha, self.levels, self.beta, self.sigma)

        def grad_fn(upstream):
            #pad upstream
            # assert len(upstream) == 3
            upstream_padded = tf.pad(upstream, image_padding, "SYMMETRIC")
            dllf_output = call_dllf(tf.transpose(flash[0,...],(2,0,1)),
            tf.transpose(denoised[0,...],(2,0,1)),
            alpha, self.levels, self.beta, self.block_h, self.block_w, self.stride_y, self.stride_x, self.margin_y, self.margin_x)
            # ah, aw, block_h, block_w = alpha_res_h, alpha_res_w, self.llf_derivative_h, self.llf_derivative_w
            blocks = tfu.split_image_into_blocks_v2(upstream_padded, self.new_IMSZ, self.new_IMSZ, self.stride_y, self.stride_x, self.new_alpha_height, self.new_alpha_width, self.block_h, self.block_w, self.stride_y, self.stride_x)
            blocks = tf.transpose(blocks,(0,1,4,2,3))
            # for i in range(aw):
            #     for j in range(ah):
            #         cv2.imwrite('./deriv_out/upstream_%i_%i.exr'%(i, j), np.abs(blocks[i,j,:,:,:].numpy().transpose(1,2,0)))
            #         cv2.imwrite('./deriv_out/dllf_%i_%i.exr'%(i, j), np.abs(dllf_output[i,j,:,:,:].numpy().transpose(1,2,0)))
            alpha_grad = tf.reduce_sum(blocks * dllf_output,axis=(2,3,4))
            if(self.alpha_res_h == 0 or self.alpha_res_w == 0):
                return (upstream, upstream, alpha_grad)
            else:
                return (upstream, upstream, alpha_grad[self.alpha_res_h:-self.alpha_res_h,self.alpha_res_w:-self.alpha_res_w])
        llf_output = llf_output[:,self.image_res_h:-self.image_res_h,self.image_res_w:-self.image_res_w]
        return tf.transpose(llf_output[None,...],(0,2,3,1)), grad_fn

    