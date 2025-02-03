from arguments_deepfnf import parse_arguments_deepfnf
parser = parse_arguments_deepfnf()
opts = parser.parse_args()
from gllf.gllf import gllf_diffable_1d
from gllf.gllf_layer import gllf_layer_radial
from gllf_halide import halide_neural_gllf
import imageio
import numpy as np
import tensorflow as tf
import cvgutils.Viz as viz
import cv2
import os
import timeit
from easydict import EasyDict as edict
tf.config.run_functions_eagerly(True)
logr = viz.logger(opts=opts)

def setup(testname,crop=False):
    input_fn = '/home/mohammad/Downloads/fft_combine/blurred.png'
    guide_fn = '/home/mohammad/Downloads/fft_combine/flash.png'

    max_levels = 4
    max_discrete_levels = 4
    alpha = 1
    beta  = 1.0
    sigma = 1.0
    IMSZ = 448
    im_i = imageio.imread(input_fn).astype(np.float32) / 255.0
    im_g = imageio.imread(guide_fn).astype(np.float32) / 255.0
    im_i[100:110,100:110,:] = -0.05
    alpha_h = tf.Variable(0.3)#np.array(tf.random.uniform(im_i.shape,0,1))
    alpha_i = tf.Variable(0.0)#np.array(tf.random.uniform(im_i.shape,0,1))
    if(crop):
        IMSZ = 32
        im_i = tf.convert_to_tensor(im_i[None,128:128+IMSZ,128:128+IMSZ,:])
        im_g = tf.convert_to_tensor(im_g[None,128:128+IMSZ,128:128+IMSZ,:])
    else:
        # IMSZ = im_i.shape[1]
        im_i = cv2.resize(im_i, (IMSZ, IMSZ))
        im_g = cv2.resize(im_g, (IMSZ, IMSZ))
        # alpha_i = cv2.resize(alpha_i, (IMSZ, IMSZ))
        # alpha_h = cv2.resize(alpha_h, (IMSZ, IMSZ))
        im_i = tf.convert_to_tensor(im_i[None,...])
        im_g = tf.convert_to_tensor(im_g[None,...])
        # alpha_i = tf.convert_to_tensor(alpha_i[None,...])
        # alpha_h = tf.convert_to_tensor(alpha_h[None,...])
    if(testname == 'test_compare_grad_slice_1d'):
        fn_grad = 'grad_test_compare_grad_slice_1d.npy'
        fn_fd = 'fd_test_compare_grad_slice_1d.npy'
    elif(testname == 'images_to_lookup_1d_grad'):
        fn_grad = 'grad_images_to_lookup_1d_grad.npy'
        fn_fd = 'fd_images_to_lookup_1d_grad.npy'
    else:
        print('unknown test')
        exit(0)
    if(os.path.exists(fn_grad)):
        grad_diff = np.load(fn_grad)
    else:
        grad_diff = None

    if(os.path.exists(fn_fd)):
        grad_fd = np.load(fn_fd)
    else:
        grad_fd = None
    return edict(im_i=im_i, im_g=im_g, max_levels=max_levels, max_discrete_levels=max_discrete_levels, alpha_h=alpha_h, alpha_i=alpha_i, beta=beta, sigma=sigma, IMSZ=IMSZ, grad_diff=grad_diff, grad_fd=grad_fd, fn_grad=fn_grad, fn_fd=fn_fd)

def tear_down():
    pass

def test_compare_gllf_radial_basis(configs):
    imlist = [configs.im_i,configs.im_g]
    gllf_layer = gllf_layer_radial(configs.max_levels, configs.max_discrete_levels, img_ct=len(imlist), basis_ct=1, llf_remap_function='gaussian_1d', yuv_gllf=False, alphas=[1], betas=[0], sigmas=[1], thresholds=None, downsample_ct=3, IMSZ=configs.IMSZ, unet_output_size=3,rbf_weights_ct=1)
    gllf_layer.scalar_alphas_net(None)
    gllf_out = gllf_layer.gllf_diffable_1d(imlist)
    
    
    fn_result_halide = halide_neural_gllf(configs.im_i[0],configs.im_g[0], gllf_layer.image_weights, gllf_layer.range_weights, gllf_layer.w_i, gllf_layer.sigma_i, configs.max_levels, configs.alpha_h, configs.beta, configs.sigma)
    fn_result_halide = fn_result_halide[0]

    visual = gllf_layer.visualize(imlist)
    g = {'gllf_in':configs.im_i[0], 'gllf_out':gllf_out[0], 'halide':fn_result_halide}
    lbl = {'gllf_in':'gllf_in','gllf_out':'gllf_out','halide':'halide'}
    for k,v in g.items():
        g[k] = cv2.resize(v.numpy(),(448,448))
    for v in visual:
        g.update({v.keys:v.image})
        lbl.update({v.key:v.label})
    
    logr.addImage(g, lbl, 'train')
    logr.takeStep()

def test_compare_gllf_radial_basis_vs_halide(configs):
    imgs = tf.stack([configs.im_i, configs.im_g],axis=0)
    # alphas = tf.stack([configs.alpha_i, configs.alpha_h],axis=0)
    alphas = [[configs.alpha_i,configs.alpha_i,configs.alpha_i,configs.alpha_i],[configs.alpha_h,configs.alpha_h,configs.alpha_h,configs.alpha_h]]
    betas = [[configs.beta,configs.beta,configs.beta,configs.beta],[0,0,0,0]]
    sigmas = [[configs.sigma,configs.sigma,configs.sigma,configs.sigma],[0,0,0,0]]
    #reduce gllf_diffable_1d to current version of halide_neural_gllf
    #change gllf_diffable_1d to a random neural variable and match halide_neural_gllf with that
    fn_result_tf = gllf_diffable_1d(imgs, alphas, configs.max_levels, configs.max_discrete_levels, betas=betas, sigmas=sigmas, min_intensity=0.0, max_intensity=1.0, IMSZ=configs.IMSZ)
    fn_result_tf = fn_result_tf[0]
    fn_result_halide = halide_neural_gllf(imgs[0,0], imgs[1,0], configs.max_levels, configs.alpha_h, configs.beta, configs.sigma)
    fn_result_halide = fn_result_halide[0]
    print('hi')
    # halide_neural_gllf(denoised_np, flash_np, levels, alpha, beta,sigma)
    # 0. run normal halide on images
    # 1. use 1 basis for all intensities w/ constant alpha and std
    # 2. make alpha, std per layer
    # 3. interpolate laplacian layers

    # execute gllf_layer w/ given images
    #
    
    # imgs = tf.stack([configs.im_i, configs.im_g],axis=0)
    # alphas = tf.stack([configs.alpha_i, configs.alpha_h],axis=0)
    
    # fn = lambda diffable_imgs: gllf_diffable_1d(diffable_imgs, alphas, configs.max_levels, configs.max_discrete_levels, betas=[configs.beta,0], sigmas=[configs.sigma,0], min_intensity=0.0, max_intensity=1.0, IMSZ=configs.IMSZ)
    
    
    # fn_result_1d = fn(imgs)[0]
    # fn_result = gllf_diffable_2d(imgs[0], imgs[1], configs.max_levels, configs.max_discrete_levels, configs.alpha_i, configs.alpha_h, beta=configs.beta, sigma=configs.sigma, IMSZ=configs.IMSZ)[0]
    # # fn_result = fn(imgs)[0]
    g = {'fn_result_halide':fn_result_halide, 'fn_result_tf':fn_result_tf}
    lbl = {'fn_result_halide':'fn_result_halide','fn_result_tf':'fn_result_tf'}
    for k,v in g.items():
        g[k] = cv2.resize(v.numpy(),(448,448))
    logr.addImage(g, lbl, 'train')
    logr.takeStep()

    # assert True

configs = setup('test_compare_grad_slice_1d',crop=False)
test_compare_gllf_radial_basis(configs)
# test_compare_gllf_radial_basis_vs_halide(configs)