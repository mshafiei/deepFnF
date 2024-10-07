from arguments_deepfnf import parse_arguments_deepfnf
parser = parse_arguments_deepfnf()
opts = parser.parse_args()
import numpy as np
import cv2
import halide as hl
import halide.imageio as io
import os
import cv2
# from net_llf_tf2_diffable import call_llf, call_dllf
from net_llf_tf2_local_alpha_diffable import call_llf, call_dllf
import gllf_network_utils as net_utils
import cvgutils.Viz as Viz
import tensorflow as tf
import utils.tf_utils as tfu
from utils.dataset import Dataset
import utils.utils as ut
tf.config.run_functions_eagerly(True)

TLIST = opts.TLIST
VPATH = opts.VPATH
BSZ = 1
IMSZ = 448
LR = 1e-4
DROP = (1.1e6, 1.25e6) # Learning rate drop

MAXITER = 1.5e6
displacement = opts.displacement
VALFREQ = opts.val_freq
SAVEFREQ = opts.save_freq
store_image = True

logger = Viz.logger(opts,opts.__dict__)
model, deepfnf_model = net_utils.CreateNetwork(opts)
dataset = Dataset(TLIST, VPATH, bsz=BSZ, psz=IMSZ, ngpus=opts.ngpus, nthreads=4 * opts.ngpus,jitter=opts.displacement,min_scale=opts.min_scale,max_scale=opts.max_scale,theta=opts.max_rotate)
example = logger.load_pickle('example.pickle')

with tf.device('/gpu:0'):
    niter = 0

    deepfnf_params = logger.load_params(opts.deepfnf_train_path)
    if(deepfnf_params != None):
        deepfnf_model.weights = deepfnf_params['params']
    else:
        print('cannot load deepfnf parameters ', opts.deepfnf_train_path)
        exit(0)

@tf.function
def prepare_input(example):
    alpha = example['alpha']
    dimmed_ambient, _ = tfu.dim_image(
        example['ambient'], alpha=alpha)
    dimmed_warped_ambient, _ = tfu.dim_image(
        example['warped_ambient'], alpha=alpha)
    # Make the flash brighter by increasing the brightness of the
    # flash-only image.
    flash = example['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
    warped_flash = example['warped_flash_only'] * \
        ut.FLASH_STRENGTH + dimmed_warped_ambient
    sig_read = example['sig_read']
    sig_shot = example['sig_shot']
    noisy_ambient, _, _ = tfu.add_read_shot_noise(
        dimmed_ambient, sig_read=sig_read, sig_shot=sig_shot)
    noisy_flash, _, _ = tfu.add_read_shot_noise(
        warped_flash, sig_read=sig_read, sig_shot=sig_shot)
    noisy = tf.concat([noisy_ambient, noisy_flash], axis=-1)
    noise_std = tfu.estimate_std(noisy, sig_read, sig_shot)
    net_input = tf.concat([noisy, noise_std], axis=-1)
    return net_input, alpha, noisy_flash, noisy_ambient

@tf.function
def gradient_validation(net_input, alpha, noisy_flash, noisy_ambient):
    eps = 0.01
    denoise = deepfnf_model.forward(net_input)
    
    deepfnf_scaled = tfu.camera_to_rgb(
        denoise / alpha, example['color_matrix'], example['adapt_matrix'])
    
    noisy_flash_scaled = tfu.camera_to_rgb(
        noisy_flash, example['color_matrix'], example['adapt_matrix'])
    
    ambient_scaled= tfu.camera_to_rgb(
        example['ambient'],
        example['color_matrix'], example['adapt_matrix'])
    
    net_ft_input = tf.concat((net_input, denoise), axis=-1)
    ah, aw = model.alpha_height, model.alpha_width
    _, h, w, _ = net_ft_input.shape
    ratio_x =  w // aw
    ratio_y =  h // ah
    block_h, block_w = ratio_y * 2, ratio_x * 2
    oy, ox = ratio_y//2, ratio_x//2
    with tf.GradientTape() as tape:
        gllf_scaled = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
        # Loss
        # l2_loss = tfu.l2_loss(gllf_scaled, ambient_scaled)
        # loss = l2_loss
    gradients_loss = tape.gradient(gllf_scaled, model.weights.values())
    # with tf.GradientTape(persistent=True) as tape:
    #     gllf_scaled = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
        
    # gradients_image = tape.gradient(gllf_scaled, model.weights.values())

    #compare images
    alpha_map = model.predict_alpha(None)
    
    for i in range(ah):
        for j in range(aw):
            dllf = call_dllf(tf.transpose(noisy_flash_scaled[0,...],(2,0,1)),
                    tf.transpose(deepfnf_scaled[0,...],(2,0,1)),
                    alpha_map, opts.llf_levels, opts.llf_beta, ah, aw, h, w, ox, oy, block_h, block_w)

            llf_0 = call_llf(tf.transpose(noisy_flash_scaled[0,...],(2,0,1)),
                    tf.transpose(deepfnf_scaled[0,...],(2,0,1)),
                    alpha_map, opts.llf_levels, opts.llf_beta, ah, aw, h, w)
            alpha_map[0,0].assign(alpha_map[0,0] - eps)
            llf_n = call_llf(tf.transpose(noisy_flash_scaled[0,...],(2,0,1)),
                    tf.transpose(deepfnf_scaled[0,...],(2,0,1)),
                    alpha_map, opts.llf_levels, opts.llf_beta, ah, aw, h, w)
            fd_llf = (llf_0 - llf_n) / (eps)

            # source_startx = ox if j == 0 else 0
            # source_endx = block_w - ox if j == aw-1 else w
            # source_starty = oy if i == 0 else 0
            # source_endy = block_h - oy if i == ah-1 else h
            
            # target_startx = 0 if j == 0 else j * ratio_x - ox
            # target_endx = w if j == aw-1 else (j+1) * ratio_x + ox
            # target_starty = 0 if i == 0 else i * ratio_y - oy
            # target_endy = h if i == ah-1 else (i+1) * ratio_y + oy
            
            # output = np.zeros_like(fd_llf, dtype=np.float32)
            # output[:,target_starty:target_endy, target_startx:target_endx] = dllf[i,j,:,source_starty:source_endy, source_startx:source_endx]
            output = dllf
            cv2.imwrite('./deriv_out/dllf_%i_%i.exr'%(i, j), output[0,0,...].numpy().transpose(1,2,0))
            cv2.imwrite('./deriv_out/fd_%i_%i.exr'%(i, j), fd_llf.numpy().transpose(1,2,0))
        cv2.imwrite('./deriv_out/flash_%i_%i.exr'%(i, j), noisy_flash_scaled[0].numpy())
        cv2.imwrite('./deriv_out/denoised_%i_%i.exr'%(i, j), deepfnf_scaled[0].numpy())
            # assert tf.reduce_mean(tf.square(dllf - fd_llf)/tf.square(fd_llf)) < 1e-4
    print('hi')
    fd = {}
    fd_grad_diff = {}
    for k_i, k in enumerate(model.weights.keys()):
        orig_shape = model.weights[k].shape
        flat = tf.Variable(tf.reshape(model.weights[k],-1))
        fd_tensor = tf.Variable(tf.zeros_like(flat))
        grad_flat = tf.Variable(tf.reshape(gradients_loss[k_i],-1))
        for i in range(flat.shape[0]):
            w_val = flat[i]

            # values = tf.identity(model.weights[k])
            flat[i].assign(w_val - eps)
            model.weights[k] = tf.reshape(flat,orig_shape)
            gllf_scaled_n = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
            l2_loss = tfu.l2_loss(gllf_scaled_n, ambient_scaled)
            # gradient_loss = tfu.gradient_loss(gllf_scaled, ambient_scaled)
            loss_n = l2_loss #+ gradient_loss
            
            flat[i].assign(w_val)
            model.weights[k] = tf.reshape(flat,orig_shape)
            gllf_scaled_p = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
            l2_loss = tfu.l2_loss(gllf_scaled_p, ambient_scaled)
            # gradient_loss = tfu.gradient_loss(gllf_scaled, ambient_scaled)
            loss_p = l2_loss #+ gradient_loss

            fd_tensor[i].assign((loss_p - loss_n) / (eps))
            flat[i].assign(w_val)
            model.weights[k] = tf.reshape(flat, orig_shape)
            # if(store_image):
            #     cv2.imwrite('gradients_image.exr', tf.abs(gradients_image[0][0]).numpy())
            #     cv2.imwrite('gllf_scaled_p.exr', tf.abs(gllf_scaled_p[0]).numpy())
            #     cv2.imwrite('gllf_scaled_n.exr', tf.abs(gllf_scaled_n[0]).numpy())
            print(i, 'th gradient: ', grad_flat[i].numpy(), ' fd ', fd_tensor[i].numpy(), ' difference ',tf.abs(grad_flat[i] - fd_tensor[i]).numpy())
            # assert tf.abs(grad_flat[i] - fd_tensor[i]) < 1e-4

        fd[k] = tf.reshape(fd_tensor, orig_shape)
        # fd_grad_diff[k] = gradients[k] - fd[k]
    # print('hi')
    
net_input, alpha, noisy_flash, noisy_ambient = prepare_input(example)
gradient_validation(net_input, alpha, noisy_flash, noisy_ambient)