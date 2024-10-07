from arguments_deepfnf import parse_arguments_deepfnf
parser = parse_arguments_deepfnf()
opts = parser.parse_args()
import numpy as np
import cv2
import halide as hl
import halide
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
from gllf_color_local_alpha_grad_sparse_Mullapudi2016 import gllf_color_local_alpha_grad_sparse_Mullapudi2016 as guided_local_laplacian_grad_sparse_Mullapudi2016
from guided_local_laplacian_color_local_alpha_grad_Mullapudi2016 import guided_local_laplacian_color_local_alpha_grad_Mullapudi2016 as guided_local_laplacian_grad_dense_Mullapudi2016
from guided_local_laplacian_color_local_alpha_Mullapudi2016 import guided_local_laplacian_color_local_alpha_Mullapudi2016 as guided_local_laplacian_Mullapudi2016
tf.config.run_functions_eagerly(True)


def denoising(guide, input, levels, alpha, beta, output_path):
    timing_iterations = 10

    # input_buf_u8, guided_buf_u8, alpha_map, ah, aw, h, w = read_input()
    ah, aw, h, w = 4, 4, 448, 448
    input_buf_u8, guided_buf_u8, alpha_map = input, guide, alpha
    # guided_buf_u8 = np.ascontiguousarray(cv2.resize(guided_buf_u8.transpose(1,2,0),(448,448)).transpose(2,0,1))
    # input_buf_u8 = np.ascontiguousarray(cv2.resize(input_buf_u8.transpose(1,2,0),(448,448)).transpose(2,0,1))
    assert input_buf_u8.dtype == np.float32
    # Convert to uint16 in range [0..1]
    guided_buf = guided_buf_u8.astype(np.float32)
    input_buf = input_buf_u8.astype(np.float32)
    h = input_buf.shape[1]
    w = input_buf.shape[2]
    inner_block_w =  w // aw
    inner_block_h =  h // ah
    offset_x =  inner_block_w // 2
    offset_y =  inner_block_h // 2
    out_h, out_w = inner_block_w + offset_x * 2, inner_block_h + offset_y * 2
    output_buf = np.empty([ah, aw, 3, out_h, out_w], dtype=input_buf.dtype)
    output_buf_dense = np.empty([ah, aw, 3, h, w], dtype=input_buf.dtype)
    aux_buf_0 = np.empty([ah, aw, 3, out_h, out_w], dtype=input_buf.dtype)
    aux_buf_dense_0 = np.empty([ah, aw, 3, h, w], dtype=input_buf.dtype)
    aux_buf_1 = np.empty([ah, aw, 3, out_h, out_w], dtype=input_buf.dtype)
    aux_buf_dense_1 = np.empty([ah, aw, 3, h, w], dtype=input_buf.dtype)
    output_buf_n = np.empty([3, h, w], dtype=input_buf.dtype)
    output_buf_0 = np.empty([3, h, w], dtype=input_buf.dtype)

    eps = 0.01
    guided_local_laplacian_grad_sparse_Mullapudi2016(guided_buf, input_buf, levels, alpha_map / (levels - 1), beta, aw, ah, w, h, offset_x, offset_y, output_buf)

    output_buf_u8 = ((output_buf))
    guided_buf_u8 = (guided_buf_u8)
    input_buf_u8 = (input_buf_u8)
    alpha_map_u8 = (alpha_map/levels)

    guide_path = output_path.replace('rgb_out.png','guide.exr')
    input_path = output_path.replace('rgb_out.png','input.exr')
    alpha_path = output_path.replace('rgb_out.png','alpha.exr')
    dump_at_all = True
    if(not dump_at_all):
        return
    dump_small_image = False
    for i in range(ah):
        for j in range(aw):
            alpha_map_n = alpha_map.copy() / (levels - 1)
            alpha_map_n[i,j] = alpha_map_n[i,j] - eps
            guided_local_laplacian_grad_dense_Mullapudi2016(guided_buf, input_buf, levels, alpha_map / (levels - 1), beta, aw, ah, w, h, output_buf_dense)
            guided_local_laplacian_Mullapudi2016(guided_buf, input_buf, levels, alpha_map_n, beta, aw, ah, w, h, output_buf_n)
            guided_local_laplacian_Mullapudi2016(guided_buf, input_buf, levels, alpha_map / (levels - 1), beta, aw, ah, w, h, output_buf_0)
            output_buf_fd = (output_buf_0 - output_buf_n) / eps

            p = output_path.replace('rgb_out.png','rgb_out_%i_%i_sparse.exr' %(i, j))
            p_fd = output_path.replace('rgb_out.png','rgb_out_%i_%i_fd.exr' %(i, j))
            p_dense = output_path.replace('rgb_out.png','rgb_out_%i_%i_dense.exr' %(i, j))
            p_center = output_path.replace('rgb_out.png','rgb_out_%i_%i_center.exr' %(i, j))
            p_aux_dense_0 = output_path.replace('rgb_out.png','rgb_out_%i_%i_aux_dense_0.exr' %(i, j))
            p_aux_dense_1 = output_path.replace('rgb_out.png','rgb_out_%i_%i_aux_dense_1.exr' %(i, j))
            p_aux_sparse_0 = output_path.replace('rgb_out.png','rgb_out_%i_%i_aux_sparse_0.exr' %(i, j))
            p_aux_sparse_1 = output_path.replace('rgb_out.png','rgb_out_%i_%i_aux_sparse_1.exr' %(i, j))
            print("Saving to %s ..." % p)
            print("Saving to %s ..." % p_dense)
            print("Saving to %s ..." % p_aux_dense_0)
            print("Saving to %s ..." % p_aux_sparse_0)
            print("Saving to %s ..." % p_aux_dense_1)
            print("Saving to %s ..." % p_aux_sparse_1)
            if(dump_small_image):
                halide.imageio.imwrite(p, np.ascontiguousarray(output_buf_u8[i,j,:,:,:]))
            else:
                output = np.zeros_like(input_buf,dtype=np.float32)
                
                #specify crop size
                #find start and end x and y
                #clip start and end of target
                # clip start and end of source similarly

                # source_startx = offset_x if j == 0 else 0
                # source_endx = out_w - offset_x if j == aw-1 else out_w
                # source_starty = offset_y if i == 0 else 0
                # source_endy = out_h - offset_y if i == ah-1 else  out_h
                
                # target_startx = max(0, j * inner_block_w - offset_x)
                # target_endx = min(w, (j+1) * inner_block_w + offset_x)
                # target_starty = max(0, i * inner_block_h - offset_y)
                # target_endy = min(h, (i+1) * inner_block_h + offset_y)

                # len_x = target_endx - target_startx
                # len_y = target_endy - target_starty

                # out_h//2 - len_x
                
                output = net_utils.composite_centered_numpy(output, output_buf_u8[i,j,:,:,:], i * inner_block_w + inner_block_w//2, j * inner_block_h + inner_block_h//2)

                
                # output[:,target_starty:target_endy, target_startx:target_endx] = output_buf_u8[i,j,:,source_starty:source_endy, source_startx:source_endx]
                print('error ', (np.square(output_buf_fd - output)/np.square(output_buf_fd)).mean())
                
                halide.imageio.imwrite(p_center, np.ascontiguousarray(output_buf_u8[i,j,:,:,:]))
                halide.imageio.imwrite(p, np.ascontiguousarray(output))
                halide.imageio.imwrite(p_fd, np.ascontiguousarray(output_buf_fd))
                halide.imageio.imwrite(p_dense, np.ascontiguousarray(output_buf_dense[i,j,:,:,:]))
                halide.imageio.imwrite(p_aux_dense_0, np.ascontiguousarray(aux_buf_dense_0[i,j,:,:,:]))
                halide.imageio.imwrite(p_aux_sparse_0, np.ascontiguousarray(aux_buf_0[i,j,:,:,:]))
                halide.imageio.imwrite(p_aux_dense_1, np.ascontiguousarray(aux_buf_dense_1[i,j,:,:,:]))
                halide.imageio.imwrite(p_aux_sparse_1, np.ascontiguousarray(aux_buf_1[i,j,:,:,:]))

    print("Saving to %s ..." % guide_path)
    halide.imageio.imwrite(guide_path, guided_buf_u8)
    print("Saving to %s ..." % input_path)
    halide.imageio.imwrite(input_path, input_buf_u8)
    print("Saving to %s ..." % alpha_path)
    halide.imageio.imwrite(alpha_path, alpha_map_u8)
    

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
    eps = 0.001
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
    inner_block_w =  w // aw
    inner_block_h =  h // ah
    block_h, block_w = inner_block_h * 2, inner_block_w * 2
    oy, ox = inner_block_h//2, inner_block_w//2
    
    denoising(np.ascontiguousarray(noisy_flash_scaled.numpy()[0].transpose(2,0,1)),
              np.ascontiguousarray(deepfnf_scaled.numpy()[0].transpose(2,0,1)),
              opts.llf_levels, model.predict_alpha(None).numpy(), 1, './deriv_out/rgb_out.png')
    with tf.GradientTape() as tape:
        gllf_scaled = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
        # Loss
        l2_loss = tfu.l2_loss(gllf_scaled, ambient_scaled)
        gradient_loss = tfu.gradient_loss(gllf_scaled, ambient_scaled)
        loss = l2_loss + gradient_loss
    gradients_loss = tape.gradient(loss, model.weights.values())
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
            alpha_map[i,j].assign(alpha_map[i,j] - eps)
            llf_n = call_llf(tf.transpose(noisy_flash_scaled[0,...],(2,0,1)),
                    tf.transpose(deepfnf_scaled[0,...],(2,0,1)),
                    alpha_map, opts.llf_levels, opts.llf_beta, ah, aw, h, w)
            fd_llf = (llf_0 - llf_n) / (eps)

            source_startx = ox if j == 0 else 0
            source_endx = block_w - ox if j == aw-1 else w
            source_starty = oy if i == 0 else 0
            source_endy = block_h - oy if i == ah-1 else h
            
            target_startx = 0 if j == 0 else j * inner_block_w - ox
            target_endx = w if j == aw-1 else (j+1) * inner_block_w + ox
            target_starty = 0 if i == 0 else i * inner_block_h - oy
            target_endy = h if i == ah-1 else (i+1) * inner_block_h + oy
            
            output = np.zeros_like(fd_llf, dtype=np.float32)
            output[:,target_starty:target_endy, target_startx:target_endx] = dllf[i,j,:,source_starty:source_endy, source_startx:source_endx]
            # output = dllf
            cv2.imwrite('./deriv_out/dllf_%i_%i.exr'%(i, j), output.transpose(1,2,0))
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
            gradient_loss = tfu.gradient_loss(gllf_scaled_n, ambient_scaled)
            loss_n = l2_loss + gradient_loss
            
            flat[i].assign(w_val)
            model.weights[k] = tf.reshape(flat,orig_shape)
            gllf_scaled_p = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
            l2_loss = tfu.l2_loss(gllf_scaled_p, ambient_scaled)
            gradient_loss = tfu.gradient_loss(gllf_scaled_p, ambient_scaled)
            loss_p = l2_loss + gradient_loss

            fd_tensor[i].assign((loss_p - loss_n) / (eps))
            flat[i].assign(w_val)
            model.weights[k] = tf.reshape(flat, orig_shape)
            # if(store_image):
            #     cv2.imwrite('gradients_image.exr', tf.abs(gradients_image[0][0]).numpy())
            #     cv2.imwrite('gllf_scaled_p.exr', tf.abs(gllf_scaled_p[0]).numpy())
            #     cv2.imwrite('gllf_scaled_n.exr', tf.abs(gllf_scaled_n[0]).numpy())
            print(i, 'th gradient: ', grad_flat[i].numpy(), ' fd ', fd_tensor[i].numpy())
            # assert tf.abs(grad_flat[i] - fd_tensor[i]) < 1e-4
        print('mean of relative error is ', tf.sqrt(tf.reduce_sum(tf.square(grad_flat - fd_tensor)) / tf.reduce_sum(tf.square(fd_tensor))))
        fd[k] = tf.reshape(fd_tensor, orig_shape)
        # fd_grad_diff[k] = gradients[k] - fd[k]
    # print('hi')
    
net_input, alpha, noisy_flash, noisy_ambient = prepare_input(example)
gradient_validation(net_input, alpha, noisy_flash, noisy_ambient)