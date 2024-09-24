#!/usr/bin/env python3
from arguments_deepfnf import parse_arguments_deepfnf
parser = parse_arguments_deepfnf()
opts = parser.parse_args()
import tensorflow as tf
import os
import argparse

import utils.np_utils as npu
import numpy as np
from test import test
import net_ksz3
import net
import net_cheap as netCheap
from net_laplacian_combine import Net as netLaplacianCombine
from net_no_change import NetNoChange as netNoChange
from net_fft_combine import Net as netFFTCombine
from net_flash_image import Net as netFlash
from net_fft import Net as netFFT
from net_laplacian_combine_pixelwise import Net as netLaplacianCombinePixelWise
from net_no_scalemap import Net as NetNoScaleMap
from net_grad import Net as NetGrad
from net_slim import Net as NetSlim
import gllf_network_utils as net_utils
import utils.utils as ut
import utils.tf_utils as tfu
from utils.dataset_prefetch import TrainSet as TrainSet_prefetch
from utils.dataset_prefetch_nthreads import TrainSet as TrainSet_prefetch_nthread
from utils.dataset import Dataset
from utils.dataset_filelock import TrainSet as Trainset_filelock
import cvgutils.Viz as Viz
import time
from tensorflow.python.profiler import profiler_v2 as profiler
import keras
from cvgutils.nn.lpips_tf2.models_tensorflow.lpips_tensorflow import load_perceptual_models, learned_perceptual_metric_model
tf.config.run_functions_eagerly(True)

# num_cores = tf.config.experimental.get_cpu_device_count()
# tf.config.threading.set_intra_op_parallelism_threads(num_cores)
# tf.config.threading.set_inter_op_parallelism_threads(1)
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["KMP_BLOCKTIME"] = "1"
# os.environ["KMP_SETTINGS"] = "1"
# os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

image_size=448
local_ckpt_dir = '/home/mohammad/cvgutils/cvgutils/nn/lpips_tf2/weights/keras'
server_ckpt_dir = '/mshvol2/users/mohammad/cvgutils/cvgutils/nn/lpips_tf2/weights/keras'
ckpt_dir = local_ckpt_dir if os.path.exists(local_ckpt_dir) else server_ckpt_dir
vgg_ckpt_fn = os.path.join(ckpt_dir, 'vgg', 'exported.weights.h5')
lin_ckpt_fn = os.path.join(ckpt_dir, 'lin', 'exported.weights.h5')
lpips_net, lpips_lin = load_perceptual_models(image_size, vgg_ckpt_fn, lin_ckpt_fn)
lpips = learned_perceptual_metric_model(lpips_net, lpips_lin, image_size)
wlpips = learned_perceptual_metric_model(lpips_net, lpips_lin, image_size, 'wlpips')

logger = Viz.logger(opts,opts.__dict__)
_, weight_dir = logger.path_parse('train')
opts.weight_file = os.path.join(weight_dir,opts.weight_file)

print("weights_dir: ",weight_dir)
opts = logger.opts
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
wts = weight_dir

if not os.path.exists(wts):
    os.makedirs(wts)

boundaries = [DROP[0], DROP[1]]
values = [float(LR), float(LR/np.sqrt(10)), float(LR/10)]
learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

def load_net(fn, model):
    if(hasattr(model,'weights')):
        wts = np.load(fn)
        for k, v in wts.items():
            model.weights[k] = tf.Variable(v)
    else:
        print('Model does not have weights')
    return model

if(opts.mode == 'test'):
    model, deepfnf_model = net_utils.CreateNetwork(opts)
    params = logger.load_params()
    if (params is not None) and ('params' in params.keys()):
        model.weights = params['params']
    test(model, opts.weight_file, opts.TESTPATH,logger)
    exit(0)
else:
    model, deepfnf_model = net_utils.CreateNetwork(opts)


#########################################################################

with tf.device('/cpu:0'):
    if opts.dataset_model == 'prefetch_nthread':
        dataset = TrainSet_prefetch_nthread(TLIST, bsz=BSZ, psz=IMSZ,
                            ngpus=opts.ngpus, nthreads=4 * opts.ngpus,jitter=opts.displacement,min_scale=opts.min_scale,max_scale=opts.max_scale,theta=opts.max_rotate)
    elif opts.dataset_model == 'prefetch':
        dataset = TrainSet_prefetch(TLIST, bsz=BSZ, psz=IMSZ,
                            ngpus=opts.ngpus, nthreads=4 * opts.ngpus,jitter=opts.displacement,min_scale=opts.min_scale,max_scale=opts.max_scale,theta=opts.max_rotate)
    elif opts.dataset_model == 'filelock':
        dataset = Trainset_filelock(TLIST, bsz=BSZ, psz=IMSZ,
                            ngpus=opts.ngpus, nthreads=4 * opts.ngpus,jitter=opts.displacement,min_scale=opts.min_scale,max_scale=opts.max_scale,theta=opts.max_rotate)
    else:
        dataset = Dataset(TLIST, VPATH, bsz=BSZ, psz=IMSZ, ngpus=opts.ngpus, nthreads=4 * opts.ngpus,jitter=opts.displacement,min_scale=opts.min_scale,max_scale=opts.max_scale,theta=opts.max_rotate)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    # opt = tf.keras.optimizers.Adam(learning_rate=LR)
    
with tf.device('/gpu:0'):
    niter = 0

    deepfnf_params = logger.load_params(opts.deepfnf_train_path)
    if(deepfnf_params != None):
        deepfnf_model.weights = deepfnf_params['params']
    else:
        print('cannot load deepfnf parameters ', opts.deepfnf_train_path)
        exit(0)
    params = logger.load_params()
    if(params is not None):
        niter = params['idx']
        model.weights = params['params']
        serialized_dict = {'module':'keras.optimizers', 'class_name':'Adam', 'config':params['state']['configs'], 'registered_name':None}
        if(opts.model == 'deepfnf_llf_diffable' and 'alpha_encoderinp_w' not in model.weights.keys()):
            #keep the fresh optimizer
            pass
        else:
            opt = tf.keras.optimizers.deserialize(serialized_dict)
            opt.build(list(params['params'].values()))
            opt.set_weights(list(params['state']['variables'].values()))
            opt.from_config(params['state']['configs'])
            print('Successfully loaded parameters from ', params['filename'], ' for iteration ', niter)

    summary = '\n'.join(['%s %s' % (i, model.weights[i].shape) for i in model.weights] + ['total parameter count = %i' % np.sum([np.prod(model.weights[i].shape) for i in model.weights]) ])
    logger.addString(summary,'model_summary')
    print("===================== Model summary =====================")
    print(summary)
    print("===================== Model summary =====================")

    #verify finite difference
    
    @tf.function
    def gradient_validation(net_input, alpha, noisy_flash, noisy_ambient):
        eps = 1e-4
        denoise = deepfnf_model.forward(net_input)
        
        deepfnf_scaled = tfu.camera_to_rgb(
            denoise / alpha, example['color_matrix'], example['adapt_matrix'])
        
        noisy_flash_scaled = tfu.camera_to_rgb(
            noisy_flash, example['color_matrix'], example['adapt_matrix'])
        
        ambient_scaled= tfu.camera_to_rgb(
            example['ambient'],
            example['color_matrix'], example['adapt_matrix'])
        
        net_ft_input = tf.concat((net_input, denoise), axis=-1)
        
        with tf.GradientTape() as tape:
            gllf_scaled = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
            # Loss
            l2_loss = tfu.l2_loss(gllf_scaled, ambient_scaled)
            # gradient_loss = tfu.gradient_loss(gllf_scaled, ambient_scaled)
            
            # lpips_loss = tf.stop_gradient(lpips([denoise, ambient]))
            loss = l2_loss # + gradient_loss# + opts.lpips * lpips_loss[0]
        gradients = tape.gradient(loss, model.weights.values())

        fd = {}
        fd_grad_diff = {}
        for k_i, k in enumerate(model.weights.keys()):
            orig_shape = model.weights[k].shape
            flat = tf.Variable(tf.reshape(model.weights[k],-1))
            fd_tensor = tf.Variable(tf.zeros_like(flat))
            grad_flat = tf.Variable(tf.reshape(gradients[k_i],-1))
            for i in range(flat.shape[0]):
                w_val = flat[i]

                # values = tf.identity(model.weights[k])
                flat[i].assign(w_val - eps)
                model.weights[k] = tf.reshape(flat,orig_shape)
                gllf_scaled = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
                l2_loss = tfu.l2_loss(gllf_scaled, ambient_scaled)
                # gradient_loss = tfu.gradient_loss(gllf_scaled, ambient_scaled)
                loss_n = l2_loss #+ gradient_loss
                
                flat[i].assign(w_val + eps)
                model.weights[k] = tf.reshape(flat,orig_shape)
                gllf_scaled = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
                l2_loss = tfu.l2_loss(gllf_scaled, ambient_scaled)
                # gradient_loss = tfu.gradient_loss(gllf_scaled, ambient_scaled)
                loss_p = l2_loss #+ gradient_loss

                fd_tensor[i].assign((loss_p - loss_n) / (2*eps))
                flat[i].assign(w_val)
                model.weights[k] = tf.reshape(flat, orig_shape)
                assert tf.abs(grad_flat[i] - fd_tensor[i]) < 1e-4

            fd[k] = tf.reshape(fd_tensor, orig_shape)
            fd_grad_diff[k] = gradients[k] - fd[k]
        print('hi')
        
        


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
    def val_step(net_input, alpha, noisy_flash, noisy_ambient):
        denoise = deepfnf_model.forward(net_input)
        
        noisy_ambient_scaled = tfu.camera_to_rgb(
            noisy_ambient / alpha, example['color_matrix'], example['adapt_matrix'])

        denoise_scaled = tfu.camera_to_rgb(
            denoise / alpha, example['color_matrix'], example['adapt_matrix'])
        
        noisy_flash_scaled = tfu.camera_to_rgb(
            noisy_flash, example['color_matrix'], example['adapt_matrix'])
        
        ambient_scaled= tfu.camera_to_rgb(
            example['ambient'],
            example['color_matrix'], example['adapt_matrix'])
        
        net_ft_input = tf.concat((net_input, denoise), axis=-1)
        
        gllf_scaled = model.forward(net_ft_input, noisy_flash_scaled, denoise_scaled)

        return gllf_scaled, denoise_scaled, ambient_scaled, noisy_flash_scaled, noisy_ambient_scaled
        

    @tf.function
    def train_step(net_input, alpha, noisy_flash, noisy_ambient):
        denoise = deepfnf_model.forward(net_input)
        
        deepfnf_scaled = tfu.camera_to_rgb(
            denoise / alpha, example['color_matrix'], example['adapt_matrix'])
        
        noisy_flash_scaled = tfu.camera_to_rgb(
            noisy_flash, example['color_matrix'], example['adapt_matrix'])
        
        ambient_scaled= tfu.camera_to_rgb(
            example['ambient'],
            example['color_matrix'], example['adapt_matrix'])
        
        net_ft_input = tf.concat((net_input, denoise), axis=-1)
        with tf.GradientTape() as tape:
            gllf_scaled = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)

            # Loss
            l2_loss = tfu.l2_loss(gllf_scaled, ambient_scaled)
            gradient_loss = tfu.gradient_loss(gllf_scaled, ambient_scaled)
            
            # lpips_loss = tf.stop_gradient(lpips([denoise, ambient]))
            loss = l2_loss + gradient_loss# + opts.lpips * lpips_loss[0]

        gradients = tape.gradient(loss, model.weights.values())
        opt.apply_gradients(zip(gradients,model.weights.values()))
        psnr_gllf = tfu.get_psnr(gllf_scaled, ambient_scaled)
        psnr_deepfnf = tfu.get_psnr(deepfnf_scaled, ambient_scaled)
        return loss, psnr_gllf, psnr_deepfnf, l2_loss, gradient_loss

    def training_iterate(net_input, alpha, noisy_flash, noisy_ambient, niter):
        loss, psnr_gllf, psnr_deepfnf, l2_loss, gradient_loss= train_step(net_input, alpha, noisy_flash, noisy_ambient)
        print('lr: ', float(opt.learning_rate.numpy()), ' iter: ',niter, ' loss: ', loss.numpy(), ' l2_loss ',l2_loss.numpy(), ' gradient_loss ',gradient_loss.numpy(), ' psnr_gllf: ', psnr_gllf.numpy(), ' psnr_deepfnf: ', psnr_deepfnf.numpy())
        
        # Save model weights if needed
        if SAVEFREQ > 0 and niter % SAVEFREQ == 0:
            store = {}
            opt.save_own_variables(store)
            fn1, fn2 = logger.save_params(model.weights, {'configs':opt.get_config(), 'variables':store},niter)
            print("Saving model to " + fn1 + " and " + fn2 +" with loss ",loss.numpy())
            print('dumping params ',model.weights['down2_1_w'][0,0,0,0])
        if(niter % 100 == 0 and niter != 0):
            logger.addScalar(loss.numpy(),'loss')
            logger.addScalar(psnr_gllf.numpy(),'psnr_gllf')
            logger.addScalar(psnr_deepfnf.numpy(),'psnr_deepfnf')
        if VALFREQ > 0 and niter % VALFREQ == 0:
            # draw example['ambient'], denoised image, flash image, absolute error
            gllf, denoisednp, ambientnp, flashnp, noisy = val_step(net_input, alpha, noisy_flash, noisy_ambient)
            psnr_deepfnf = tfu.get_psnr(denoisednp, ambientnp)
            logger.addImage({'flash':flashnp.numpy()[0], 'noisy':noisy.numpy()[0], 'ambient':ambientnp.numpy()[0], 'denoised_deepfnf':denoisednp.numpy()[0], 'denoised_gllf':gllf.numpy()[0]},{'flash':'Flash','noisy':'Noisy','ambient':'Ambient','denoised_deepfnf':'DeepFnF','denoised_gllf':'DeepFnF+GLLF'},'train')
        logger.takeStep()

    def training_iterate_synthetic(flash, denoised, niter):
        net_input = tf.concat((flash, denoised),axis=-1)
        
        with tf.GradientTape() as tape:
            denoise_scaled = model.forward(net_input, flash, denoised)

            # Loss
            l2_loss = tfu.l2_loss(denoise_scaled, flash)
            gradient_loss = tfu.gradient_loss(denoise_scaled, flash)
            
            # lpips_loss = tf.stop_gradient(lpips([denoise, ambient]))
            loss = l2_loss + gradient_loss# + opts.lpips * lpips_loss[0]
        
        if VALFREQ > 0 and niter % VALFREQ == 0:
            # draw example['ambient'], denoised image, flash image, absolute error
            logger.addImage({'flash':flash.numpy()[0], 'denoised':denoised.numpy()[0], 'ambient':denoised.numpy()[0], 'gllf_out':denoise_scaled.numpy()},{'flash':'Flash','ambient':'Ambient','denoised':'Denoise', 'gllf_out':'GLLF output'},'train')
        logger.takeStep()

        gradients = tape.gradient(loss, model.weights.values())
        opt.apply_gradients(zip(gradients,model.weights.values()))
        print('loss ', loss)
        return loss

    
    for example in dataset.iterator:
        net_input, alpha, noisy_flash, noisy_ambient = prepare_input(example)
        if(niter > MAXITER):
            break
        niter += 1
        # gradient_validation(net_input, alpha, noisy_flash, noisy_ambient)
        training_iterate(net_input, alpha, noisy_flash, noisy_ambient, niter)

    #synthetic test case
    # flash = np.ones([1,448,448,3],dtype=np.float32)
    # denoised = np.ones([1,448,448,3],dtype=np.float32)
    # flash[:,100:200,100:200,:] = 0.2
    # denoised[:,300:400,300:400,:] = 0.2
    # flash = tf.convert_to_tensor(flash)
    # denoised = tf.convert_to_tensor(denoised)
    # for _ in range(10000):
    #     if(niter > MAXITER):
    #         break
    #     niter += 1
    #     # training_iterate(example, niter)
    #     training_iterate_synthetic(flash, denoised, niter)

            
store = {}
opt.save_own_variables(store)
fn1, fn2 = logger.save_params(model.weights, {'configs':opt.get_config(), 'variables':store},niter)
print("Saving model to " + fn1 + " and " + fn2)
