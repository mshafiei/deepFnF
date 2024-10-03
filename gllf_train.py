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
from datetime import datetime
from cvgutils.nn.lpips_tf2.models_tensorflow.lpips_tensorflow import load_perceptual_models, learned_perceptual_metric_model
import cv2
# tf.config.run_functions_eagerly(True)

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
    deepfnf_params = logger.load_params(opts.deepfnf_train_path)
    if(deepfnf_params != None):
        deepfnf_model.weights = deepfnf_params['params']
    else:
        print('cannot load deepfnf parameters ', opts.deepfnf_train_path)
        exit(0)
    params = logger.load_params()
    if (params is not None) and ('params' in params.keys()):
        model.weights = params['params']
    model.deepfnf_model = deepfnf_model
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
    def gradient_validation(net_input, alpha, noisy_flash, noisy_ambient, example):
        def loss_eval(model, gt, net_ft_input, noisy_flash_scaled, deepfnf_scaled):
            refined_scaled = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
            # Loss
            l2_loss = tfu.l2_loss(refined_scaled, gt)
            gradient_loss = tfu.gradient_loss(refined_scaled, noisy_flash_scaled)
            
            # lpips_loss = tf.stop_gradient(lpips([denoise, ambient]))
            loss = l2_loss + gradient_loss# + opts.lpips * lpips_loss[0]
            return loss
        eps = 1e-3 #choose larger values for larger alpha map
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
            loss = loss_eval(model, noisy_flash_scaled, net_ft_input, noisy_flash_scaled, deepfnf_scaled)
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
                loss_n = loss_eval(model, noisy_flash_scaled, net_ft_input, noisy_flash_scaled, deepfnf_scaled)
                
                flat[i].assign(w_val + eps)
                model.weights[k] = tf.reshape(flat,orig_shape)
                loss_p = loss_eval(model, noisy_flash_scaled, net_ft_input, noisy_flash_scaled, deepfnf_scaled)

                fd_tensor[i].assign((loss_p - loss_n) / (2*eps))
                flat[i].assign(w_val)
                model.weights[k] = tf.reshape(flat, orig_shape)
                err = tf.abs(grad_flat[i] - fd_tensor[i])
                print(i,'/',flat.shape[0],' err: ', err)
                

            fd[k] = tf.reshape(fd_tensor, orig_shape)
            fd_grad_diff[k] = tf.reduce_mean(tf.abs(gradients - fd[k]) / (tf.abs(fd[k])+0.0001))
            print('relative error: ',fd_grad_diff[k].numpy())
            # fd_grad_diff[k] = gradients - fd[k]
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
    def val_step(net_input, alpha, noisy_flash, noisy_ambient, example):
        denoise = tf.stop_gradient(deepfnf_model.forward(net_input))
        
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
        
        refinement_output = model.forward(net_ft_input, noisy_flash_scaled, denoise_scaled)
        if(type(refinement_output) == tuple):
            refined_scaled = refinement_output[0]
            llf_alpha = refinement_output[1]
        else:
            refined_scaled = refinement_output
            llf_alpha = None
        return refined_scaled, denoise_scaled, ambient_scaled, noisy_flash_scaled, noisy_ambient_scaled, llf_alpha
    
    @tf.function
    def predict_and_scale(net_input, alpha, noisy_flash, noisy_ambient, example):
        denoise = tf.stop_gradient(deepfnf_model.forward(net_input))
        deepfnf_scaled = tfu.camera_to_rgb(
            denoise / alpha, example['color_matrix'], example['adapt_matrix'])
        
        noisy_flash_scaled = tfu.camera_to_rgb(
            noisy_flash, example['color_matrix'], example['adapt_matrix'])
        
        ambient_scaled= tfu.camera_to_rgb(
            example['ambient'],
            example['color_matrix'], example['adapt_matrix'])

        return denoise, deepfnf_scaled, noisy_flash_scaled, ambient_scaled
    
    @tf.function
    def predict_losses(net_input, alpha, noisy_flash, noisy_ambient, example):
        denoise, deepfnf_scaled, noisy_flash_scaled, ambient_scaled = predict_and_scale(net_input, alpha, noisy_flash, noisy_ambient, example)
        net_ft_input = tf.concat((net_input, denoise), axis=-1)
        refinement_out = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
        if(type(refinement_out) == tuple):
            refined_scaled = refinement_out[0]
            alpha_map = refinement_out[1]
        else:
            refined_scaled = refinement_out
            alpha_map = None

        psnr_refined = tfu.get_psnr(refined_scaled, ambient_scaled)
        psnr_deepfnf = tfu.get_psnr(deepfnf_scaled, ambient_scaled)
        wlpips_deepfnf = wlpips([deepfnf_scaled, ambient_scaled])
        lpips_deepfnf = lpips([deepfnf_scaled, ambient_scaled])
        wlpips_refined = wlpips([refined_scaled, ambient_scaled])
        lpips_refined = lpips([refined_scaled, ambient_scaled])
        losses = {'psnr_refined':psnr_refined, 'psnr_deepfnf':psnr_deepfnf,
            'wlpips_deepfnf':wlpips_deepfnf, 'lpips_deepfnf':lpips_deepfnf,
            'wlpips_refined':wlpips_refined, 'lpips_refined':lpips_refined}
        return losses

    @tf.function
    def train_step(net_input, alpha, noisy_flash, noisy_ambient, example):
        denoise, deepfnf_scaled, noisy_flash_scaled, ambient_scaled = predict_and_scale(net_input, alpha, noisy_flash, noisy_ambient, example)
        
        net_ft_input = tf.concat((net_input, denoise), axis=-1)
        with tf.GradientTape() as tape:
            refinement_out = model.forward(net_ft_input, noisy_flash_scaled, deepfnf_scaled)
            if(type(refinement_out) == tuple):
                refined_scaled = refinement_out[0]
                alpha_map = refinement_out[1]
            else:
                refined_scaled = refinement_out
                alpha_map = None
            # Loss
            l2_loss = tf.convert_to_tensor(0.0) if opts.l2 == 0 else tfu.l2_loss(refined_scaled, ambient_scaled)
            gradient_loss = tf.convert_to_tensor(0.0) if opts.grad == 0 else tfu.gradient_loss(refined_scaled, ambient_scaled)
            wlpips_loss = tf.convert_to_tensor(0.0) if opts.wlpips == 0 else wlpips([refined_scaled, ambient_scaled])[0]
            lpips_loss = tf.convert_to_tensor(0.0) if opts.lpips == 0 else lpips([refined_scaled, ambient_scaled])[0]
            
            # lpips_loss = tf.stop_gradient(lpips([denoise, ambient]))
            loss = l2_loss + gradient_loss + opts.lpips * lpips_loss + opts.wlpips * wlpips_loss

        gradients = tape.gradient(loss, model.weights.values())
        opt.apply_gradients(zip(gradients,model.weights.values()))

        losses = {'loss':loss, 'l2_loss':l2_loss, 'gradient_loss':gradient_loss,
        'wlpips_loss':opts.wlpips * wlpips_loss, 'lpips_loss':opts.lpips * lpips_loss}
        return losses

    def training_iterate(net_input, alpha, noisy_flash, noisy_ambient, niter, example):
        
        losses = train_step(net_input, alpha, noisy_flash, noisy_ambient, example)

        
        
        # Save model weights if needed
        if SAVEFREQ > 0 and niter % SAVEFREQ == 0:
            store = {}
            opt.save_own_variables(store)
            fn1, fn2 = logger.save_params(model.weights, {'configs':opt.get_config(), 'variables':store},niter)
            print("Saving model to " + fn1 + " and " + fn2 +" with loss ",float(losses['loss'].numpy()))
            # print('dumping params ',model.weights['down2_1_w'][0,0,0,0])
        if(niter % 100 == 0 and niter != 0):
            [logger.addScalar(float(v.numpy()),k) for k, v in losses.items()]
        if niter % VALFREQ == 0:
            additional_loss = predict_losses(net_input, alpha, noisy_flash, noisy_ambient, example)
            losses.update(additional_loss)
            # draw example['ambient'], denoised image, flash image, absolute error
            gllf, denoisednp, ambientnp, flashnp, noisy, alpha_map = val_step(net_input, alpha, noisy_flash, noisy_ambient, example)
            annotation_deepfnf = '<br>PSNR:%.3f<br>LPIPS:%.3f<br>WLPIPS:%.3f'%(additional_loss['psnr_deepfnf'],additional_loss['lpips_deepfnf'],additional_loss['wlpips_deepfnf'])
            annotation_ours = '<br>PSNR:%.3f<br>LPIPS:%.3f<br>WLPIPS:%.3f'%(additional_loss['psnr_refined'],additional_loss['lpips_refined'],additional_loss['wlpips_refined'])
            annotation = {'flash':None,'noisy':None,'ambient':None,'denoised_deepfnf':annotation_deepfnf,'denoised_gllf':annotation_ours,'alpha_map':None}
            
            alpha_map = cv2.resize(alpha_map.numpy(), (448,448))
            
            images = {'flash':flashnp.numpy()[0], 'noisy':noisy.numpy()[0], 'ambient':ambientnp.numpy()[0], 'denoised_deepfnf':denoisednp.numpy()[0], 'denoised_gllf':gllf.numpy()[0],'alpha_map':alpha_map[:,:,None]}
            lbls = {'flash':'Flash','noisy':'Noisy','ambient':'Ambient','denoised_deepfnf':'DeepFnF','denoised_gllf':'DeepFnF+GLLF','alpha_map':'$\\alpha$'}
            if('filename' in example.keys()):
                logger.addImage(images, lbls,'train',annotation=annotation, image_filename=example['filename'])


        losses_str = ', '.join('%s_%.07f'%(k, float(v.numpy())) for k, v in losses.items())
        print(datetime.now(), ' lr: ', float(opt.learning_rate.numpy()), ' iter: ',niter, losses_str, ' alpha %0.4f' % float(example['alpha'].numpy()))
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

    
    for data in dataset.iterator:
        net_input, alpha, noisy_flash, noisy_ambient = prepare_input(data)
        if(niter > MAXITER):
            break
        niter += 1
        if(opts.overfit):
            #if example does not exist, save it, otherwise load it
            overfit_example_gt_data_fn = './overfit_example_data_gt.pkl'
            overfit_example_noisy_data_fn = './overfit_example_data_noisy.pkl'
            if(os.path.exists(overfit_example_gt_data_fn) and os.path.exists(overfit_example_noisy_data_fn)):
                print('loaded example from file')
                data_gt = logger.load_pickle(overfit_example_gt_data_fn)
                data_noisy = logger.load_pickle(overfit_example_noisy_data_fn)
                net_input = data_noisy['net_input']
                alpha = data_noisy['alpha']
                noisy_flash = data_noisy['noisy_flash']
                noisy_ambient = data_noisy['noisy_ambient']
                niter = data_noisy['niter']
                
            else:
                print('could not load example from file')
                denoise = tf.stop_gradient(deepfnf_model.forward(net_input))
                data_noisy = {'net_input':net_input, 'alpha':alpha, 'noisy_flash':noisy_flash, 'noisy_ambient':noisy_ambient, 'niter':niter, 'denoise':denoise}
                logger.dump_pickle(overfit_example_gt_data_fn, data)
                logger.dump_pickle(overfit_example_noisy_data_fn, data_noisy)
            for _ in range(int(MAXITER)):
                niter += 1
                training_iterate(net_input, alpha, noisy_flash, noisy_ambient, niter, data)
        else:
            # gradient_validation(net_input, alpha, noisy_flash, noisy_ambient)
            training_iterate(net_input, alpha, noisy_flash, noisy_ambient, niter, data)

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
