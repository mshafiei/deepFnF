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
# import net_ksz3
import deepfnf
import net_cheap as netCheap
# from net_laplacian_combine import Net as netLaplacianCombine
# from net_no_change import NetNoChange as netNoChange
from net_fft_combine import Net as netFFTCombine
from net_flash_image import Net as netFlash
from net_fft import Net as netFFT
# from net_laplacian_combine_pixelwise import Net as netLaplacianCombinePixelWise
from net_no_scalemap import Net as NetNoScaleMap
from net_grad import Net as NetGrad
# from net_slim import Net as NetSlim
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
from easydict import EasyDict as edict
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
SAVEFREQ_RARE = opts.save_freq_rare
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

    if(opts.double_network):
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
    def prepare_input(example, clamp=False, std_input=True):
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

        if(clamp):
            noisy_ambient = tf.maximum(noisy_ambient,0)
            noisy_flash = tf.maximum(noisy_flash,0)
        
        noisy = tf.concat([noisy_ambient, noisy_flash], axis=-1)
        if(std_input):
            
            noise_std = tfu.estimate_std(noisy, sig_read, sig_shot)
            net_input = tf.concat([noisy, noise_std], axis=-1)
        else:
            net_input = noisy

        return net_input, alpha, noisy_flash, noisy_ambient


    @tf.function
    def val_step(net_input, alpha, noisy_flash, noisy_ambient, example, double_network=True):
        outputs = edict()
        model_input = edict()
        if(double_network):
            outputs.denoise = tf.stop_gradient(deepfnf_model.forward(net_input))
            outputs.deepfnf_scaled = tfu.camera_to_rgb(
            outputs.denoise / alpha, example['color_matrix'], example['adapt_matrix'])
            outputs.net_ft_input = tf.concat((net_input, outputs.denoise), axis=-1)
            model_input.deepfnf_scaled = outputs.deepfnf_scaled
        else:
            outputs.net_ft_input = net_input
        
        outputs.noisy_ambient_scaled = tfu.camera_to_rgb(
            noisy_ambient / alpha, example['color_matrix'], example['adapt_matrix'])
        
        outputs.noisy_flash_scaled = tfu.camera_to_rgb(
            noisy_flash, example['color_matrix'], example['adapt_matrix'])
        
        outputs.ambient_scaled= tfu.camera_to_rgb(
            example['ambient'],
            example['color_matrix'], example['adapt_matrix'])
        
        model_input.update(edict(net_ft_input=outputs.net_ft_input, noisy_flash_scaled=outputs.noisy_flash_scaled, color_matrix=example['color_matrix'], adapt_matrix=example['adapt_matrix']))
        outputs.model_output = model.forward(model_input)
        return outputs, model_input


    @tf.function
    def predict_losses(net_input, alpha, noisy_flash, noisy_ambient, example, double_network=True):
        output = edict()
        losses = edict()
        output.noisy_ambient =  tfu.camera_to_rgb(
            noisy_ambient / alpha, example['color_matrix'], example['adapt_matrix'])

        output.noisy_flash_scaled = tfu.camera_to_rgb(
            noisy_flash, example['color_matrix'], example['adapt_matrix'])
        
        output.ambient_scaled= tfu.camera_to_rgb(
            example['ambient'],
            example['color_matrix'], example['adapt_matrix'])

        if(double_network):
            output.denoise = tf.stop_gradient(deepfnf_model.forward(net_input))
            net_ft_input = tf.concat((net_input, output.denoise), axis=-1)
            output.deepfnf_scaled = tfu.camera_to_rgb(
            output.denoise / alpha, example['color_matrix'], example['adapt_matrix'])
            psnr_deepfnf = tfu.get_psnr(output.deepfnf_scaled, output.ambient_scaled)
            wlpips_deepfnf = wlpips([output.deepfnf_scaled, output.ambient_scaled])
            lpips_deepfnf = lpips([output.deepfnf_scaled, output.ambient_scaled])
            losses.psnr_deepfnf = psnr_deepfnf
            losses.wlpips_deepfnf = wlpips_deepfnf
            losses.lpips_deepfnf = lpips_deepfnf
        else:
            net_ft_input = net_input

        output.net_ft_input = net_ft_input
        output.color_matrix = example['color_matrix']
        output.adapt_matrix = example['adapt_matrix']
        double_deepfnf = model.forward(output)

        psnr_refined = tfu.get_psnr(double_deepfnf.output, output.ambient_scaled)
        wlpips_refined = wlpips([double_deepfnf.output, output.ambient_scaled])
        lpips_refined = lpips([double_deepfnf.output, output.ambient_scaled])
        losses.psnr_refined = psnr_refined
        losses.wlpips_refined = wlpips_refined
        losses.lpips_refined = lpips_refined

        return losses, output

    @tf.function
    def train_step(net_input, alpha, noisy_flash, noisy_ambient, example,double_network=True):
        model_inputs = edict()
        
        model_inputs.noisy_ambient_scaled =  tfu.camera_to_rgb(noisy_ambient / alpha,
        example['color_matrix'], example['adapt_matrix'])

        model_inputs.noisy_flash_scaled = tfu.camera_to_rgb(noisy_flash,
        example['color_matrix'], example['adapt_matrix'])
        
        model_inputs.ambient_scaled= tfu.camera_to_rgb(example['ambient'],
            example['color_matrix'], example['adapt_matrix'])

        if(double_network):
            model_inputs.denoise = deepfnf_model.forward(net_input)
            net_ft_input = tf.concat((net_input, model_inputs.denoise), axis=-1)
            model_inputs.deepfnf_scaled = tfu.camera_to_rgb(
            model_inputs.denoise / alpha, example['color_matrix'], example['adapt_matrix'])
        else:
            net_ft_input = net_input
        
        
        model_inputs.net_ft_input = net_ft_input
        model_inputs.color_matrix = example['color_matrix']
        model_inputs.adapt_matrix = example['adapt_matrix']
        
        with tf.GradientTape() as tape:
            double_deepfnf = model.forward(model_inputs)
            # Loss
            l2_loss = tf.convert_to_tensor(0.0) if opts.l2 == 0 else tfu.l2_loss(double_deepfnf.output, model_inputs.ambient_scaled)
            gradient_loss = tf.convert_to_tensor(0.0) if opts.grad == 0 else tfu.gradient_loss(double_deepfnf.output, model_inputs.ambient_scaled)
            wlpips_loss = tf.convert_to_tensor(0.0) if opts.wlpips == 0 else wlpips([double_deepfnf.output, model_inputs.ambient_scaled])[0]
            lpips_loss = tf.convert_to_tensor(0.0) if opts.lpips == 0 else lpips([double_deepfnf.output, model_inputs.ambient_scaled])[0]
            
            # lpips_loss = tf.stop_gradient(lpips([denoise, ambient]))
            loss = opts.l2 * l2_loss + opts.grad * gradient_loss + opts.lpips * lpips_loss + opts.wlpips * wlpips_loss

        gradients = tape.gradient(loss, model.weights.values())
        # tf.print(gradients)
        opt.apply_gradients(zip(gradients,model.weights.values()))
        psnr_metric = tfu.get_psnr(double_deepfnf.output, model_inputs.ambient_scaled)
        losses = {'loss':loss, 'l2_loss':l2_loss, 'gradient_loss':gradient_loss,
        'wlpips_loss':opts.wlpips * wlpips_loss, 'lpips_loss':opts.lpips * lpips_loss, 'psnr':psnr_metric}
        return losses

    def training_iterate(net_input, alpha, noisy_flash, noisy_ambient, niter, example, double_network):
        losses = train_step(net_input, alpha, noisy_flash, noisy_ambient, example, double_network)

        # Save model weights if needed
        if SAVEFREQ > 0 and niter % SAVEFREQ == 0:
            store = {}
            opt.save_own_variables(store)
            fn1, fn2 = logger.save_params(model.weights, {'configs':opt.get_config(), 'variables':store},niter)
            print("Saving model to " + fn1 + " and " + fn2 +" with loss ",float(losses['loss'].numpy()))
            # print('dumping params ',model.weights['down2_1_w'][0,0,0,0])
        if SAVEFREQ > 0 and niter % SAVEFREQ_RARE == 0:
            store = {}
            opt.save_own_variables(store)
            fn1, fn2 = logger.save_params(model.weights, {'configs':opt.get_config(), 'variables':store},niter,suffix="rare_%i"%niter)
            print("Saving model to " + fn1 + " and " + fn2 +" with loss ",float(losses['loss'].numpy()))
            # print('dumping params ',model.weights['down2_1_w'][0,0,0,0])

        def visualize():
            additional_loss, deepfnf_out_dict = predict_losses(net_input, alpha, noisy_flash, noisy_ambient, example, double_network)
            deepfnf_out = deepfnf_out_dict
            losses.update(additional_loss)
            # draw example['ambient'], denoised image, flash image, absolute error
            val_output_dict, model_inputs_dict = val_step(net_input, alpha, noisy_flash, noisy_ambient, example, double_network)
            val_output, model_inputs = val_output_dict, model_inputs_dict
            if(double_network):
                annotation_deepfnf = '<br>PSNR:%.3f<br>LPIPS:%.3f<br>WLPIPS:%.3f'%(additional_loss['psnr_deepfnf'],additional_loss['lpips_deepfnf'],additional_loss['wlpips_deepfnf'])
            annotation_ours = '<br>PSNR:%.3f<br>LPIPS:%.3f<br>WLPIPS:%.3f'%(additional_loss['psnr_refined'],additional_loss['lpips_refined'],additional_loss['wlpips_refined'])
            annotation = {'flash':None,'noisy':None,'ambient':None,'denoised_gllf':annotation_ours,'alpha_map':None}
            if(double_network):
                annotation.update({'denoised_deepfnf':annotation_deepfnf})
            # exposure = 4 if opts.llf_sigma == 0 else 0
            
            images = {'flash':model_inputs.noisy_flash_scaled.numpy()[0], 'noisy':deepfnf_out.noisy_ambient.numpy()[0], 'ambient':deepfnf_out.ambient_scaled.numpy()[0], 'denoised_gllf':val_output.model_output.output}
            lbls = {'flash':'Flash','noisy':'Noisy','ambient':'Ambient','denoised_gllf':'DeepFnF+GLLF'}
            if(double_network):
                images.update({'denoised_deepfnf':deepfnf_out.deepfnf_scaled.numpy()[0]})
                lbls.update({'denoised_deepfnf':'DeepFnF'})
            if('alpha_map_h' in val_output.model_output):
                alpha_map_h = cv2.resize(val_output.model_output.alpha_map_h.numpy()[0], (448,448))[None,...]
                # gllf = gllf.numpy()[0] * 2.0**exposure
                annotation =  None if opts.llf_sigma == 0 else annotation
                alpha_h_min = tf.reduce_min(alpha_map_h)
                alpha_h_max = tf.reduce_max(alpha_map_h)
                alpha_map_h = (alpha_map_h - alpha_h_min) / (alpha_h_max - alpha_h_min)
                images.update({'alpha_map_h':alpha_map_h})
                lbls.update({'alpha_map_h':'$\\huge{\\alpha_h \\in [%.02f,%.02f]}$'%(alpha_h_min, alpha_h_max)})
            if('alpha_map_i' in val_output.model_output):
                alpha_map_i = cv2.resize(val_output.model_output.alpha_map_i.numpy()[0], (448,448))[None,...]
                alpha_i_min = tf.reduce_min(alpha_map_i)
                alpha_i_max = tf.reduce_max(alpha_map_i)
                alpha_map_i = (alpha_map_i - alpha_i_min) / (alpha_i_max - alpha_i_min)
                images.update({'alpha_map_i':alpha_map_h})
                lbls.update({'alpha_map_i':'$\\huge{\\alpha_i \\in [%.02f,%.02f]}$'%(alpha_i_min, alpha_i_max)})
            if("llf_guide" in val_output.model_output):
                images.update({'llf_guide':val_output.model_output.llf_guide.numpy()})
                lbls.update({'llf_guide':'I_h'})
            if("llf_input" in val_output.model_output):
                images.update({'llf_input':val_output.model_output.llf_input.numpy()})
                lbls.update({'llf_input':'I_i'})

            if('filename' in example.keys()):
                logger.addImage(images, lbls,'train',cols=5, annotation=annotation, image_filename=example['filename'], font_size_scale=2,vertical_spacing_scale=2)
    
        if((niter == 0 or niter % opts.visualize_freq == 0 )and opts.no_visualize is False):
            visualize()
        
        if niter % VALFREQ == 0:
            additional_loss, _ = predict_losses(net_input, alpha, noisy_flash, noisy_ambient, example, double_network)
            losses.update(additional_loss)
            [logger.addScalar(float(v.numpy()),k) for k, v in losses.items()]

        #log losses
        for k, v in losses.items():
            logger.addScalar(float(v.numpy()), k)
        losses_str = ', '.join('%s_%.07f'%(k, float(v.numpy())) for k, v in losses.items())
        print(datetime.now(), ' lr: ', float(opt.learning_rate.numpy()), ' iter: ',niter, losses_str, ' alpha %0.4f' % float(example['alpha'].numpy()))
        logger.takeStep()

    for data in dataset.iterator:
        net_input, alpha, noisy_flash, noisy_ambient = prepare_input(data,clamp=logger.opts.clamp_dataset, std_input=logger.opts.std_input)
        if(niter > MAXITER):
            break
        niter += 1
        if(opts.overfit):
            # model.weights['alpha_weight'] = model.weights['alpha_weight'] * 0
            #if example does not exist, save it, otherwise load it
            suffix=''
            suffix += 'nostd' if opts.std_input == False else ''
            suffix += 'clamp' if opts.clamp_dataset == True else ''
            overfit_example_gt_data_fn = './overfit_example_data_gt%s.pkl' % suffix
            overfit_example_noisy_data_fn = './overfit_example_data_noisy%s.pkl' % suffix
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
                denoise = None
                if(opts.double_network):
                    denoise = tf.stop_gradient(deepfnf_model.forward(net_input))
                data_noisy = {'net_input':net_input, 'alpha':alpha, 'noisy_flash':noisy_flash, 'noisy_ambient':noisy_ambient, 'niter':niter, 'denoise':denoise}
                data_gt = data
                logger.dump_pickle(overfit_example_gt_data_fn, data)
                logger.dump_pickle(overfit_example_noisy_data_fn, data_noisy)
            data.update(data_noisy)
            data.update(data_gt)
            net_input, alpha, noisy_flash, noisy_ambient = prepare_input(data)
            for _ in range(int(MAXITER)):
                niter += 1
                training_iterate(net_input, alpha, noisy_flash, noisy_ambient, niter, data, logger.opts.double_network)
        else:
            # gradient_validation(net_input, alpha, noisy_flash, noisy_ambient)
            training_iterate(net_input, alpha, noisy_flash, noisy_ambient, niter, data, logger.opts.double_network)

            
store = {}
opt.save_own_variables(store)
fn1, fn2 = logger.save_params(model.weights, {'configs':opt.get_config(), 'variables':store},niter)
print("Saving model to " + fn1 + " and " + fn2)
