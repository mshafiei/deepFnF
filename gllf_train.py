#!/usr/bin/env python3
from arguments_deepfnf import parse_arguments_deepfnf
parser = parse_arguments_deepfnf()
opts = parser.parse_args()
import tensorflow as tf
import os
import argparse
import timeit
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
from gllf.gllf_utils import prepare_input
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
LR = opts.learning_rate
DROP = (1.1e6, 1.25e6) # Learning rate drop

MAXITER = opts.max_iter
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
    if(deepfnf_model != None):
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
    def val_step(net_input, alpha, noisy_flash, noisy_ambient, example, double_network=True):
        outputs = edict()
        model_input = edict()
        if(double_network):
            outputs.denoise = tf.stop_gradient(deepfnf_model.forward(net_input))
            outputs.deepfnf_scaled = tfu.camera_to_rgb(
            outputs.denoise / alpha, example['color_matrix'], example['adapt_matrix'])
            outputs.net_ft_input = tf.concat((net_input, outputs.denoise), axis=-1)
            model_input.deepfnf_scaled = outputs.deepfnf_scaled
            model_input.denoise = outputs.denoise
        else:
            outputs.net_ft_input = net_input
        
        outputs.noisy_ambient_scaled = tfu.camera_to_rgb(
            noisy_ambient / alpha, example['color_matrix'], example['adapt_matrix'])
        
        outputs.noisy_flash_scaled = tfu.camera_to_rgb(
            noisy_flash, example['color_matrix'], example['adapt_matrix'])

        outputs.noisy_flash = tfu.camera_to_rgb(
            noisy_flash, example['color_matrix'], example['adapt_matrix'])
        
        outputs.ambient_scaled= tfu.camera_to_rgb(
            example['ambient'],
            example['color_matrix'], example['adapt_matrix'])
        
        model_input.update(edict(noisy_ambient_scaled=outputs.noisy_ambient_scaled,
                                 noisy_flash=outputs.noisy_flash, 
                                 net_ft_input=outputs.net_ft_input, 
                                 noisy_flash_scaled=outputs.noisy_flash_scaled,
                                 color_matrix=example['color_matrix'],
                                 adapt_matrix=example['adapt_matrix'],
                                 alpha=example['alpha']))
        output_dict=edict(model_input)
        model_input.noflash_wb_fn = lambda img: tfu.camera_to_rgb(
            img / alpha, example['color_matrix'], example['adapt_matrix'])
        model_input.flash_wb_fn = lambda img: tfu.camera_to_rgb(
            img, example['color_matrix'], example['adapt_matrix'])

        outputs.model_output = model.forward(model_input)
        if(logger.opts.normalize_before_gllf):
            outputs.model_output.output = tfu.camera_to_rgb(
            outputs.model_output.output / alpha, example['color_matrix'], example['adapt_matrix'])
        return outputs, output_dict


    
    def predict_losses(net_input, alpha, noisy_flash, noisy_ambient, example, validation=True, double_network=True):
        net_inp = edict()
        losses = edict()
        net_inp.noisy_ambient_scaled =  tfu.camera_to_rgb(
            noisy_ambient / alpha, example['color_matrix'], example['adapt_matrix'])
        
        net_inp.noisy_flash = tfu.camera_to_rgb(
            noisy_flash, example['color_matrix'], example['adapt_matrix'])
        
        net_inp.noisy_flash_scaled = tfu.camera_to_rgb(
            noisy_flash, example['color_matrix'], example['adapt_matrix'])
        
        net_inp.ambient_scaled= tfu.camera_to_rgb(
            example['ambient'],
            example['color_matrix'], example['adapt_matrix'])
        
        

        if(double_network):
            net_inp.denoise = tf.stop_gradient(deepfnf_model.forward(net_input))
            net_ft_input = tf.concat((net_input, net_inp.denoise), axis=-1)
            net_inp.deepfnf_scaled = tfu.camera_to_rgb(
            net_inp.denoise / alpha, example['color_matrix'], example['adapt_matrix'])
            psnr_deepfnf = tfu.get_psnr(tf.maximum(net_inp.deepfnf_scaled,0), tf.maximum(net_inp.ambient_scaled,0))
            losses.psnr_deepfnf = psnr_deepfnf
            if(validation):
                wlpips_deepfnf = wlpips([net_inp.deepfnf_scaled, net_inp.ambient_scaled])
                lpips_deepfnf = lpips([net_inp.deepfnf_scaled, net_inp.ambient_scaled])
                losses.wlpips_deepfnf = wlpips_deepfnf
                losses.lpips_deepfnf = lpips_deepfnf

        else:
            net_ft_input = net_input

        net_inp.net_ft_input = net_ft_input
        net_inp.color_matrix = example['color_matrix']
        net_inp.adapt_matrix = example['adapt_matrix']
        net_inp.alpha = example['alpha']
        output = edict(net_inp)
        net_inp.noflash_wb_fn = lambda img: tfu.camera_to_rgb(
            img / alpha, example['color_matrix'], example['adapt_matrix'])
        net_inp.flash_wb_fn = lambda img: tfu.camera_to_rgb(
            img, example['color_matrix'], example['adapt_matrix'])
        double_deepfnf = model.forward(net_inp)
        if(logger.opts.normalize_before_gllf):
            double_deepfnf.output = tfu.camera_to_rgb(
            double_deepfnf.output / alpha, example['color_matrix'], example['adapt_matrix'])

        if(validation):
            wlpips_refined = wlpips([double_deepfnf.output, output.ambient_scaled])
            lpips_refined = lpips([double_deepfnf.output, output.ambient_scaled])
        
            losses.wlpips_refined = wlpips_refined
            losses.lpips_refined = lpips_refined

        psnr_refined = tfu.get_psnr(double_deepfnf.output, output.ambient_scaled)
        losses.psnr_refined = psnr_refined
        return losses, output
    

    def eval_latency(fn,fn_name,timing_iterations=3):
        t = timeit.Timer(fn, setup=fn)
        avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
        print("function %s took %fms" % (fn_name,(avg_time_sec * 1e3)))

    @tf.function
    def train_step(net_input, alpha, noisy_flash, noisy_ambient, example, alpha_coeffs, double_network=True):
        model_inputs = edict()
        
        model_inputs.noisy_ambient_scaled =  tfu.camera_to_rgb(noisy_ambient / alpha,
        example['color_matrix'], example['adapt_matrix'])

        model_inputs.noisy_flash = noisy_flash

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
        model_inputs.alpha = example['alpha']
        model_inputs.noflash_wb_fn = lambda img: tfu.camera_to_rgb(
            img / alpha, example['color_matrix'], example['adapt_matrix'])
        model_inputs.flash_wb_fn = lambda img: tfu.camera_to_rgb(
            img, example['color_matrix'], example['adapt_matrix'])
        with tf.GradientTape() as tape:
            double_deepfnf = model.forward(model_inputs)
            if(logger.opts.normalize_before_gllf):
                double_deepfnf.output = tfu.camera_to_rgb(double_deepfnf.output / alpha,
                        example['color_matrix'], example['adapt_matrix'])
            # double_deepfnf.output = tfu.camera_to_rgb(double_deepfnf.output,
            # example['color_matrix'], example['adapt_matrix'])
            # Loss
            
            l2_loss = tf.convert_to_tensor(0.0) if opts.l2 == 0 else tfu.l2_loss(double_deepfnf.output, tf.maximum(model_inputs.ambient_scaled,0))
            gradient_loss = tf.convert_to_tensor(0.0) if opts.grad == 0 else tfu.gradient_loss(double_deepfnf.output, tf.clip_by_value(model_inputs.ambient_scaled,0,1))
            wlpips_loss = tf.convert_to_tensor(0.0) if opts.wlpips == 0 else wlpips([double_deepfnf.output, tf.clip_by_value(model_inputs.ambient_scaled,0,1)])[0]
            lpips_loss = tf.convert_to_tensor(0.0) if opts.lpips == 0 else lpips([double_deepfnf.output, tf.clip_by_value(model_inputs.ambient_scaled,0,1)])[0]
            
            # lpips_loss = tf.stop_gradient(lpips([denoise, ambient]))
            loss = opts.l2 * l2_loss + opts.grad * gradient_loss + opts.lpips * lpips_loss + opts.wlpips * wlpips_loss
        
        if('llf_input' in double_deepfnf):
            double_deepfnf.llf_input = tfu.camera_to_rgb(double_deepfnf.llf_input,
                example['color_matrix'], example['adapt_matrix'])
        if('llf_guide' in double_deepfnf):
            double_deepfnf.llf_guide = tfu.camera_to_rgb(double_deepfnf.llf_guide,
                example['color_matrix'], example['adapt_matrix'])
        
        gradients = tape.gradient(loss, model.weights.values())
        # keys = list(model.weights.keys())
        # vals = list(model.weights.values())
        # for idx in range(len(vals)):
        #     if('alpha' in keys[idx]):
        #         gradients[idx] = alpha_coeffs
        
        opt.apply_gradients(zip(gradients,model.weights.values()))
        psnr_metric = tfu.get_psnr(double_deepfnf.output, model_inputs.ambient_scaled)
        losses = {'loss':loss, 'l2_loss':l2_loss, 'gradient_loss':gradient_loss,
        'wlpips_loss':opts.wlpips * wlpips_loss, 'lpips_loss':opts.lpips * lpips_loss, 'psnr':psnr_metric}
        return losses

    def eval_latency_prepare(net_input, alpha, noisy_flash, noisy_ambient, example,double_network=True):
        model_inputs = edict()
        
        model_inputs.noisy_ambient_scaled =  tfu.camera_to_rgb(noisy_ambient / alpha,
        example['color_matrix'], example['adapt_matrix'])

        model_inputs.noisy_flash = noisy_flash

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
        model_inputs.alpha = example['alpha']

        model_inputs.noflash_wb_fn = lambda img: tfu.camera_to_rgb(
            img / alpha, example['color_matrix'], example['adapt_matrix'])
        model_inputs.flash_wb_fn = lambda img: tfu.camera_to_rgb(
            img, example['color_matrix'], example['adapt_matrix'])

        fn = lambda: model.forward(model_inputs)
        eval_latency(fn, opts.model)
        
    def training_iterate(net_input, alpha, noisy_flash, noisy_ambient, niter, example, double_network):
        if(opts.eval_latency and niter <= 3):
            eval_latency_prepare(net_input, alpha, noisy_flash, noisy_ambient, example, double_network)
        alpha_coeffs = 1.
        # if(niter < 500):
        #     alpha_coeffs = 0.1
        # elif(niter < 1000):
        #     alpha_coeffs = 0.5
        # else:
        #     alpha_coeffs = 1
        losses = train_step(net_input, alpha, noisy_flash, noisy_ambient, example, alpha_coeffs, double_network)

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
            additional_loss, deepfnf_out_dict = predict_losses(net_input, alpha, noisy_flash, noisy_ambient, example, validation=True, double_network=double_network)
            deepfnf_out = deepfnf_out_dict
            losses.update(additional_loss)
            # draw example['ambient'], denoised image, flash image, absolute error
            val_output_dict, model_inputs_dict = val_step(net_input, alpha, noisy_flash, noisy_ambient, example, double_network=double_network)
            val_output, model_inputs = val_output_dict, model_inputs_dict
            if(double_network):
                annotation_deepfnf = '<br>PSNR:%.3f<br>LPIPS:%.3f<br>WLPIPS:%.3f'%(additional_loss['psnr_deepfnf'],additional_loss['lpips_deepfnf'],additional_loss['wlpips_deepfnf'])
            annotation_ours = '<br>PSNR:%.3f<br>LPIPS:%.3f<br>WLPIPS:%.3f'%(additional_loss['psnr_refined'],additional_loss['lpips_refined'],additional_loss['wlpips_refined'])
            annotation = {'flash':None,'noisy':None,'ambient':None,'denoised_gllf':annotation_ours,'alpha_map':None}
            if(double_network):
                annotation.update({'denoised_deepfnf':annotation_deepfnf})
            # exposure = 4 if opts.llf_sigma == 0 else 0
            
            images = {'flash':model_inputs.noisy_flash_scaled.numpy()[0], 'noisy':deepfnf_out.noisy_ambient_scaled.numpy()[0], 'ambient':deepfnf_out.ambient_scaled.numpy()[0], 'denoised_gllf':val_output.model_output.output}
            lbls = {'flash':'Flash','noisy':'Noisy','ambient':'Ambient','denoised_gllf':'DeepFnF+GLLF'}

            if(hasattr(model, 'visualize')):
                model_visualization = model.visualize(model_inputs)
                for model_viz in model_visualization:
                    images.update({model_viz.key:model_viz.image})
                    lbls.update({model_viz.key:model_viz.label})

            if(double_network):
                images.update({'denoised_deepfnf':deepfnf_out.deepfnf_scaled.numpy()[0]})
                lbls.update({'denoised_deepfnf':'DeepFnF'})
            for k,v in val_output.model_output.items():
                if('visualize' in k):
                    visualize_image = cv2.resize(v.image.numpy()[0], (448,448))[None,...]
                    images.update({k:visualize_image})
                    lbls.update({k:v.label})
            # if('alpha_map_h' in val_output.model_output):
            #     alpha_map_h = cv2.resize(val_output.model_output.alpha_map_h.numpy()[0], (448,448))[None,...]
            #     # gllf = gllf.numpy()[0] * 2.0**exposure
            #     annotation =  None if opts.llf_sigma == 0 else annotation
            #     alpha_h_min = tf.reduce_min(alpha_map_h)
            #     alpha_h_max = tf.reduce_max(alpha_map_h)
            #     alpha_map_h = (alpha_map_h - alpha_h_min) / (alpha_h_max - alpha_h_min)
            #     images.update({'alpha_map_h':alpha_map_h})
            #     lbls.update({'alpha_map_h':'$\\huge{\\alpha_h \\in [%.02f,%.02f]}$'%(alpha_h_min, alpha_h_max)})
            # if('alpha_map_i' in val_output.model_output):
            #     alpha_map_i = cv2.resize(val_output.model_output.alpha_map_i.numpy()[0], (448,448))[None,...]
            #     alpha_i_min = tf.reduce_min(alpha_map_i)
            #     alpha_i_max = tf.reduce_max(alpha_map_i)
            #     alpha_map_i = (alpha_map_i - alpha_i_min) / (alpha_i_max - alpha_i_min)
            #     images.update({'alpha_map_i':alpha_map_i})
            #     lbls.update({'alpha_map_i':'$\\huge{\\alpha_i \\in [%.02f,%.02f]}$'%(alpha_i_min, alpha_i_max)})
            # if("llf_guide" in val_output.model_output):
            #     images.update({'llf_guide':val_output.model_output.llf_guide.numpy()})
            #     lbls.update({'llf_guide':'I_h'})
            # if("llf_input" in val_output.model_output):
            #     images.update({'llf_input':val_output.model_output.llf_input.numpy()})
            #     lbls.update({'llf_input':'I_i'})

            if('filename' in example.keys()):
                logger.addImage(images, lbls,'train',cols=4, annotation=annotation, image_filename=example['filename'], font_size_scale=2,vertical_spacing_scale=2)
    
        if((niter == 0 or niter % opts.visualize_freq == 0 ) and opts.no_visualize is False):
            tf.config.run_functions_eagerly(True)
            visualize()
            tf.config.run_functions_eagerly(False)
        
        if niter % VALFREQ == 0:
            additional_loss, _ = predict_losses(net_input, alpha, noisy_flash, noisy_ambient, example, validation=True, double_network=double_network)
            losses.update(additional_loss)
            [logger.addScalar(float(v.numpy()),k) for k, v in losses.items()]
        if((niter == 0) or (niter % logger.opts.print_val_freq == 0)):
            additional_loss, _ = predict_losses(net_input, alpha, noisy_flash, noisy_ambient, example, validation=False, double_network=double_network)
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
                data_gt = Viz.load_pickle(overfit_example_gt_data_fn)
                data_noisy = Viz.load_pickle(overfit_example_noisy_data_fn)
                net_input = data_noisy['net_input']
                alpha = data_noisy['alpha']
                noisy_flash = data_noisy['noisy_flash']
                noisy_ambient = data_noisy['noisy_ambient']
                # niter = data_noisy['niter']
            else:
                print('could not load example from file')
                denoise = None
                if(opts.double_network):
                    denoise = tf.stop_gradient(deepfnf_model.forward(net_input))
                data_noisy = {'net_input':net_input, 'alpha':alpha, 'noisy_flash':noisy_flash, 'noisy_ambient':noisy_ambient, 'niter':niter, 'denoise':denoise}
                data_gt = data
                Viz.dump_pickle(overfit_example_gt_data_fn, data)
                Viz.dump_pickle(overfit_example_noisy_data_fn, data_noisy)
            data.update(data_noisy)
            data.update(data_gt)
            net_input, alpha, noisy_flash, noisy_ambient = prepare_input(data,clamp=logger.opts.clamp_dataset, std_input=logger.opts.std_input)
            for _ in range(int(MAXITER)):
                training_iterate(net_input, alpha, noisy_flash, noisy_ambient, niter, data, logger.opts.double_network)
                niter += 1
        else:
            # gradient_validation(net_input, alpha, noisy_flash, noisy_ambient)
            training_iterate(net_input, alpha, noisy_flash, noisy_ambient, niter, data, logger.opts.double_network)
            niter += 1

            
store = {}
opt.save_own_variables(store)
fn1, fn2 = logger.save_params(model.weights, {'configs':opt.get_config(), 'variables':store},niter)
print("Saving model to " + fn1 + " and " + fn2)
