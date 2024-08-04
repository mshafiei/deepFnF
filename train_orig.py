#!/usr/bin/env python3
from arguments_deepfnf import parse_arguments_deepfnf
parser = parse_arguments_deepfnf()
opts = parser.parse_args()
import tensorflow as tf
# device = tf.config.list_physical_devices('GPU')[0]
# tf.config.experimental.set_memory_growth(device, True)
# if(opts.mode == "train"):
#     # tf.disable_v2_behavior()
#     # tf.enable_eager_execution()
#     physical_devices = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# else:
#     tf.enable_eager_execution()
#     physical_devices = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os
import argparse

import utils.np_utils as npu
import numpy as np
from test import test
import net_ksz3
import net
import net_cheap as netCheap
from net_laplacian_combine import Net as netLaplacianCombine
from net_fft_combine import Net as netFFTCombine
from net_flash_image import Net as netFlash
from net_fft import Net as netFFT
from net_laplacian_combine_pixelwise import Net as netLaplacianCombinePixelWise
from net_no_scalemap import Net as NetNoScaleMap
from net_grad import Net as NetGrad
from net_slim import Net as NetSlim
import unet
import unet_llf
import utils.utils as ut
import utils.tf_utils as tfu
from utils.dataset_prefetch import TrainSet as TrainSet_prefetch
from utils.dataset_prefetch_nthreads import TrainSet as TrainSet_prefetch_nthread
from utils.dataset import Dataset
from utils.dataset_filelock import TrainSet as Trainset_filelock
import cvgutils.Viz as Viz
import time
from tensorflow.python.profiler import profiler_v2 as profiler
from lpips_tf2.models_tensorflow.lpips_tensorflow import perceptual_model, linear_model, learned_perceptual_metric_model
# tf.config.run_functions_eagerly(True)

image_size=448
ckpt_dir = './lpips_tf2/weights/keras'
vgg_ckpt_fn = os.path.join(ckpt_dir, 'vgg', 'exported.weights.h5')
lin_ckpt_fn = os.path.join(ckpt_dir, 'lin', 'exported.weights.h5')
lpips = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)

# profiler.warmup()


if(opts.use_gpu):
    os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ':' + '/home/mohammad/Projects/Halide/python_bindings/apps_gpu/'
    os.environ["PATH"] = os.environ["PATH"] + ':' + '/home/mohammad/Projects/Halide/python_bindings/apps_gpu/'
else:
    os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ':' + '/home/mohammad/Projects/Halide/python_bindings/apps_cpu/'
    os.environ["PATH"] = os.environ["PATH"] + ':' + '/home/mohammad/Projects/Halide/python_bindings/apps_cpu/'

logger = Viz.logger(opts,opts.__dict__)
_, weight_dir = logger.path_parse('train')
opts.weight_file = os.path.join(weight_dir,opts.weight_file)

# profiler_path = os.path.join(weight_dir,'profiler')
# if(not os.path.exists(profiler_path)):
#     os.makedirs(profiler_path)
# profiler.start(logdir=profiler_path)


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

def CreateNetwork(opts):
    
    if(opts.model == 'net_flash_image'):
        model = netFlash()
    elif(opts.model == 'deepfnf_llf'):
        from net_llf_tf2 import Net as netLLF
        model = netLLF(opts.llf_alpha, opts.llf_beta, opts.llf_levels, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_llf_diffable'):
        from net_llf_tf2_diffable import Net as netLLF
        model = netLLF(opts.llf_alpha, opts.llf_beta, opts.llf_levels, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_combine_laplacian'):
        model = netLaplacianCombine(opts.sigmoid_offset, opts.sigmoid_intensity, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_combine_laplacian_pixelwise'):
        model = netLaplacianCombinePixelWise(opts.n_pyramid_levels, num_basis=opts.num_basis, ksz=opts.ksz, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_combine_fft'):
        model = netFFTCombine(opts.sigmoid_offset, opts.sigmoid_intensity, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_fft'):
        model = netFFT(ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_grad'):
        model = NetGrad(ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor)
    elif(opts.model == 'deepfnf' and (not opts.scalemap)):
        model = NetNoScaleMap(ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor)
    elif(opts.model == 'deepfnf' and opts.ksz == 3):
        model = net_ksz3.Net(ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor)
    elif(opts.model == 'deepfnf'):
        # model = netCheap.Net(ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor)
        model = net.Net(ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor)
    elif(opts.model == 'deepfnf-slim'):
        model = NetSlim(ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor)
    elif(opts.model == 'unet_llf'):
        model = unet_llf.Net(opts.llf_alpha, opts.llf_beta, opts.llf_levels, ksz=opts.ksz, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'unet'):
        model = unet.Net(ksz=opts.ksz, burst_length=2,channels_count_factor=opts.channels_count_factor)
    return model

def load_net(fn, model):
    if(hasattr(model,'weights')):
        wts = np.load(fn)
        for k, v in wts.items():
            model.weights[k] = tf.Variable(v)
    else:
        print('Model does not have weights')
    return model

if(opts.mode == 'test'):
    model = CreateNetwork(opts)
    params = logger.load_params()
    if (params is not None) and ('params' in params.keys()):
        model.weights = params['params']
    test(model, opts.weight_file, opts.TESTPATH,logger)
    exit(0)
else:
    model = CreateNetwork(opts)
def get_lr(niter):
    if niter < DROP[0]:
        return LR
    elif niter >= DROP[0] and niter < DROP[1]:
        return LR / np.sqrt(10.)
    else:
        return LR / 10.

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
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    
with tf.device('/gpu:0'):
    niter = 0
    # Set up optimizer
    # lr = tf.placeholder(shape=[], dtype=tf.float32)
    # lr = tf.keras.Input(name="lr", shape=(), dtype=tf.dtypes.float32)
    # opt = tf.train.AdamOptimizer(lr)
    # opt = tf.optimizers.Adam(LR)
    
    # Data loading setup


# # Calculate grads for each tower
#     @tf.function
#     def train_step(example):
#         alpha = 1
#         scale=5
#         noisy = tf.concat([example['ambient']*scale, example['flash_only']*scale], axis=-1)
#         net_input = tf.concat([noisy, noisy], axis=-1)

#         with tf.GradientTape() as tape:
#             denoise = model.forward(net_input,alpha)    
#             loss = tfu.l2_loss(denoise, example['ambient']*scale)
#         opt.minimize(loss, list(model.weights.values()), tape=tape)
#         return loss, tfu.l2_loss(example['ambient']*scale, example['flash_only']*scale)
#load parameters
    params = logger.load_params()
    if(params is not None):
        niter = params['idx']
        model.weights = params['params']
        serialized_dict = {'module':'keras.optimizers', 'class_name':'Adam', 'config':params['state']['configs'], 'registered_name':None}
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
    def val_step(example):
        net_input, alpha, noisy_flash, noisy_ambient = prepare_input(example)
        if(opts.model == "deepfnf_llf" or opts.model == "deepfnf_combine_laplacian_pixelwise"):
            denoise = model.forward(net_input,alpha) / alpha
        else:
            denoise = model.forward(net_input) / alpha

        denoise = tfu.camera_to_rgb(
            denoise, example['color_matrix'], example['adapt_matrix'])
        
        ambient = tfu.camera_to_rgb(
            example['ambient'],
            example['color_matrix'], example['adapt_matrix'])
        
        noisy_flash = tfu.camera_to_rgb(
            noisy_flash,
            example['color_matrix'], example['adapt_matrix'])
        noisy_ambient = tfu.camera_to_rgb(
            noisy_ambient / alpha,
            example['color_matrix'], example['adapt_matrix'])
        return denoise, ambient, noisy_flash, noisy_ambient
        

    @tf.function
    def train_step(example):
        net_input, alpha, _, _ = prepare_input(example)
# 0.013875767
        # psnr = tfu.get_psnr(denoise, ambient)
        # if(opts.lpips):
        #     lpips_loss = tf.reduce_mean(lpips_tf.lpips(denoise, ambient, model='net-lin', net='alex'))
        # else:
        #     lpips_loss = 0
        # if(opts.wlpips):
        #     # wo_reduction = lpips_tf.lpips(denoise, ambient, model='net-lin', net='alex') * tf.square(denoise - ambient)
        #     wlpips_loss = tf.reduce_mean(lpips_tf.lpips(denoise, ambient, model='net-lin', net='alex') * tf.square(denoise - ambient))
        # else:
        #     wlpips_loss = 0
        # @tf.function
        # def loss_function(x):
        #     denoise = x[0]
        #     ambient = x[1]
            # return tfu.l2_loss(denoise, ambient) + tfu.gradient_loss(denoise, ambient)
        with tf.GradientTape() as tape:
            if(opts.model == "deepfnf_llf" or opts.model == "deepfnf_llf_diffable" or opts.model == "deepfnf_combine_laplacian_pixelwise"):
                denoise = model.forward(net_input,alpha) / alpha
            else:
                denoise = model.forward(net_input) / alpha

            denoise = tfu.camera_to_rgb(
                denoise, example['color_matrix'], example['adapt_matrix'])
            
            ambient = tfu.camera_to_rgb(
                example['ambient'],
                example['color_matrix'], example['adapt_matrix'])
            # Loss
            l2_loss = tfu.l2_loss(denoise, ambient)
            gradient_loss = tfu.gradient_loss(denoise, ambient)
            # lpips_loss = tf.stop_gradient(lpips([denoise, ambient]))
            loss = l2_loss + gradient_loss# + opts.lpips * lpips_loss[0]

        gradients = tape.gradient(loss, model.weights.values())
        opt.apply_gradients(zip(gradients,model.weights.values()))
        # opt.minimize(loss, model.weights.values(), tape=tape)
        return loss
# Saving model to logs-grid/deepfnf-0085-pixw/train/params/params_10.pickle and logs-grid/deepfnf-0085-pixw/train/params/latest_parameters.pickle with loss  0.14356029
# dumping params  tf.Tensor(-0.050535727, shape=(), dtype=float32)
    # dataset.iterator = dataset.iterator.repeat(10000)

    def training_iterate(example, niter):
        # print('alpha ', example['alpha'])
        loss = train_step(example)
        # logger.addScalar(loss.numpy(),'loss')
        print('iter ',niter, ' loss ', loss.numpy())
        # Save model weights if needed
        if SAVEFREQ > 0 and niter % SAVEFREQ == 0:
            store = {}
            opt.save_own_variables(store)
            fn1, fn2 = logger.save_params(model.weights, {'configs':opt.get_config(), 'variables':store},niter)
            print("Saving model to " + fn1 + " and " + fn2 +" with loss ",loss.numpy())
            print('dumping params ',model.weights['down2_1_w'][0,0,0,0])
        if VALFREQ > 0 and niter % VALFREQ == 0:
            # draw example['ambient'], denoised image, flash image, absolute error
            denoisednp, ambientnp, flashnp, noisy = val_step(example)
            logger.addImage({'flash':flashnp.numpy()[0], 'ambient':ambientnp.numpy()[0], 'denoised':denoisednp.numpy()[0], 'noisy':noisy.numpy()[0]},{'flash':'Flash','ambient':'Ambient','denoised':'Denoise','noisy':'Noisy'},'train')
        logger.takeStep()
    
    # fn1, fn2 = logger.save_params(model.weights, opt.get_config(),niter)
    if(opts.dataset_model == 'prefetch_nthread'):
        for i in range(int(MAXITER)):
            niter += 1
            example = dataset.get_next()
            training_iterate(example, niter)
            a = tf.config.experimental.get_memory_usage(device='GPU:0')
            b = tf.config.experimental.get_memory_info('GPU:0')
            print('a ', a, ' b ', b)
            del example
    else:
        for example in dataset.iterator:
            if(niter > MAXITER):
                break
            niter += 1
            training_iterate(example, niter)
        # log losses
        # visualize training
        # visualize validation
        # save weights
        # ut.vprint(niter, 'loss.t', loss)
        # logger.addScalar(loss,'loss')
        # logger.takeStep()
        # loss = l2_loss + gradient_loss + lpips_loss + wlpips_loss
        # lvals = [loss, l2_loss, gradient_loss, psnr, lpips_loss, wlpips_loss]
        # lnms = ['loss', 'l2_pixel', 'l1_gradient', 'psnr', "lpips_loss","wlpips_loss"]

        # tower_loss.append(loss)
        # tower_lvals.append(lvals)

        # grads = opt.compute_gradients(
        #     loss_function, var_list=list())
        # tower_grads.append(grads)
            
# profiler.stop()
store = {}
opt.save_own_variables(store)
fn1, fn2 = logger.save_params(model.weights, {'configs':opt.get_config(), 'variables':store},niter)
print("Saving model to " + fn1 + " and " + fn2)
