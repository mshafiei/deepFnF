#!/usr/bin/env python3
from arguments_deepfnf import parse_arguments_deepfnf
parser = parse_arguments_deepfnf()
opts = parser.parse_args()
import tensorflow.compat.v1 as tf
if(opts.mode == "train"):
    tf.disable_v2_behavior()
    tf.disable_eager_execution()
else:
    tf.enable_eager_execution()
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os
import argparse

import utils.np_utils as npu
import numpy as np
from test import test, test_idx_model_stats
import net_ksz3
import net
from net_laplacian_combine import Net as netLaplacianCombine
# from net_llf import Net as netLLF
from net_fft_combine import Net as netFFTCombine
from net_flash_image import Net as netFlash
from net_fft import Net as netFFT
from net_laplacian_combine_pixelwise import Net as netLaplacianCombinePixelWise
from net_no_scalemap import Net as NetNoScaleMap
from net_grad import Net as NetGrad
from net_slim import Net as NetSlim
import unet
import utils.utils as ut
import utils.tf_utils as tfu
from utils.dataset import Dataset
import lpips_tf
import cvgutils.Viz as Viz
# import tensorflow.contrib.eager as tfe
import time


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

def CreateNetwork(opts):
    
    if(opts.model == 'net_flash_image'):
        model = netFlash()
    # elif(opts.model == 'deepfnf_llf'):
    #     model = netLLF(opts.llf_alpha, opts.llf_beta, opts.llf_levels, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
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
        model = net.Net(ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor)
    elif(opts.model == 'deepfnf-slim'):
        model = NetSlim(ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor)
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
    g = tf.Graph()
    run_meta = tf.RunMetadata()
    Gs = [g, run_meta]
    with g.as_default():
        model = CreateNetwork(opts)
        print("Restoring model from " + opts.weight_file)
        model = load_net(opts.weight_file, model)
        stats = test_idx_model_stats(opts.TESTPATH,0,0,{},{},logger,model,{},{},Gs)
        logger.dumpDictJson(stats,'model_stats','test')

    model = CreateNetwork(opts)
    model = load_net(opts.weight_file, model)
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

# Check for saved weights & optimizer states
msave = ut.ckpter(wts + '/iter_*.model.npz')
ssave = ut.ckpter(wts + '/iter_*.state.npz')
ut.logopen(wts + '/train.log')
niter = msave.iter


with tf.device('/cpu:0'):
    global_step = tf.placeholder(dtype=tf.int64, shape=[])

    # Set up optimizer
    lr = tf.placeholder(shape=[], dtype=tf.float32)
    opt = tf.train.AdamOptimizer(lr)

    # Data loading setup
    dataset = Dataset(TLIST, VPATH, bsz=BSZ, psz=IMSZ,
                      ngpus=opts.ngpus, nthreads=4 * opts.ngpus,jitter=opts.displacement,min_scale=opts.min_scale,max_scale=opts.max_scale,theta=opts.max_rotate)

    # Calculate grads for each tower
    tower_grads = []
    tower_loss, tower_lvals = [], []
    for i in range(opts.ngpus):
        with tf.device('/gpu:%d' % i):
            example = dataset.batches[i]

            alpha = example['alpha'][:, None, None, None]
            dimmed_ambient, _ = tfu.dim_image(
                example['ambient'], alpha=alpha)
            dimmed_warped_ambient, _ = tfu.dim_image(
                example['warped_ambient'], alpha=alpha)

            # Make the flash brighter by increasing the brightness of the
            # flash-only image.
            flash = example['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
            warped_flash = example['warped_flash_only'] * \
                ut.FLASH_STRENGTH + dimmed_warped_ambient

            sig_read = example['sig_read'][:, None, None, None]
            sig_shot = example['sig_shot'][:, None, None, None]
            noisy_ambient, _, _ = tfu.add_read_shot_noise(
                dimmed_ambient, sig_read=sig_read, sig_shot=sig_shot)
            noisy_flash, _, _ = tfu.add_read_shot_noise(
                warped_flash, sig_read=sig_read, sig_shot=sig_shot)

            noisy = tf.concat([noisy_ambient, noisy_flash], axis=-1)
            noise_std = tfu.estimate_std(noisy, sig_read, sig_shot)
            net_input = tf.concat([noisy, noise_std], axis=-1)

            if(opts.model == "deepfnf_llf" or opts.model == "deepfnf_combine_laplacian_pixelwise"):
                denoise = model.forward(net_input,alpha) / alpha
            else:
                denoise = model.forward(net_input) / alpha

            denoise = tfu.camera_to_rgb(
                denoise, example['color_matrix'], example['adapt_matrix'])
            denoise
            ambient = tfu.camera_to_rgb(
                example['ambient'],
                example['color_matrix'], example['adapt_matrix'])

            # Loss
            l2_loss = tfu.l2_loss(denoise, ambient)
            gradient_loss = tfu.gradient_loss(denoise, ambient)
            psnr = tfu.get_psnr(denoise, ambient)
            if(opts.lpips):
                lpips_loss = tf.reduce_mean(lpips_tf.lpips(denoise, ambient, model='net-lin', net='alex'))
            else:
                lpips_loss = 0
            if(opts.wlpips):
                # wo_reduction = lpips_tf.lpips(denoise, ambient, model='net-lin', net='alex') * tf.square(denoise - ambient)
                wlpips_loss = tf.reduce_mean(lpips_tf.lpips(denoise, ambient, model='net-lin', net='alex') * tf.square(denoise - ambient))
            else:
                wlpips_loss = 0
            loss = l2_loss + gradient_loss + lpips_loss + wlpips_loss
            lvals = [loss, l2_loss, gradient_loss, psnr, lpips_loss, wlpips_loss]
            lnms = ['loss', 'l2_pixel', 'l1_gradient', 'psnr', "lpips_loss","wlpips_loss"]

            tower_loss.append(loss)
            tower_lvals.append(lvals)

            grads = opt.compute_gradients(
                loss, var_list=list(model.weights.values()))
            tower_grads.append(grads)

    # Update step
    with tf.device('/cpu:0'):
        grads = ut.average_gradients(tower_grads)
        tStep = opt.apply_gradients(grads)

        # Aggregate losses for output
        loss = tf.reduce_mean(tf.stack(tower_loss, axis=0), axis=0)
        lvals = tf.reduce_mean(tf.stack(tower_lvals, axis=0), axis=0)
        tnms = [l + '.t' for l in lnms]
        vnms = [l + '.v' for l in lnms]

#########################################################################
# Start TF session
sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
dataset.init_handles(sess)

#########################################################################

mfn = wts + "/model.npz"
sfn = wts + "/state.npz"

ut.mprint("Saving model to " + mfn)
ut.saveNet(mfn, model.weights, sess)
ut.mprint("Saving state to " + sfn)
ut.saveAdam(sfn, opt, model.weights, sess)
ut.mprint("Done!")
msave.clean(every=SAVEFREQ, last=1)
ssave.clean(every=SAVEFREQ, last=1)

# Load saved weights if any
if niter > 0:
    mfn = wts + "/iter_%06d.model.npz" % niter
    sfn = wts + "/iter_%06d.state.npz" % niter

    ut.mprint("Restoring model from " + mfn)
    ut.loadNet(mfn, model.weights, sess)
    ut.mprint("Restoring state from " + sfn)
    ut.loadAdam(sfn, opt, model.weights, sess)
    ut.mprint("Done!")

#########################################################################
# Main Training loop

stop = False
ut.mprint("Starting from Iteration %d" % niter)
dataset.swap_train(sess)
summary = '\n'.join(['%s %s' % (i, model.weights[i].shape) for i in model.weights] + ['total parameter count = %i' % np.sum([np.prod(model.weights[i].shape) for i in model.weights]) ])
logger.addString(summary,'model_summary')
print("===================== Model summary =====================")
print(summary)
print("===================== Model summary =====================")


# mfn = wts + "/model.npz"
# sfn = wts + "/state.npz"

# ut.mprint("Saving model to " + mfn)
# ut.saveNet(mfn, model.weights, sess)
# ut.mprint("Saving state to " + sfn)
# ut.saveAdam(sfn, opt, model.weights, sess)
# ut.mprint("Done!")
# msave.clean(every=SAVEFREQ, last=1)
# ssave.clean(every=SAVEFREQ, last=1)

while niter < MAXITER and not ut.stop:

    # Validate model every so often
    if niter % VALFREQ == 0 and niter != 0:
        ut.mprint("Validating model")
        dataset.swap_val(sess)
        vloss = []
        while True:
            try:
                outs = sess.run(lvals, feed_dict={global_step: niter})
                vloss.append(outs)
            except tf.errors.OutOfRangeError:
                break
        vloss = np.mean(np.stack(vloss, axis=0), axis=0)
        ut.vprint(niter, vnms, vloss.tolist())

        dataset.swap_train(sess)

    # Run training step and print losses
    if niter % 100 == 0:
        outs = sess.run(
            [lvals, tStep],
            feed_dict={lr: get_lr(niter), global_step: niter}
        )
        ut.vprint(niter, tnms, outs[0].tolist())
        ut.vprint(niter, ['lr'], [get_lr(niter)])
        if(type(outs[0].tolist()[0]) == float):
            logger.addScalar(outs[0].tolist()[0],'loss','val')
        if(type(outs[0].tolist()[-1]) == float):
            logger.addScalar(outs[0].tolist()[0],'psnr','val')
    else:
        start = time.time()
        outs = sess.run(
            [loss, tStep],
            feed_dict={lr: get_lr(niter), global_step: niter}
        )
        end = time.time()
        ut.vprint(niter, ['loss.t', 'running time (ms)'], [outs[0], (end - start)*1000])
        if(type(outs[0]) == np.float32):
            logger.addScalar(outs[0],'loss')
    logger.takeStep()

    niter = niter + opts.ngpus

    # Save model weights if needed
    if SAVEFREQ > 0 and niter % SAVEFREQ == 0:
        mfn = wts + "/iter_%06d.model.npz" % niter
        sfn = wts + "/iter_%06d.state.npz" % niter

        ut.mprint("Saving model to " + mfn)
        ut.saveNet(mfn, model.weights, sess)
        ut.mprint("Saving state to " + sfn)
        ut.saveAdam(sfn, opt, model.weights, sess)
        ut.mprint("Done!")
        msave.clean(every=SAVEFREQ, last=1)
        ssave.clean(every=SAVEFREQ, last=1)


# Save last
if msave.iter < niter:
    mfn = wts + "/iter_%06d.model.npz" % niter
    sfn = wts + "/iter_%06d.state.npz" % niter

    ut.mprint("Saving model to " + mfn)
    ut.saveNet(mfn, model.weights, sess)
    ut.mprint("Saving state to " + sfn)
    ut.saveAdam(sfn, opt, model.weights, sess)
    ut.mprint("Done!")
    msave.clean(every=SAVEFREQ, last=1)
    ssave.clean(every=SAVEFREQ, last=1)