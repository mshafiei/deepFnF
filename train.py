#!/usr/bin/env python3

import os
import argparse

import utils.np_utils as npu
import numpy as np
import tensorflow as tf

import net_tmp
import utils.utils as ut
import utils.tf_utils as tfu
from utils.dataset import Dataset
import cvgutils.Viz as Viz

parser = argparse.ArgumentParser()
parser.add_argument('--TLIST', type=str, default='data/train_1600.txt', help='Training dataset filename')
parser.add_argument('--VPATH', type=str, default='data/valset', help='Validation dataset')
parser.add_argument('--model', type=str, default='deepfnf+fft_helmholz',choices=['deepfnf','deepfnf+fft','deepfnf+fft_grad_image','deepfnf+fft_helmholz'], help='Validation dataset')
parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
parser.add_argument('--weight_dir', type=str, default='wts', help='Weight dir')
parser.add_argument('--visualize_freq', type=int, default=5001, help='How many iterations before visualization')
parser.add_argument('--val_freq', type=int, default=10000, help='How many iterations before visualization')
parser.add_argument('--min_lmbda_phi', type=float, default=1, help='The min value of lambda phi')
parser.add_argument('--min_lmbda_psi', type=float, default=1, help='The min value of lambda psi')
parser.add_argument('--fixed_lambda', action='store_true',help='Do not change the delta value')
parser.add_argument('--max_lambda', type=float,default=0.001,help='Maximum lambda for initialization')

parser = Viz.logger.parse_arguments(parser)
opts = parser.parse_args()
logger = Viz.logger(opts,opts.__dict__)
opts = logger.opts
TLIST = opts.TLIST
VPATH = opts.VPATH
BSZ = 1
IMSZ = 448
LR = 1e-4
DROP = (1.1e6, 1.25e6) # Learning rate drop
MAXITER = 1.5e6

VALFREQ = 1e4
SAVEFREQ = 5e4

min_lph = np.log(np.exp(opts.min_lmbda_phi) - 1)
min_lps = np.log(np.exp(opts.min_lmbda_psi) - 1)
if(opts.fixed_lambda):
    lambda_phi = tf.constant(np.float32(min_lph))
    lambda_psi = tf.constant(np.float32(min_lps))
    
wts = opts.weight_dir
if not os.path.exists(wts):
    os.makedirs(wts)
if(opts.model == 'deepfnf' or opts.model == 'deepfnf+fft_grad_image'):
    outchannels = 3
elif(opts.model == 'deepfnf+fft' or opts.model == 'deepfnf+fft_helmholz'):
    outchannels = 6
model = net_tmp.Net(opts.model,outchannels,opts.min_lmbda_phi,opts.min_lmbda_psi,opts.fixed_lambda,opts.max_lambda,ksz=15, num_basis=90, burst_length=2)


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
                      ngpus=opts.ngpus, nthreads=4 * opts.ngpus)

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

            model_outpt = model.forward(net_input)

            ambient = tfu.camera_to_rgb(
                example['ambient'],
                example['color_matrix'], example['adapt_matrix'])
            noisy_scaled = tfu.camera_to_rgb(
                noisy_ambient/example['alpha'],
                example['color_matrix'], example['adapt_matrix'])
                
            if('fft' in opts.model):
                lmbda = model.weights['lmbda']
                if('deepfnf+fft' == opts.model):
                    gx = model_outpt[...,:3]
                    gy = model_outpt[...,3:]
                if('deepfnf+fft_helmholz' == opts.model):
                    if(not opts.fixed_lambda):
                        lambda_phi = model.weights['lambda_phi']
                        lambda_psi = model.weights['lambda_psi']
                    phi = model_outpt[...,:3]
                    a = model_outpt[...,3:]
                    phix = tfu.dx_tf(phi)
                    phiy = tfu.dy_tf(phi)
                    ax = tfu.dx_tf(a)
                    ay = tfu.dy_tf(a)
                    gx = tf.math.softplus(lambda_phi) * phix - tf.math.softplus(lambda_psi) * ay
                    gy = tf.math.softplus(lambda_phi) * phiy + tf.math.softplus(lambda_psi) * ax
                if('deepfnf+fft_grad_image' == opts.model):
                    gx = tfu.dx_tf(model_outpt)
                    gy = tfu.dy_tf(model_outpt)
                # screen_poisson = lambda x,y,z,w,u : tf.map_fn(tfu.screen_poisson,([x]*BSZ,y,z,w,u))
                reshape = lambda x :tf.transpose(x,[0,3,1,2])
                fft_res = tfu.screen_poisson(lmbda,reshape(noisy_ambient),reshape(gx), reshape(gy),IMSZ)
                # fft_res = gx + gy + noisy_ambient * lmbda
                fft_res = tf.transpose(fft_res,[0,2,3,1])
                dx = tfu.dx_tf(fft_res)
                dy = tfu.dy_tf(fft_res)
                denoise = tfu.camera_to_rgb(
                fft_res/alpha, example['color_matrix'], example['adapt_matrix'])
                sp_loss = tfu.l2_loss(denoise, ambient)
                psnr = tfu.get_psnr(denoise, ambient)
                

                lvals = [sp_loss, psnr]
                lnms = ['loss', 'psnr']
                loss = sp_loss
            if('deepfnf' == opts.model):# Loss
                denoise = tfu.camera_to_rgb(
                    model_outpt/alpha, example['color_matrix'], example['adapt_matrix'])
                l2_loss = tfu.l2_loss(denoise, ambient)
                gradient_loss = tfu.gradient_loss(denoise, ambient)
                psnr = tfu.get_psnr(denoise, ambient)

                loss = l2_loss + gradient_loss
                lvals = [loss, l2_loss, gradient_loss, psnr]
                lnms = ['loss', 'l2_pixel', 'l1_gradient', 'psnr']

            tower_loss.append(loss)
            tower_lvals.append(lvals)

            grads = opt.compute_gradients(
                loss, var_list=list(model.weights.values()))
            tower_grads.append(grads)

    # Update step
    with tf.device('/gpu:0'):
        grads = ut.average_gradients(tower_grads)
        tStep = opt.apply_gradients(grads)

        # Aggregate losses for output
        loss = tf.reduce_mean(tf.stack(tower_loss, axis=0), axis=0)
        lvals = tf.reduce_mean(tf.stack(tower_lvals, axis=0), axis=0)
        tnms = [l + '.t' for l in lnms]
        vnms = [l + '.v' for l in lnms]

#########################################################################
# Start TF session
config=tf.ConfigProto(
    allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.85
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
dataset.init_handles(sess)

#########################################################################
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
# sess.run(tf.math.softplus(delta))
# sess.run(tf.math.softplus(delta))
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
    vis_tf,lossdict, losspredict_tf,paramstr = {},{},{},''
    mode=None

    if niter % opts.val_freq == 0:
        outs = sess.run(
            [lvals,psnr, tStep],
            feed_dict={lr: get_lr(niter), global_step: niter}
        )
        ut.vprint(niter, tnms, outs[0].tolist())
        ut.vprint(niter, ['lr'], [get_lr(niter)])
        lossdict.update({k:v for k,v in zip(tnms,outs[0].tolist())})
        lossdict.update({'psnr':outs[1]})
        mode = 'val'
    else:
        outs = sess.run(
            [loss, psnr,tStep],
            feed_dict={lr: get_lr(niter), global_step: niter}
        )
        lossdict.update({'psnr':outs[1]})
        ut.vprint(niter, ['loss.t'], [outs[0]])
        lossdict.update({'loss.t':outs[0]})
        mode = 'train'
    if(niter % opts.visualize_freq == 0):
        if('fft' in opts.model):
            vis_tf.update({'fft_res':fft_res,'gx':gx,'gy':gy,'dx':dx,'dy':dy})
            losspredict_tf.update({'lambda':model.weights['lmbda']})
        if('helmholz' in opts.model):
            # pass
            vis_tf.update({'ax':ax,'ay':ay,'phix':phix,'phiy':phiy})
            if(opts.fixed_lambda):
                losspredict_tf.update({'lambda_phi':tf.math.softplus(lambda_phi)})
                losspredict_tf.update({'lambda_psi':tf.math.softplus(lambda_psi)})
            else:
                losspredict_tf.update({'lambda_phi':tf.math.softplus(model.weights['lambda_phi'])})
                losspredict_tf.update({'lambda_psi':tf.math.softplus(model.weights['lambda_psi'])})
        if('grad_image' in opts.model):
            vis_tf.update({'g':model_outpt})
        lossdict = sess.run(losspredict_tf) if len(losspredict_tf) > 0 else {}
        paramstr += r'\lambda%.4f~' % lossdict['lambda'] if 'lambda' in lossdict.keys() else ''
        paramstr += r'\lambda_{phi}%.4f' % lossdict['lambda_phi'] if 'lambda_phi' in lossdict.keys() else ''
        paramstr += r'\lambda_{psi}ta%.4f' % lossdict['lambda_psi'] if 'lambda_psi' in lossdict.keys() else ''
        
        fetch = {'denoise':denoise,'ambient':ambient,'noisy':noisy_scaled,'flash':noisy_flash,'alpha':example['alpha'],'color_matrix':example['color_matrix'], 'adapt_matrix':example['adapt_matrix']}
        fetch.update(vis_tf)
        fetches = sess.run(fetch)
        visouts = npu.visualize(fetches,opts)
        mtrcs_pred = npu.metrics(fetches['denoise'],fetches['ambient'])
        mtrcs_noisy = npu.metrics(fetches['noisy'],fetches['ambient'])
        logger.addImage(visouts,npu.labels(mtrcs_pred,mtrcs_noisy,opts),'images',dim_type='BHWC',text=r'$%s$'%paramstr)
        logger.addImage(visouts,npu.labels(mtrcs_pred,mtrcs_noisy,opts),'images_inset',dim_type='BHWC',text=r'$%s$'%paramstr,addinset=True)
        # logger.addMetrics(losses,'train')
    # logger.addMetrics(loss_dict,mode)
    logger.addMetrics(lossdict,mode)
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