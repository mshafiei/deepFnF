#!/usr/bin/env python3

import os
import argparse

import utils.np_utils as npu
import numpy as np
import tensorflow as tf

import net
import utils.utils as ut
import utils.tf_utils as tfu
from utils.dataset import Dataset
import cvgutils.Viz as Viz

parser = argparse.ArgumentParser()
parser.add_argument('--TLIST', type=str, default='data/train_1600.txt', help='Training dataset filename')
parser.add_argument('--VPATH', type=str, default='data/valset', help='Validation dataset')
parser.add_argument('--model', type=str, default='deepfnf+fft',choices=['deepfnf','deepfnf+fft'], help='Validation dataset')
parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
parser.add_argument('--weight_dir', type=str, default='wts', help='Weight dir')
parser.add_argument('--visualize_freq', type=int, default=5001, help='How many iterations before visualization')

parser = Viz.logger.parse_arguments(parser)
opts = parser.parse_args()
opts.expname = 'deepfnf-fft'
opts.logdir = 'deepfnf-fft'
logger = Viz.logger(opts,opts.__dict__)

TLIST = opts.TLIST
VPATH = opts.VPATH

BSZ = 1
IMSZ = 448
LR = 1e-4
DROP = (1.1e6, 1.25e6) # Learning rate drop
MAXITER = 1.5e6

VALFREQ = 4e1
SAVEFREQ = 5e4

wts = opts.weight_dir
if not os.path.exists(wts):
    os.makedirs(wts)
if(opts.model == 'deepfnf'):
    outchannels = 3
elif(opts.model == 'deepfnf+fft'):
    outchannels = 6

model = net.Net(opts.model,outchannels,ksz=15, num_basis=90, burst_length=2)


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
            pre_ambient = example['ambient']
            ambient = tfu.camera_to_rgb(
                example['ambient'],
                example['color_matrix'], example['adapt_matrix'])
            noisy_ambient_map = tfu.camera_to_rgb(
                noisy_ambient/example['alpha'],
                example['color_matrix'], example['adapt_matrix'])

            # Loss
            if('deepfnf+fft' == opts.model):
                lmbda = model.weights['lmbda']
                # screen_poisson = lambda x,y,z,w,u : tf.map_fn(tfu.screen_poisson,([x]*BSZ,y,z,w,u))
                reshape = lambda x :tf.transpose(x,[0,3,1,2])
                fft_res = tfu.screen_poisson(lmbda,reshape(noisy_ambient),reshape(model_outpt[...,:3]), reshape(model_outpt[...,3:]),IMSZ)
                fft_res = tf.transpose(fft_res,[0,2,3,1])
                denoise = tfu.camera_to_rgb(
                fft_res/alpha, example['color_matrix'], example['adapt_matrix'])
                sp_loss = tfu.l2_loss(denoise, ambient)
                psnr = tfu.get_psnr(denoise, ambient)
                

                lvals = [sp_loss, psnr]
                lnms = ['loss', 'psnr']
                loss = sp_loss
            elif('deepfnf' == opts.model):
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
def dx(im):
    im = np.pad(im,((0,0),(0,0),(1,0),(0,0)))
    return im[:,:,1:,:]-im[:,:,:-1,:]
def dy(im):
    im = np.pad(im,((0,0),(1,0),(0,0),(0,0)))
    return im[:,1:,:,:]-im[:,:-1,:,:]
def metrics(preds,gts):
    mse, psnr, ssim = [], [], []
    for pred,gt in zip(preds,gts):
        pred = np.clip(pred,0,1)
        gt = np.clip(gt,0,1)
        mse.append(npu.get_mse(pred,gt))
        psnr.append(npu.get_psnr(pred,gt))
        ssim.append(npu.get_ssim(pred,gt))
    return {'mse':mse,'psnr':psnr,'ssim':ssim}

def visualize(inpt):

    # g = self.quad_model(inpt['net_input'])
    g,fft_res = inpt['g'],inpt['fft_res']
    gx = g[...,:3]
    gy = g[...,3:]
    # predict = tfu.camera_to_rgb_batch(predict/inpt['alpha'],inpt)
    dx = np.roll(fft_res, 1, axis=[2]) - fft_res
    dy = np.roll(fft_res, 1, axis=[1]) - fft_res

    dxx = np.roll(dx, 1, axis=[2]) - dx
    dyy = np.roll(dy, 1, axis=[1]) - dy
    gxx = np.roll(gx, 1, axis=[2]) - gx
    gyy = np.roll(gy, 1, axis=[1]) - gy
    loss_data = ((dx - gx) ** 2 + (dy - gy) ** 2).mean()
    loss_smoothness = ((fft_res/inpt['alpha'] - inpt['preambient']) ** 2).mean()
    
    
    
    out = [inpt['predict'],inpt['ambient'],inpt['noisy'],fft_res/inpt['alpha'],
            np.abs(gxx)*1000,np.abs(dxx/inpt['alpha'])*100,
            np.abs(gyy)*1000,np.abs(dyy/inpt['alpha'])*100,
            np.abs(gx),np.abs(dx/inpt['alpha'])*100,
            np.abs(gy),np.abs(dy/inpt['alpha'])*100]
    return out,{'loss_data':loss_data,'loss_smoothness':loss_smoothness}
def labels(mtrcs_pred,mtrcs_inpt):
    out = [r'$Prediction~PSNR~%.3f$'%mtrcs_pred['psnr'][0],
           r'$Ground Truth$',r'$I_{noisy}~PSNR:%.3f$'%mtrcs_inpt['psnr'][0],
           r'$I$',r'$Unet~output~(|g^x_x|)~\times~1000$',r'$|I_{xx}|~\times~100$',r'$Unet~output~(|g^y_y|)~\times~1000$',r'$|I_{yy}|~\times~100$',
    r'$Unet~output~(|g^x|)~\times~1.$',r'$|I_{x}|~\times~100$',r'$Unet~output (|g^y|)~\times~1.$',r'$|I_{y}|~\times~100$']
    return out
    # predict,(gt,grad_x,dx,grad_y,dy) = self(inpt)
    # return [predict,gt,grad_x[None,...],dx,grad_y[None,...],dy]
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
        loss_dict = {i:j for i,j in zip(tnms,outs[0].tolist())}
        mode = 'val'
    else:
        outs = sess.run(
            [loss, psnr, tStep,model.weights['lmbda']],
            feed_dict={lr: get_lr(niter), global_step: niter}
        )
        loss_dict = {'loss.t':outs[0],'psnr':outs[1],'lmbda':outs[3]}
        ut.vprint(niter, ['loss.t','loss.psnr'], [outs[0],outs[1]])
        mode = 'train'

    if(niter % opts.visualize_freq == 0):
        visouts = sess.run([pre_ambient,denoise,model_outpt,fft_res,alpha,ambient,noisy_ambient_map])
        inpt = {'g':visouts[2],'alpha':visouts[4],'fft_res':visouts[3],'preambient':visouts[0],'ambient':visouts[5],'predict':visouts[1],'noisy':visouts[6]}
        visouts, losses = visualize(inpt)
        mtrcs_pred = metrics(inpt['predict'],inpt['ambient'])
        mtrcs_noisy = metrics(inpt['noisy'],inpt['ambient'])

        logger.addImage(visouts,labels(mtrcs_pred,mtrcs_noisy),'',dim_type='BHWC')
        logger.addMetrics(losses,'train')
    
    logger.addMetrics(loss_dict,mode)
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