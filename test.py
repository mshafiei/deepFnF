#!/usr/bin/env python3
import tensorflow as tf
import os
import argparse

import utils.np_utils as npu
import numpy as np

import utils.utils as ut
import utils.tf_utils as tfu
import tqdm
import time
from bilateral import bilateralFilter, bilateralSolve
from BilateralParallel import bilateral_rgb
import cvgutils.Linalg as Linalg
import cvgutils.Viz as viz

@tf.function
def eval_model_w_alpha(model, netinput, alpha):
    return model.forward(netinput, alpha)

@tf.function
def eval_original_Deepfnf(model, netinput, alpha):
    return model.deepfnfDenoising(netinput, alpha)

@tf.function
def eval_laplacian_interpolation(model, netinput, alpha):
    return model.pyramid(netinput, alpha)

@tf.function
def eval_model(model, netinput):
    return model.forward(netinput)
def test_idx(datapath,k,c,metrics,metrics_list,logger,model,errors_dict,errors, errval):
    levelKey = 'Level %d' % (4 - k)
    npz_fn = '%s/%d/%d.npz' % (datapath, k, c)
    data = np.load(npz_fn)
    alpha = data['alpha'][None, None, None, None]
    print('alpha ', alpha, ' npz ', npz_fn)
    ambient = data['ambient']
    dimmed_ambient, _ = tfu.dim_image(data['ambient'], alpha=alpha)
    dimmed_warped_ambient, _ = tfu.dim_image(
        data['warped_ambient'], alpha=alpha)

    # Make the flash brighter by increasing the brightness of the
    # flash-only image.
    flash = data['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
    warped_flash = data['warped_flash_only'] * \
        ut.FLASH_STRENGTH + dimmed_warped_ambient

    noisy_ambient = data['noisy_ambient']
    noisy_flash = data['noisy_warped_flash']
    # noisy_flash = warped_flash[0,0,0,0,:,:,:,:]
    noisy = tf.concat([noisy_ambient, noisy_flash], axis=-1)
    noise_std = tfu.estimate_std(
        noisy, data['sig_read'], data['sig_shot'])
    net_input = tf.concat([noisy, noise_std], axis=-1)
    
    start = time.time_ns()
    denoise_original = None
    laplacian_pyramid = None
    if(logger.opts.model == 'deepfnf_combine_fft' or logger.opts.model == 'deepfnf_combine_laplacian' or logger.opts.model == 'net_flash_image' or logger.opts.model == 'deepfnf_llf'):
        denoise = eval_model_w_alpha(model, net_input, data['alpha'])
        denoise_original = eval_original_Deepfnf(model, net_input, data['alpha'])[0]/alpha
        laplacian_pyramid = eval_laplacian_interpolation(model, net_input,data['alpha'])
    elif(logger.opts.model == 'deepfnf_combine_laplacian_pixelwise'):
        denoise = eval_model_w_alpha(model, net_input, data['alpha'])
        laplacianWeights = model.getLaplacianWeights()
    else:
        denoise = eval_model(model, net_input)

    end = time.time_ns()
    print('forward pass takes ', (end - start)/1000000, 'ms')

    denoise = denoise / alpha
    denoise = noisy_flash
    ambient = tfu.camera_to_rgb(
        ambient, data['color_matrix'], data['adapt_matrix'])
    denoise = tfu.camera_to_rgb(
        denoise, data['color_matrix'], data['adapt_matrix'])
    noisy_wb = tfu.camera_to_rgb(
        noisy_ambient/alpha, data['color_matrix'], data['adapt_matrix'])
    flash_wb = tfu.camera_to_rgb(
        noisy_flash, data['color_matrix'], data['adapt_matrix'])

    if(not(denoise_original is None)):
        denoise_original = tfu.camera_to_rgb(
            denoise_original, data['color_matrix'], data['adapt_matrix'])
        
    
    # ambient = np.array(tf.squeeze(tf.clip_by_value(ambient, 0., 1.)))
    denoise = tf.squeeze(denoise)
    if(logger.opts.fft_lmbda_pp):
        lmbda = logger.opts.fft_lmbda_pp
        denoise1 = tfu.screen_poisson(lmbda, denoise[:,:,0],denoise[:,:,0]*0,denoise[:,:,0]*0, 448)
        denoise2 = tfu.screen_poisson(lmbda, denoise[:,:,1],denoise[:,:,1]*0,denoise[:,:,1]*0, 448)
        denoise3 = tfu.screen_poisson(lmbda, denoise[:,:,2],denoise[:,:,2]*0,denoise[:,:,2]*0, 448)
        denoise = tf.squeeze(tf.stack([denoise1, denoise2, denoise3],axis=-1))
    
    ambient = np.array(ambient).squeeze()
    denoise = np.array(denoise).squeeze()
    flash_wb = np.array(flash_wb).squeeze()
    if(not(denoise_original is None)):
        denoise_original = np.array(denoise_original).squeeze()

    if(logger.opts.bilateral_pp):
        params = {}
        params['BILATERAL_SIGMA_SPATIAL'] = logger.opts.bilateral_spatial
        params['BILATERAL_SIGMA_LUMA'] = logger.opts.bilateral_luma
        params['LOSS_SMOOTH_MULT'] = logger.opts.bilateral_smooth
        params['A_VERT_DIAG_MIN'] = 1e-3
        params['NUM_PCG_ITERS'] = 30
        params['NUM_NEIGHBORS'] = logger.opts.bilateral_neighbors
        params['bs_lam'] = logger.opts.bs_lam
        # denoise = bilateral_rgb(flash_wb, denoise,flash_wb*0 + 1, params)
        denoise = bilateralFilter(denoise,flash_wb,params)
    
    denoise = np.clip(denoise, 0., 1.).squeeze()
    ambient = np.clip(ambient, 0., 1.).squeeze()
    if(not(denoise_original is None)):
        denoise_original = np.clip(denoise_original, 0., 1.).squeeze()

    noisy_wb = np.clip(noisy_wb, 0., 1.).squeeze()
    flash_wb = np.clip(flash_wb, 0., 1.).squeeze()

    
    if(errval != None):
        metrics_pred = errval.eval(ambient[None,...],denoise[None,...])
        for x in metrics_pred.keys():
            metrics[x] = np.array(metrics_pred[x])[0]
        # metrics.update({'psnr':metrics_pred['psnr'], 'ssim':metrics_pred['ssim'],'msssim':metrics_pred['msssim'],'lpips':metrics_pred['lpips'],'wlpips':metrics_pred['wlpips']})

    # metrics.update({'mse':npu.get_mse(denoise, ambient),'psnr':npu.get_psnr(denoise, ambient)})
    # for key,v in metrics.items():
    #     if(not(key in metrics_list[levelKey].keys())):
    #         metrics_list[levelKey][key] = []
    # [metrics_list[levelKey][key].append(v) for key,v in metrics.items()]
            
    #draw laplacian interpolation function
    kernel, inv_kernel = tfu.sigmoid(logger.opts.sigmoid_offset,logger.opts.sigmoid_intensity,[448,448])
    kernel = np.repeat(np.array(tf.signal.fftshift(kernel))[:,:,None],3,axis=-1)
    inv_kernel = np.repeat(np.array(tf.signal.fftshift(inv_kernel))[:,:,None],3,axis=-1)

    if(logger.opts.model == 'deepfnf_combine_laplacian'):
        xs = np.arange(model.laplacian_levels)
        ys = tfu.laplacian_interpolation_function(np.arange(model.laplacian_levels), model.laplacian_levels, model.x0, model.k)
        laplacian_interpolation_plot = viz.plot(xs, ys) / 255

    blank = inv_kernel * 0 + 1
    if(c % 1 == 0):
        im = {'blank':blank,'denoise':denoise, 'ambient':ambient, 'noisy':noisy_wb,'flash':flash_wb,'kernel':kernel,'1-kernel':inv_kernel, 'laplacian_interpolation_plot':laplacian_interpolation_plot}
        lbl = {'blank':r'$Measurements$','denoise':r'$I$','ambient':r'$I_{ambient}$','noisy':r'$I_{noisy}$','flash':r'$I_{flash}$','kernel':r'$k$','1-kernel':r'$1-k$', 'laplacian_interpolation_plot':'L interpolation'}
        annotation = {'blank':'<br>alpha:%.3f<br>PSNR:%.3f<br>LPIPS:%.3f<br>SSIM:%.3f<br>MSSSIM:%.3f<br>WLPIPS:%.3f'%(np.mean(alpha),metrics['psnr'],metrics['lpips'],metrics['ssim'],metrics['msssim'],metrics['wlpips'])}
        if(not(denoise_original is None)):
            im.update({'denoise_original': denoise_original})
            lbl.update({'denoise_original': "deepfnf"})
        if(laplacian_pyramid is not None):
            for i, l in enumerate(laplacian_pyramid):
                im.update({'laplacian_%i'%i: np.array(np.squeeze(l) / np.squeeze(alpha) * 10)})
                lbl.update({'laplacian_%i'%i: "l_%i"%i})
        # logger.addImage(im,lbl,'deepfnf',comp_lbls=['denoise','ambient'],dim_type='HWC',addinset=False,annotation=annotation,ltype='Jupyter',mode='test')
        logger.addImage(im,lbl,'deepfnf',dim_type='HWC',addinset=False,annotation=annotation,ltype='Jupyter',mode='test')
    logger.takeStep()

    
    mean_mtrcs = {key:'%.4f'%np.mean(np.array(v)) for key,v in metrics_list[levelKey].items()}
    errstr = ['%s: %s' %(key,v) for key,v in mean_mtrcs.items()]
    errors_dict[levelKey] = mean_mtrcs
    errors[levelKey] = ', '.join(errstr)
    print(errors[levelKey])

def test(model, model_path, datapath,logger):
    print('Done\n')
    k_val = None
    i_val = None
    if(logger.opts.test_idx != -1):
        idx = int(logger.opts.test_idx)
        k_val = idx // 128
        i_val = idx % 128
    errors = {}
    errors_dict = {}
    # logger.dumpDictJson(stats,'model_stats','train')
    errval = Linalg.ErrEvalTF2('ssim,msssim,mse,psnr,lpips, wlpips',image_size=448)
    metrics_list = logger.loadDictJson('test_errors_samples','test')
    if(metrics_list is None):
        metrics_list = {}
    # startK = len(metrics_list) - 1 if len(metrics_list) > 0 else 0

    if(k_val == None or i_val == None):
        for k in range(0, 4):
            metrics = {}
            levelKey = 'Level %d' % (4 - k)
            if(levelKey not in metrics_list.keys()):
                metrics_list[levelKey] = {}
            startc = 0
            if(len(metrics_list[levelKey])):
                startc = len(metrics_list[levelKey]['psnr']) - 1 if len(metrics_list[levelKey]['psnr']) > 0 else 0
                for i in range(startc):
                    logger.takeStep()
                    continue
            if(startc >= 127):
                test_idx(datapath,k,0,metrics,metrics_list,logger,model,errors_dict,errors, errval)
                logger.dumpDictJson(errors_dict,'test_errors','test')
            for c in tqdm.trange(startc,logger.opts.test_set_count,1):
                with tf.device('/gpu:0'):
                    test_idx(datapath,k,c,metrics,metrics_list,logger,model,errors_dict,errors, errval)
                    logger.dumpDictJson(metrics_list,'test_errors_samples','test')
                    logger.dumpDictJson(errors_dict,'test_errors','test')
                    
    else:
        metrics = {}
        metrics_list = {}
        with tf.device('/gpu:0'):
            test_idx(datapath,k_val,i_val,metrics,metrics_list,logger,model,errors_dict,errors, errval)
    logger.dumpDictJson(errors_dict,'test_errors','test')
