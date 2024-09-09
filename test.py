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
def eval_original_model(model, netinput, alpha):
    return model.forward(netinput, alpha)

@tf.function
def eval_laplacian_interpolation(model, netinput, alpha):
    return model.pyramid(netinput, alpha)

@tf.function
def eval_model(model, netinput):
    return model.forward(netinput)

def update_reduced_errors_from_sampls(metrics_list, errors_dict, errors, levelKey):
    mean_mtrcs = {}
    for key,v in metrics_list[levelKey].items():
        if(key == 'running_time'):
            print('running time vector ', v)
            mean_mtrcs[key] = '%.4f'%np.median(np.array(v))
        else:
            mean_mtrcs[key] = '%.4f'%np.mean(np.array(v))
    errstr = ['%s: %s' %(key,v) for key,v in mean_mtrcs.items()]
    errors_dict[levelKey] = mean_mtrcs
    errors[levelKey] = ', '.join(errstr)
    print('mean error: ', errors[levelKey])
    
def test_idx(datapath,k,c,metrics,metrics_list,logger,model,errors_dict,errors, errval):
    levelKey = 'Level %d' % (6 - k)
    npz_fn = '%s/%d/%d.npz' % (datapath, k, c)
    data = np.load(npz_fn)
    alpha = data['alpha'][None, None, None, None].astype(np.float32)
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
    denoise_original_deepfnf = None
    if(logger.opts.model == 'deepfnf_llf'):
        denoised, flash = eval_original_Deepfnf(model, net_input, alpha)
        denoise = model.llf(denoised, flash)
    elif(logger.opts.model == 'unet_llf'):
        denoised, flash = eval_original_model(model, net_input, alpha)
        denoise = model.llf(denoised, flash)
    elif(logger.opts.model == 'deepfnf_combine_fft' or logger.opts.model == 'deepfnf_combine_laplacian' or logger.opts.model == 'net_flash_image' or logger.opts.model == 'deepfnf_llf_diffable'or logger.opts.model == 'flash'or logger.opts.model == 'noisy'):
        denoise = eval_model_w_alpha(model, net_input, alpha)
        # denoise_original_deepfnf = eval_original_Deepfnf(model, net_input, alpha)[0]/alpha
        # laplacian_pyramid = eval_laplacian_interpolation(model, net_input,alpha)
    elif(logger.opts.model == 'deepfnf_combine_laplacian_pixelwise'):
        denoise = eval_model_w_alpha(model, net_input, data['alpha'])
        laplacianWeights = model.getLaplacianWeights()
    else:
        denoise = eval_model(model, net_input)

    end = time.time_ns()
    running_time = (end - start)/1000000
    print('forward pass takes ', running_time, 'ms')

    denoise = denoise / alpha
    # denoise = noisy_flash
    ambient = tfu.camera_to_rgb(
        ambient, data['color_matrix'], data['adapt_matrix'])
    denoise = tfu.camera_to_rgb(
        denoise, data['color_matrix'], data['adapt_matrix'])
    noisy_wb = tfu.camera_to_rgb(
        noisy_ambient/alpha, data['color_matrix'], data['adapt_matrix'])
    flash_wb = tfu.camera_to_rgb(
        noisy_flash, data['color_matrix'], data['adapt_matrix'])

    if(not(denoise_original_deepfnf is None)):
        denoise_original_deepfnf = tfu.camera_to_rgb(
            denoise_original_deepfnf, data['color_matrix'], data['adapt_matrix'])
        
    
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
    if(not(denoise_original_deepfnf is None)):
        denoise_original_deepfnf = np.array(denoise_original_deepfnf).squeeze()

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
    if(not(denoise_original_deepfnf is None)):
        denoise_original_deepfnf = np.clip(denoise_original_deepfnf, 0., 1.).squeeze()

    noisy_wb = np.clip(noisy_wb, 0., 1.).squeeze()
    flash_wb = np.clip(flash_wb, 0., 1.).squeeze()

    original_metrics = None
    if(errval != None):
        metrics_pred = errval.eval(ambient[None,...],denoise[None,...])
        for x in metrics_pred.keys():
            metrics[x] = np.array(metrics_pred[x])[0]
        if(denoise_original_deepfnf is not None):
            original_metrics = {}
            original_metrics_pred = errval.eval(ambient[None,...],denoise_original_deepfnf[None,...])
            for x in original_metrics_pred.keys():
                original_metrics[x] = np.array(original_metrics_pred[x])[0]

        metrics.update({'psnr':metrics_pred['psnr'], 'ssim':metrics_pred['ssim'],'msssim':metrics_pred['msssim'],'lpips':metrics_pred['lpips'],'wlpips':metrics_pred['wlpips']})
    print('running_time1: ', running_time)
    metrics.update({'mse':npu.get_mse(denoise, ambient),'psnr':npu.get_psnr(denoise, ambient),'running_time':running_time})
    for key,v in metrics.items():
        if(not(key in metrics_list[levelKey].keys()) and 'spatial' not in key):
            metrics_list[levelKey][key] = []
    for key,v in metrics.items():
        if('spatial' not in key):
            metrics_list[levelKey][key].append(np.array(v).item())
            
    #draw laplacian interpolation function
    kernel, inv_kernel = tfu.sigmoid(logger.opts.sigmoid_offset,logger.opts.sigmoid_intensity,[448,448])
    kernel = np.repeat(np.array(tf.signal.fftshift(kernel))[:,:,None],3,axis=-1)
    inv_kernel = np.repeat(np.array(tf.signal.fftshift(inv_kernel))[:,:,None],3,axis=-1)

    if(logger.opts.model == 'deepfnf_combine_laplacian'):
        xs = np.arange(model.laplacian_levels+1)
        source_laplacian_weight = list(tfu.laplacian_interpolation_factor(np.arange(model.laplacian_levels), model.laplacian_levels, model.x0, model.k))
        source_laplacian_weight.append(1)
        # source_laplacian_weight = [1,1,1,1,1,1]
        laplacian_interpolation_plot = viz.plot(xs, source_laplacian_weight) / 255

    blank = inv_kernel * 0 + 1
    cols = 5
    if(c % logger.opts.visualize_freq == 0 and logger.opts.no_visualize is False):
        im = {'flash':flash_wb, 'noisy':noisy_wb, 'ambient':ambient, 'denoise':denoise}
        lbl = {'flash':r'$I_{flash}$', 'noisy':r'$I_{noisy}$', 'ambient':r'$I_{ambient}$','denoise':r'$I_{ours}$'}
        if(not(denoise_original is None)):
            cols+=1
            im.update({'denoise_original': denoise_original})
            lbl.update({'denoise_original': "deepfnf"})

        im.update({'blank':blank})
        lbl.update({'blank':r'$Measurements$'})
        # annotation = {'blank':'<br>alpha:%.3f<br>PSNR:%.3f<br>LPIPS:%.3f<br>SSIM:%.3f<br>MSSSIM:%.3f<br>WLPIPS:%.3f'%(np.mean(alpha),metrics['psnr'],metrics['lpips'],metrics['ssim'],metrics['msssim'],metrics['wlpips'])}
        annotation = {'noisy':'<br>Darkened:x%i'%(int(np.round(1/np.mean(alpha)))),'denoise':'<br>PSNR:%.3f<br>LPIPS:%.3f<br>WLPIPS:%.3f'%(metrics['psnr'],metrics['lpips'],metrics['wlpips'])}
        if(laplacian_pyramid is not None):
            im.update({'laplacian_interpolation_plot':laplacian_interpolation_plot})
            lbl.update({'laplacian_interpolation_plot':'L interpolation'})
            for i, l in enumerate(laplacian_pyramid):
                im.update({'laplacian_%i'%i: np.array(np.squeeze(l) / np.squeeze(alpha) * 10)})
                lbl.update({'laplacian_%i'%i: "l_%i~coeff:%f"%(i,np.array(source_laplacian_weight)[i])})
        if('spatial_lpips' in metrics.keys()):
            im.update({'spatial_lpips':metrics['spatial_lpips']})
            lbl.update({'spatial_lpips':r'$LPIPS$'})
            if(original_metrics is not None):
                im.update({'spatial_lpips_original':original_metrics['spatial_lpips']})
                lbl.update({'spatial_lpips_original':r'$LPIPS-deepfnf$'})
        if('spatial_euclidean_distance' in metrics.keys()):
            im.update({'spatial_euclidean_distance':metrics['spatial_euclidean_distance']})
            lbl.update({'spatial_euclidean_distance':r'$|I_{ours} - I_{ambient}|$'})
            if(original_metrics is not None):
                im.update({'spatial_euclidean_distance_original':original_metrics['spatial_euclidean_distance']})
                lbl.update({'spatial_euclidean_distance_original':r'$I_{deepfnf} - I_{ambient}$'})
        if('spatial_wlpips' in metrics.keys()):
            im.update({'spatial_wlpips':metrics['spatial_wlpips']})
            lbl.update({'spatial_wlpips':r'$WLPIPS$'})
            if(original_metrics is not None):
                im.update({'spatial_wlpips_original':original_metrics['spatial_wlpips']})
                lbl.update({'spatial_wlpips_original':r'$WLPIPS-deepfnf$'})

        # logger.addImage(im,lbl,'deepfnf',comp_lbls=['denoise','ambient'],dim_type='HWC',addinset=False,annotation=annotation,ltype='Jupyter',mode='test')
        if(logger.opts.separate_images):
            logger.addIndividualImages(im,lbl,'deepfnf',mode='test',annotation=annotation, idx='%03i_%03i'%(k,c))
            # logger.addImage(im,lbl,'deepfnf',dim_type='HWC',addinset=False,annotation=annotation,ltype='Jupyter',cols=cols,mode='test')
        else:
            logger.addImage(im,lbl,'deepfnf',dim_type='HWC',addinset=False,annotation=annotation,ltype='Jupyter',cols=cols,mode='test',idx='%03i_%03i'%(k,c))
    logger.takeStep()

    update_reduced_errors_from_sampls(metrics_list, errors_dict, errors, levelKey)

def test(model, model_path, datapath,logger):
    k_val = None
    i_val = None
    if(logger.opts.test_idx != -1):
        idx = int(logger.opts.test_idx)
        k_val = idx // 128
        i_val = idx % 128
    errors = {}
    errors_dict = {}
    # logger.dumpDictJson(stats,'model_stats','train')
    errval = Linalg.ErrEvalTF2('ssim,msssim,mse,psnr,lpips, wlpips, wlpips_abs, spatial_euclidean_distance, spatial_lpips, spatial_wlpips',image_size=448)
    metrics_list = logger.loadDictJson('test_errors_samples','test')
    if(metrics_list is None):
        metrics_list = {}
    # startK = len(metrics_list) - 1 if len(metrics_list) > 0 else 0
    if(logger.opts.subset_idx != -1):
        subset_idx_start = logger.opts.subset_idx
        subset_idx_start_end = logger.opts.subset_idx+1
    else:
        subset_idx_start = 0
        subset_idx_start_end = len(os.listdir(datapath))

    if(k_val == None or i_val == None):
        for k in range(subset_idx_start, subset_idx_start_end):
            metrics = {}
            levelKey = 'Level %d' % (6 - k)
            if(levelKey not in metrics_list.keys()):
                metrics_list[levelKey] = {}
            startc = 0
            if(len(metrics_list[levelKey])):
                startc = len(metrics_list[levelKey]['psnr']) - 1 if len(metrics_list[levelKey]['psnr']) > 0 else 0
                for i in range(startc):
                    logger.takeStep()
                    continue
            if(startc >= logger.opts.test_set_count):
                update_reduced_errors_from_sampls(metrics_list, errors_dict, errors, levelKey)
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
