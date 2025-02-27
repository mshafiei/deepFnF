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
# from bilateral import bilateralFilter, bilateralSolve
# from BilateralParallel import bilateral_rgb
import cvgutils.Linalg as Linalg
import cvgutils.Viz as viz
import cvgutils.Utils as utils
from timeit import default_timer as timer
from gllf.gllf_utils import prepare_input_edict
from easydict import EasyDict as edict

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

def visualize(data,k,c, logger, errval, metrics, metrics_list, errors_dict,errors,denoised_deepfnf, datapath, running_time, model):
    levelKey = 'Level %d' % (6 - k)
    ambient = data['ambient']
    alpha = tf.squeeze(data['alpha']).astype(np.float32)
    noisy_ambient = data['noisy_ambient']
    noisy_flash = data['noisy_flash']
    deepfnf_scaled = tfu.camera_to_rgb(
            denoised_deepfnf / tf.squeeze(alpha), data['color_matrix'], data['adapt_matrix'])
    denoise_original = None
    laplacian_pyramid = None
    denoise_original_deepfnf = None
    alpha_map = None
    gllf_guide = None
    # denoise = noisy_flash
    ambient = tfu.camera_to_rgb(
        ambient, data['color_matrix'], data['adapt_matrix'])
    if(logger.opts.model != 'net_llf_tf2_tf_Deepfnf_Deepfnf_flash' and logger.opts.model != 'deepfnf_llf_alpha_map_unet' and logger.opts.model != 'deepfnf_llf_alpha_map_unet_v2' and logger.opts.model != 'net_llf_tf2_tf_local_alpha_Deepfnf_alpha'):
        denoise = denoise / alpha
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
        # denoise = bilateralFilter(denoise,flash_wb,params)
    
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
        if(deepfnf_scaled is not None):
            original_metrics = {}
            original_metrics_pred = errval.eval(ambient[None,...],deepfnf_scaled.numpy())
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
    # for key, v in metrics_list.items():
    #     if('psnr' in key.lower() or 'wlpips' == key.lower() or 'lpips' == key.lower()):
    #         print(key,':',v)

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
        im = {'flash':flash_wb, 'noisy':noisy_wb, 'ambient':ambient}
        lbl = {'flash':r'$I_{flash}$', 'noisy':r'$I_{noisy}$', 'ambient':r'$I_{ambient}$'}
        if(not(denoise_original is None)):
            cols+=1
            im.update({'denoise_original': denoise_original})
            lbl.update({'denoise_original': "deepfnf"})
        if(deepfnf_scaled is not None):
            im.update({'deepfnf_scaled':deepfnf_scaled})
            lbl.update({'deepfnf_scaled':r'$DeepFnF$'})
        im.update({'denoise':denoise})
        lbl.update({'denoise':r'$Deepfnf+GLLF$'})
        if(gllf_guide is not None):
            im.update({'gllf_guide':gllf_guide})
            lbl.update({'gllf_guide':r'$I_{H}$'})
        if(alpha_map is not None):
            import cv2
            alpha_map = cv2.resize(alpha_map.numpy(), (448,448))
            alpha_min, alpha_max = alpha_map.min(), alpha_map.max()
            alpha_map = (alpha_map - alpha_min) / (alpha_max - alpha_min)
            im.update({'alpha_map':alpha_map})
            lbl.update({'alpha_map':'$\\alpha \\in [%.02f, %.02f]$' %(alpha_min, alpha_max)})

        if(logger.opts.visualize_metrics):
            im.update({'blank':blank})
            lbl.update({'blank':r'$Measurements$'})
        annotation = {'noisy':'<br>Darkened:x%i'%(int(np.round(1/np.mean(alpha)))),'denoise':'<br>PSNR:%.3f<br>LPIPS:%.3f<br>WLPIPS:%.3f'%(metrics['psnr'],metrics['lpips'],metrics['wlpips'])}
        if(original_metrics is not None):
            annotation.update({'deepfnf_scaled':'<br>PSNR:%.3f<br>LPIPS:%.3f<br>WLPIPS:%.3f'%(original_metrics['psnr'],original_metrics['lpips'],original_metrics['wlpips'])})
        # annotation = {'blank':'<br>alpha:%.3f<br>PSNR:%.3f<br>LPIPS:%.3f<br>SSIM:%.3f<br>MSSSIM:%.3f<br>WLPIPS:%.3f'%(np.mean(alpha),metrics['psnr'],metrics['lpips'],metrics['ssim'],metrics['msssim'],metrics['wlpips'])}
        
        if(laplacian_pyramid is not None):
            im.update({'laplacian_interpolation_plot':laplacian_interpolation_plot})
            lbl.update({'laplacian_interpolation_plot':'L interpolation'})
            for i, l in enumerate(laplacian_pyramid):
                im.update({'laplacian_%i'%i: np.array(np.squeeze(l) / np.squeeze(alpha) * 10)})
                lbl.update({'laplacian_%i'%i: "l_%i~coeff:%f"%(i,np.array(source_laplacian_weight)[i])})
        if('spatial_lpips' in metrics.keys() and logger.opts.visualize_metrics):
            im.update({'spatial_lpips':metrics['spatial_lpips']})
            lbl.update({'spatial_lpips':r'$LPIPS$'})
            if(original_metrics is not None):
                im.update({'spatial_lpips_original':original_metrics['spatial_lpips']})
                lbl.update({'spatial_lpips_original':r'$LPIPS-deepfnf$'})
        if('spatial_euclidean_distance' in metrics.keys() and logger.opts.visualize_metrics):
            im.update({'spatial_euclidean_distance':metrics['spatial_euclidean_distance']})
            lbl.update({'spatial_euclidean_distance':r'$|I_{ours} - I_{ambient}|$'})
            if(original_metrics is not None):
                im.update({'spatial_euclidean_distance_original':original_metrics['spatial_euclidean_distance']})
                lbl.update({'spatial_euclidean_distance_original':r'$I_{deepfnf} - I_{ambient}$'})
        if('spatial_wlpips' in metrics.keys() and logger.opts.visualize_metrics):
            im.update({'spatial_wlpips':metrics['spatial_wlpips']})
            lbl.update({'spatial_wlpips':r'$WLPIPS$'})
            if(original_metrics is not None):
                im.update({'spatial_wlpips_original':original_metrics['spatial_wlpips']})
                lbl.update({'spatial_wlpips_original':r'$WLPIPS-deepfnf$'})

        filename = '%s/%i/%i.npz' % (datapath, k, c)
        # logger.addImage(im,lbl,'deepfnf',comp_lbls=['denoise','ambient'],dim_type='HWC',addinset=False,annotation=annotation,ltype='Jupyter',mode='test')
        if(logger.opts.separate_images):
            logger.addIndividualImages(im,lbl,'deepfnf',mode='test',annotation=annotation, idx='%03i_%03i'%(k,c))
            # logger.addImage(im,lbl,'deepfnf',dim_type='HWC',addinset=False,annotation=annotation,ltype='Jupyter',cols=cols,mode='test')
        else:
            logger.addImage(im,lbl,'deepfnf',dim_type='HWC',annotation=annotation,ltype='Jupyter',cols=cols,mode='test',idx='%03i_%03i'%(k,c), image_filename=filename, vertical_spacing_scale=3)
    logger.takeStep()

    update_reduced_errors_from_sampls(metrics_list, errors_dict, errors, levelKey)
    

def test_single_image(data, logger, model, errval, metrics, metrics_list, errors_dict, levelKey, iteration_count, filename):
    """Process a single test image through the model and compute metrics.

    Args:
        npz_data: Dictionary containing image data loaded from npz file
        logger: Logger object for saving results and visualizations
        model: Neural network model to evaluate
        errval: Error evaluation object for computing metrics
        metrics: Dictionary to store computed metrics
        metrics_list: Dictionary storing metrics for all processed images
        errors_dict: Dictionary storing aggregated error metrics
        levelKey: String key indicating the current image resolution level
        npz_fn: Filename of the npz data being processed
        iteration_count: Current iteration number
    """
    # Convert npz data to tensors
    
    
    # Prepare inputs for the model, applying optional clamping and standardization
    model_inputs = prepare_input_edict(data,clamp=logger.opts.clamp_dataset, std_input=logger.opts.std_input)
    
    # Run inference through the model
    model_output = model.forward(model_inputs)
    
    ambient = tfu.camera_to_rgb(
            data['ambient'], data['color_matrix'], data['adapt_matrix'])
    
    if(logger.opts.normalize_before_gllf):
        output = tfu.camera_to_rgb(
            model_output.output / data['alpha'], data['color_matrix'], data['adapt_matrix'])
    else:
        output = model_output.output
    # Compute quality metrics between model output and ground truth ambient image
    metrics_pred = errval.eval(tf.convert_to_tensor(ambient), output, output_numpy = True)
    
    # Update metrics dictionary with computed values
    metrics.update({'psnr':metrics_pred['psnr'], 'lpips':metrics_pred['lpips'],'wlpips':metrics_pred['wlpips']})
    
    # Periodically save visualizations based on visualize_freq
    if(iteration_count % logger.opts.visualize_freq == 0):
        # Toggle eager execution for visualization
        eagerly_state = tf.config.functions_run_eagerly()
        if(not eagerly_state):
            tf.config.run_functions_eagerly(True)
            
        # Create visualization with metrics annotations
        subtext = 'image_filename: ' + filename
        annotation = errval.create_annotation(metrics_pred, {'psnr':'PSNR', 'lpips':'LPIPS', 'wlpips':'WLPIPS'},['psnr', 'lpips', 'wlpips'])
        images, lbls = model.visualize(model_inputs)
        
        # Save visualizations through logger
        logger.addImage(images, lbls,'test',cols=6, annotation={'output':annotation}, image_filename=filename, font_size_scale=2,vertical_spacing_scale=2, subtext=subtext)
        logger.addIndividualImages({'output':images['output']}, {'output':lbls['output']})
        
        # Restore previous eager execution state
        if(not eagerly_state):
            tf.config.run_functions_eagerly(False)
    
    # Initialize metrics list for new level if needed
    if(levelKey not in metrics_list.keys()):
        metrics_list[levelKey] = {}
        
    # Update metrics lists with results from this image
    metrics_list[levelKey].update({os.path.basename(filename):metrics_pred})
    
    # Compute average metrics across all processed images at this level
    errors_dict[levelKey] = utils.aggregate_list_of_dicts(metrics_list[levelKey])
    errors_dict[levelKey] = utils.divide_dicts(errors_dict[levelKey],len(metrics_list[levelKey]))
    
    # Save updated metrics to disk
    logger.dumpDictJson(metrics_list,'test_errors_samples','test')
    logger.dumpDictJson(errors_dict,'test_errors','test')
    
def test(model, model_path, datapath,logger):
    errval = Linalg.ErrEvalTF2('psnr,lpips, wlpips',image_size=448)
    tf.config.run_functions_eagerly(True)
    k_val = None
    i_val = None
    if(logger.opts.test_idx != -1):
        idx = int(logger.opts.test_idx)
        k_val = idx // 128
        i_val = idx % 128
    errors = {}
    errors_dict = {}
    # logger.dumpDictJson(stats,'model_stats','train')
    # errval = Linalg.ErrEvalTF2('ssim,msssim,mse,psnr,lpips, wlpips, wlpips_abs, spatial_euclidean_distance, spatial_lpips, spatial_wlpips',image_size=448)
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
            levelKey = 'Level %d' % (4 - k)
            if(levelKey not in metrics_list.keys()):
                metrics_list[levelKey] = {}
            startc = 0
            if(len(metrics_list[levelKey])):
                startc = len(metrics_list[levelKey]) - 1 if len(metrics_list[levelKey]) > 0 else 0
                for i in range(startc):
                    logger.takeStep()
                    continue
            if(startc >= logger.opts.test_set_count):
                update_reduced_errors_from_sampls(metrics_list, errors_dict, errors, levelKey)
                logger.dumpDictJson(errors_dict,'test_errors','test')
            for c in tqdm.trange(startc,logger.opts.test_set_count,1):
                #if large image, loop over 448x448 patches
                levelKey = 'Level %d' % (4 - k)
                npz_fn = '%s/%d/%d.npz' % (datapath, k, c)
                inset_fn = '%03d_%03d.npz' % (k, c)
                npz_data = np.load(npz_fn,allow_pickle=True)
                if(logger.opts.large_images):
                    if(logger.insets_json is not None):
                        fn_exists = False
                        for key in logger.insets_json['insets'].keys():
                            fn_exists = fn_exists or key in inset_fn
                    if(not fn_exists):
                        continue
                        
                    h, w = npz_data['ambient'][0,:,:,0].shape
                    data_cropped = {}
                    results_cropped = {}
                    input_keys = list(data.files)
                    w_iter_ct = w//448+1
                    h_iter_ct = h//448+1
                    pad_w = 448 - (w % 448)
                    pad_h = 448 - (h % 448)

                    for j in range(w_iter_ct):
                        results_cropped_h = {}
                        for i in range(h_iter_ct):
                            for key in input_keys:
                                if(len(npz_data[key].shape) == 4):
                                    data_cropped[key] = np.array(npz_data[key][:,i*448:(i+1)*448,j*448:(j+1)*448,:])
                                    res_w, res_h = 0, 0
                                    if(j == w_iter_ct-1):
                                        res_w = pad_w
                                    if(i == h_iter_ct-1):
                                        res_h = pad_h
                                    data_cropped[key] = tf.pad(data_cropped[key], [[0,0],[0,res_h],[0,res_w],[0,0]])
                                else:
                                    data_cropped[key] = npz_data[key]
                            #Process images and get model results
                            k,c, logger, results_cropped_ij, datapath, running_time, model = test_idx(datapath,data_cropped,k,c,logger,model)
                            results_cropped_ij.update(data_cropped)
                            #concatenate results
                            for key in results_cropped_ij.keys():
                                if(len(results_cropped_ij[key].shape) != 4):
                                    continue
                                if(key not in results_cropped_h.keys()):
                                    results_cropped_h[key] = [results_cropped_ij[key]]
                                else:
                                    results_cropped_h[key].append(results_cropped_ij[key])
                        for key in results_cropped_h.keys():
                            if(len(results_cropped_h[key][0].shape) != 4):
                                continue
                            concat_image = tf.concat(results_cropped_h[key],axis=1)
                            if(key not in results_cropped.keys()):
                                results_cropped[key] = [concat_image]
                            else:
                                results_cropped[key].append(concat_image)
                    for key in results_cropped_h.keys():
                        if(len(results_cropped_h[key][0].shape) != 4):
                            continue
                        results_cropped[key] = tf.concat(results_cropped[key],axis=2)[:,:h,:w,:]
                    large_images = {}
                    lbl = {}
                    for key in results_cropped.keys():
                        if(len(results_cropped[key].shape) == 4):
                            large_images[key] = results_cropped[key]
                            lbl[key] = key
                    #Apply scale to all images
                    #return all images from process function
                    # model_output.noisy_flash_scaled = tfu.camera_to_rgb(
                    #         noisy_flash, data['color_matrix'], data['adapt_matrix'])
                    # model_output.noisy_ambient_scaled = tfu.camera_to_rgb(
                    #         noisy_ambient / alpha, data['color_matrix'], data['adapt_matrix'])
                    # model_output.ambient_scaled = tfu.camera_to_rgb(
                    #         data['ambient'], data['color_matrix'], data['adapt_matrix'])
                    order_keys = ['noisy_flash_scaled','noisy_ambient_scaled','ambient_scaled','deepfnf_scaled','output','llf_input','llf_guide','llf_alpha_h','llf_alpha_i']
                    ordered_dict = {}
                    ordered_labels = {}
                    for key in order_keys:
                        if(key in large_images.keys()):
                            ordered_dict[key] = large_images[key]
                            ordered_labels[key] = key
                    logger.addImage(ordered_dict,ordered_labels,'deepfnf',dim_type='HWC',cols=6,mode='test',idx='%03i_%03i'%(k,c), image_filename=str(inset_fn), vertical_spacing_scale=3)
                    #Visualize
                    # visualize(results_cropped,k,c, logger, errval, metrics, metrics_list, errors_dict,errors,denoised_deepfnf, datapath, running_time, model)
                    print('hi')

                    #pad flash, ambient, noisy, std
                    #process w/ the model
                    #concatenate
                else:
                    # Original code (backup)
                    # data = {key: tf.convert_to_tensor(npz_data[key],dtype=tf.float32) for key in npz_data.files}
                    # model_inputs = prepare_input_edict(data,clamp=logger.opts.clamp_dataset, std_input=logger.opts.std_input)
                    # model_output = model.forward(model_inputs)
                    # metrics_pred = errval.eval(tf.convert_to_tensor(data['ambient']), model_output.output, output_numpy = True)
                    # metrics.update({'psnr':metrics_pred['psnr'], 'lpips':metrics_pred['lpips'],'wlpips':metrics_pred['wlpips']})
                    # if(c % logger.opts.visualize_freq == 0):
                    #     eagerly_state = tf.config.functions_run_eagerly()
                    #     if(not eagerly_state):
                    #         tf.config.run_functions_eagerly(True)
                    #     subtext = 'image_filename: ' + npz_fn
                    #     annotation = errval.create_annotation(metrics_pred, {'psnr':'PSNR', 'lpips':'LPIPS', 'wlpips':'WLPIPS'},['psnr', 'lpips', 'wlpips'])
                    #     images, lbls = model.visualize(model_inputs)
                    #     logger.addImage(images, lbls,'test',cols=6, annotation={'output':annotation}, image_filename=npz_fn, font_size_scale=2,vertical_spacing_scale=2, subtext=subtext)
                    #     logger.addIndividualImages({'output':images['output']}, {'output':lbls['output']})
                    #     if(not eagerly_state):
                    #         tf.config.run_functions_eagerly(False)
                    # if(levelKey not in metrics_list.keys()):
                    #     metrics_list[levelKey] = {}
                    # metrics_list[levelKey].update({os.path.basename(npz_fn):metrics_pred})
                    # errors_dict[levelKey] = utils.aggregate_list_of_dicts(metrics_list[levelKey])
                    # errors_dict[levelKey] = utils.divide_dicts(errors_dict[levelKey],len(metrics_list[levelKey]))
                    # logger.dumpDictJson(metrics_list,'test_errors_samples','test')
                    # logger.dumpDictJson(errors_dict,'test_errors','test')
                    data = {key: tf.convert_to_tensor(npz_data[key],dtype=tf.float32) for key in npz_data.files}
                    test_single_image(data, logger, model, errval, metrics, metrics_list, errors_dict, levelKey, c, npz_fn)
                logger.takeStep()
                    
    else:
        metrics = {}
        metrics_list = {}
        test_idx(datapath,k_val,i_val,metrics,metrics_list,logger,model,errors_dict,errors, errval)
    logger.dumpDictJson(errors_dict,'test_errors','test')
