#!/usr/bin/env python3

import os
import argparse

import utils.np_utils as npu
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.eager as tfe

import utils.utils as ut
import utils.tf_utils as tfu
import tqdm
# tf.enable_eager_execution()

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--wts', default='wts/model.npz', help='path to trianed model')
# parser = Viz.logger.parse_arguments(parser)
# opts = parser.parse_args()

# logger = Viz.logger(opts,opts.__dict__)
# opts = logger.opts




# datapath = '/home/mohammad/Projects/optimizer/DifferentiableSolver/data/testset_nojitter'
# model_path = opts.wts

# model = net.Net(ksz=15, num_basis=90, burst_length=2)

def test_idx(datapath,k,c,metrics,metrics_list,logger,model,errors_dict,errors):
    data = np.load('%s/%d/%d.npz' % (datapath, k, c))
    alpha = data['alpha'][None, None, None, None]
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

    denoise = model.forward(net_input)
    denoise = denoise / alpha

    ambient = tfu.camera_to_rgb(
        ambient, data['color_matrix'], data['adapt_matrix'])
    denoise = tfu.camera_to_rgb(
        denoise, data['color_matrix'], data['adapt_matrix'])
    noisy_wb = tfu.camera_to_rgb(
        noisy_ambient/alpha, data['color_matrix'], data['adapt_matrix'])
    flash_wb = tfu.camera_to_rgb(
        noisy_flash/alpha, data['color_matrix'], data['adapt_matrix'])
    flash_wbx1 = tfu.camera_to_rgb(
        noisy_flash/alpha * 0.1, data['color_matrix'], data['adapt_matrix'])
    flash_wbx01 = tfu.camera_to_rgb(
        noisy_flash/alpha * 0.01, data['color_matrix'], data['adapt_matrix'])
    ambient = np.clip(ambient, 0., 1.).squeeze()
    denoise = np.clip(denoise, 0., 1.).squeeze()
    noisy_wb = np.clip(noisy_wb, 0., 1.).squeeze()
    flash_wb = np.clip(flash_wb, 0., 1.).squeeze()
    flash_wbx1 = np.clip(flash_wbx1, 0., 1.).squeeze()
    flash_wbx01 = np.clip(flash_wbx01, 0., 1.).squeeze()
    # if(erreval != None):
    #     piq_metrics_pred = erreval.eval(ambient,denoise,dtype='np',imformat='HWC')
    #     metrics.update({'msssim':piq_metrics_pred['msssim'],'lpipsVGG':piq_metrics_pred['lpipsVGG'],'lpipsAlex':piq_metrics_pred['lpipsAlex']})
    metrics.update({'mse':npu.get_mse(denoise, ambient),'psnr':npu.get_psnr(denoise, ambient),'ssim':npu.get_ssim(denoise, ambient)})
    for key,v in metrics.items():
        if(not(key in metrics_list.keys())):
            metrics_list[key] = []
    [metrics_list[key].append(v) for key,v in metrics.items()]
    if(c % 1 == 0):
        im = {'denoise':denoise, 'ambient':ambient, 'noisy':noisy_wb,'flash':flash_wb,'flashx1':flash_wbx1,'flashx01':flash_wbx01}
        lbl = {'denoise':r'$I$','ambient':r'$I_{ambient}$','noisy':r'$I_{noisy}$','flash':r'$I_{flash}$','flashx1':r'$I_{flash} \times 0.1$','flashx01':r'$I_{flash} \times 0.01$'}
        annotation = {'denoise':'%s<br>PSNR:%.3f<br>SSIM:%.3f'%('DeepFnF',metrics['psnr'],metrics['ssim'])}
        # logger.addIndividualImages(im,lbl,'deepfnf',dim_type='HWC',addinset=False,annotation=annotation,ltype='filesystem')
        logger.addImage(im,lbl,'deepfnf',comp_lbls=['denoise','ambient'],dim_type='HWC',addinset=False,annotation=annotation,ltype='Jupyter')
    logger.takeStep()

    mean_mtrcs = {key:'%.4f'%np.mean(np.array(v)) for key,v in metrics_list.items()}
    errstr = ['%s: %s' %(key,v) for key,v in mean_mtrcs.items()]
    errors_dict['Level %d' % (4 - k)] = mean_mtrcs
    errors['Level %d' % (4 - k)] = ', '.join(errstr)
    print(errors['Level %d' % (4 - k)])

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
    if(k_val == None or i_val == None):
        for k in range(4):
            metrics = {}
            metrics_list = {}
            for c in tqdm.trange(0,128,10):
                test_idx(datapath,k,c,metrics,metrics_list,logger,model,errors_dict,errors)
    else:
        metrics = {}
        metrics_list = {}
        test_idx(datapath,k_val,i_val,metrics,metrics_list,logger,model,errors_dict,errors)
    logger.dumpDictJson(errors_dict,'test_errors','test')
