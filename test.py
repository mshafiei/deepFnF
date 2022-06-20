#!/usr/bin/env python3

import os
import argparse

import utils.np_utils as npu
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import net
import utils.utils as ut
import utils.tf_utils as tfu
import cvgutils.Viz as Viz
import cvgutils.Linalg as linalg
import tqdm
# tf.enable_eager_execution()

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--wts', default='wts/model.npz', help='path to trianed model')
# parser = Viz.logger.parse_arguments(parser)
# opts = parser.parse_args()

# logger = Viz.logger(opts,opts.__dict__)
# opts = logger.opts

def load_net(fn, model):
    wts = np.load(fn)
    for k, v in wts.items():
        model.weights[k] = tfe.Variable(v)


# datapath = '/home/mohammad/Projects/optimizer/DifferentiableSolver/data/testset_nojitter'
# model_path = opts.wts

# model = net.Net(ksz=15, num_basis=90, burst_length=2)

def test(model, model_path, datapath,logger):
    print("Restoring model from " + model_path)
    load_net(model_path, model)
    print('Done\n')
    erreval = linalg.ErrEval('cpu')
    errors = {}
    errors_dict = {}
    for k in range(4):
        metrics = {}
        metrics_list = {}
        for c in tqdm.trange(128):
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
            ambient = np.clip(ambient, 0., 1.).squeeze()
            denoise = np.clip(denoise, 0., 1.).squeeze()
            noisy_wb = np.clip(noisy_wb, 0., 1.).squeeze()
            noisy_flash = np.clip(noisy_flash, 0., 1.).squeeze()
            if(erreval != None):
                piq_metrics_pred = erreval.eval(ambient,denoise,dtype='np',imformat='HWC')
                metrics.update({'msssim':piq_metrics_pred['msssim'],'lpipsVGG':piq_metrics_pred['lpipsVGG'],'lpipsAlex':piq_metrics_pred['lpipsAlex']})
            metrics.update({'mse':npu.get_mse(denoise, ambient),'psnr':npu.get_psnr(denoise, ambient),'ssim':npu.get_ssim(denoise, ambient)})
            for key,v in metrics.items():
                if(not(key in metrics_list.keys())):
                    metrics_list[key] = []
            [metrics_list[key].append(v) for key,v in metrics.items()]
            if(c % 10 == 0):
                im = {'denoise':denoise, 'ambient':ambient, 'noisy':noisy_wb,'flash':noisy_flash}
                lbl = {'denoise':r'$I$','ambient':r'$I_{ambient}$','noisy':r'$I_{noisy}$','flash':r'$I_{flash}$'}
                annotation = {'denoise':'%s<br>PSNR:%.3f<br>SSIM:%.3f'%('DeepFnF',metrics['psnr'],metrics['ssim'])}
                logger.addIndividualImages(im,lbl,'deepfnf',dim_type='HWC',addinset=False,annotation=annotation,ltype='filesystem')
            logger.takeStep()

        mean_mtrcs = {key:'%.4f'%np.mean(np.array(v)) for key,v in metrics_list.items()}
        errstr = ['%s: %s' %(key,v) for key,v in mean_mtrcs.items()]
        errors_dict['Level %d' % (4 - k)] = mean_mtrcs
        errors['Level %d' % (4 - k)] = ', '.join(errstr)
        print(errors['Level %d' % (4 - k)])
    logger.dumpDictJson(errors_dict,'test_errors','test')
