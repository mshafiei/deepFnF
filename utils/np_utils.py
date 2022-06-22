from typing import OrderedDict
from skimage.measure import compare_ssim
import numpy as np
import imageio
import utils.tf_utils as tfu

def imsave(nm, img):
    if len(img.shape) == 4:
        img = np.squeeze(img, 0)
    img = np.uint8(np.clip(img,0,1) * 255.)
    imageio.imsave(nm, img)


def get_mse(pred, gt):
    return np.mean(np.square(pred-gt))


def get_psnr(pred, gt):
    pred = pred.clip(0., 1.)
    gt = gt.clip(0., 1.)
    mse = np.mean((pred-gt)**2.0)
    psnr = -10. * np.log10(mse)
    return psnr
def metrics(preds,gts,ignorelist='ssim'):
    mtrs = {}
    for pred,gt in zip(preds,gts):
        pred = np.clip(pred,0,1)
        gt = np.clip(gt,0,1)
        mtrs.update({'mse':get_mse(pred,gt)})
        mtrs.update({'psnr':get_psnr(pred,gt)})
        if('ssim' not in ignorelist):
            mtrs.update({'ssim':get_ssim(pred,gt)})
    return mtrs

def get_ssim(pred, gt):
    ssim = compare_ssim(
        pred,
        gt,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        K1=0.01,
        K2=0.03,
        sigma=1.5)
    return ssim
def dx_np(x):
    #x is b,h,w,c
    return np.roll(x, 1, axis=[2]) - x
def dy_np(x):
    #x is b,h,w,c
    return np.roll(x, 1, axis=[1]) - x

def visualize(inpt,opts):
    wb = lambda x: tfu.camera_to_rgb_np(
                    x, inpt['color_matrix'], inpt['adapt_matrix'])
    out = OrderedDict()
    out['denoise'] = inpt['denoise']
    out['ambient'] = inpt['ambient']
    out['noisy'] = inpt['noisy']
    out['flash'] = wb(inpt['flash'])

    if('fft' in opts.model):
        out['fft_res'] = wb(np.abs(inpt['fft_res'])/inpt['alpha'])
        out['gx'] = wb(np.abs(inpt['gx'][...,:3])/inpt['alpha'])
        out['dx'] = wb(np.abs(inpt['dx'])/inpt['alpha'])
        out['gy'] = wb(np.abs(inpt['gy'][...,:3])/inpt['alpha'])
        out['dy'] = wb(np.abs(inpt['dy'])/inpt['alpha'])

    if(opts.model == 'deepfnf+fft_grad_image'):
        out['g'] = wb(np.abs(inpt['g'])/inpt['alpha'])
    if('deepfnf+fft_helmholz' == opts.model):
        out['phix'] = wb(np.abs(inpt['phix'])/inpt['alpha'])
        out['phiy'] = wb(np.abs(inpt['phiy'])/inpt['alpha'])
        out['ax'] = wb(np.abs(inpt['ax'])/inpt['alpha'])
        out['ay'] = wb(np.abs(inpt['ay'])/inpt['alpha'])
    return out
    
def labels(mtrcs_pred,mtrcs_inpt,opts):
    out = OrderedDict()
    out['denoise'] = r'$Prediction~PSNR~%.3f$'%mtrcs_pred['psnr']
    out['ambient'] = r'$Ground Truth$'
    out['noisy'] = r'$I_{noisy}~PSNR:%.3f$'%mtrcs_inpt['psnr']
    out['flash'] = r'$I_{flash}$'

    if('fft' in opts.model):
        out['fft_res'] = r'$I$'
        out['gx'] = r'$(|g^x|)$'
        out['dx'] = r'$|I_{x}|$'
        out['gy'] = r'$(|g^y|)$'
        out['dy'] = r'$|I_{y}|$'

    if(opts.model == 'deepfnf+fft_grad_image'):
        out['g'] = r'$|g|$'
    if('deepfnf+fft_helmholz' == opts.model):
        out['phix'] = r'$|\nabla_x \phi|$'
        out['phiy'] = r'$|\nabla_y \phi|$'
        out['ax'] = r'$|\nabla_x a|$'
        out['ay'] = r'$|\nabla_y a|$'
    return out