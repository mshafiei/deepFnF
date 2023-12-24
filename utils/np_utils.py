from collections import OrderedDict
from skimage.metrics import structural_similarity
import numpy as np
import imageio
import utils.tf_utils as tfu
import math

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
    ssim = structural_similarity(
        pred,
        gt,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        K1=0.01,
        K2=0.03,
        sigma=1.5)
    return ssim

def screen_poisson(lambda_d, img,grad_x,grad_y,axes=(-2,-1)):
    
    img_freq = np.fft.fft2(img,axes=axes)
    grad_x_freq = np.fft.fft2(grad_x,axes=axes)
    grad_y_freq = np.fft.fft2(grad_y,axes=axes)
    
    sx = np.fft.fftfreq(img.shape[axes[-1]])
    sx = np.repeat(sx, img.shape[axes[-2]])
    sx = np.reshape(sx, [img.shape[axes[-1]], img.shape[axes[-2]]])
    sx = np.transpose(sx)
    sy = np.fft.fftfreq(img.shape[axes[-2]])
    sy = np.repeat(sy, img.shape[axes[-1]])
    sy = np.reshape(sy, img.shape)

    # Fourier transform of shift operators
    Dx_freq = 2 * math.pi * (np.exp(-1j * sx) - 1)
    Dy_freq = 2 * math.pi * (np.exp(-1j * sy) - 1)

    # my_grad_x_freq = Dx_freq * img_freqs)
    # my_grad_x_freq & my_grad_y_freq should be the same as grad_x_freq & grad_y_freq
    lambda_d = np.minimum(1,np.maximum(0,lambda_d))
    recon_freq = (lambda_d * img_freq + (1 - lambda_d) * np.conjugate(Dx_freq) * grad_x_freq + np.conjugate(Dy_freq) * grad_y_freq) / \
                (lambda_d + (1 - lambda_d) * (np.conjugate(Dx_freq) * Dx_freq + np.conjugate(Dy_freq) * Dy_freq))
    return np.real(np.fft.ifft2(recon_freq))

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