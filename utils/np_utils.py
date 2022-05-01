from skimage.measure import compare_ssim
import numpy as np
import imageio

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

    out = [inpt['denoise'],inpt['ambient'],inpt['noisy'],inpt['flash']]
    if('fft' in opts.model):
        out += [inpt['fft_res'],
            np.abs(inpt['gx'])*1000,np.abs(inpt['dx'])*1000,
            np.abs(inpt['gy'])*1000,np.abs(inpt['dy'])*1000]
    if(opts.model == 'deepfnf+fft_grad_image'):
        out += [np.abs(inpt['g'])*1000]
    if('deepfnf+fft_helmholz' == opts.model):
        out += [np.abs(inpt['phix'])*1000, np.abs(inpt['phiy'])*1000, np.abs(inpt['ax'])*1000, np.abs(inpt['ay'])*1000]
    return out
    
def labels(mtrcs_pred,mtrcs_inpt,opts):
    out = [r'$Prediction~PSNR~%.3f$'%mtrcs_pred['psnr'],
           r'$Ground Truth$',r'$I_{noisy}~PSNR:%.3f$'%mtrcs_inpt['psnr'],r'$I_{flash}$']
    if('fft' in opts.model):
        out += [r'$I$',r'$(|g^x|)~\times~1000.$',r'$|I_{x}|~\times~1000$',r'$(|g^y|)~\times~1000.$',r'$|I_{y}|~\times~1000$']
    if(opts.model == 'deepfnf+fft_grad_image'):
        out += [r'$|g| \times 1000$']
    if('deepfnf+fft_helmholz' == opts.model):
        out += [r'$|\nabla_x \phi| \times 1000$',r'$|\nabla_y \phi| \times 1000$',r'$|\nabla_x a| \times 1000$',r'$|\nabla_y a| \times 1000$']
    return out