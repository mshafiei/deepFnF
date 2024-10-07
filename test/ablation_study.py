from arguments_deepfnf import parse_arguments_deepfnf
parser = parse_arguments_deepfnf()
opts = parser.parse_args()
import cvgutils.Viz as Viz
import numpy as np
import utils.tf_utils as tfu
import os
from cvgutils.nn.lpips_tf2.models_tensorflow.lpips_tensorflow import load_perceptual_models, learned_perceptual_metric_model
from guided_local_laplacian_color_Mullapudi2016 import guided_local_laplacian_color_Mullapudi2016 as guided_local_laplacian_Mullapudi2016
from local_laplacian import local_laplacian

logger = Viz.logger(opts,opts.__dict__)

#load data
overfit_example_gt_data_fn = './overfit_example_data_gt.pkl'
overfit_example_noisy_data_fn = './overfit_example_data_noisy.pkl'
data_gt = logger.load_pickle(overfit_example_gt_data_fn)
data_noisy = logger.load_pickle(overfit_example_noisy_data_fn)
net_input = np.squeeze(data_noisy['net_input'])
ambient_gt = np.squeeze(data_gt['ambient'])
alpha = np.squeeze(data_noisy['alpha'])
noisy_flash = np.squeeze(data_noisy['noisy_flash'])
noisy_ambient = np.squeeze(data_noisy['noisy_ambient'])
niter = data_noisy['niter']
denoise = np.squeeze(data_noisy['denoise'])

#swizzle
denoise_hl = np.ascontiguousarray(denoise.transpose(2,0,1))
noisy_ambient_hl = np.ascontiguousarray(noisy_ambient.transpose(2,0,1))
noisy_flash_hl = np.ascontiguousarray(noisy_flash.transpose(2,0,1))

denoise = tfu.camera_to_rgb(
    denoise[None,...] / alpha,
    data_gt['color_matrix'], data_gt['adapt_matrix'])
ambient_gt = tfu.camera_to_rgb(
    ambient_gt[None,...],
    data_gt['color_matrix'], data_gt['adapt_matrix'])
noisy_ambient= tfu.camera_to_rgb(
    noisy_ambient[None,...] / alpha,
    data_gt['color_matrix'], data_gt['adapt_matrix'])
noisy_flash= tfu.camera_to_rgb(
    noisy_flash[None,...],
    data_gt['color_matrix'], data_gt['adapt_matrix'])

noisy_ambient = np.squeeze(noisy_ambient)
denoise = np.squeeze(denoise)
noisy_flash = np.squeeze(noisy_flash)

denoise_r = denoise_hl[0,:,:]
denoise_g = denoise_hl[1,:,:]
denoise_b = denoise_hl[2,:,:]
noisy_flash_r = denoise_hl[0,:,:]
noisy_flash_g = denoise_hl[1,:,:]
noisy_flash_b = denoise_hl[2,:,:]

output_gllf = np.empty_like(denoise)
output_llf_r = np.empty_like(denoise_hl[0,:,:])
output_llf_g = np.empty_like(denoise_hl[0,:,:])
output_llf_b = np.empty_like(denoise_hl[0,:,:])

noisy_flash = np.ascontiguousarray(noisy_flash.transpose(2,0,1)) * 2
output_gllf = np.ascontiguousarray(output_gllf.transpose(2,0,1))
denoise = np.ascontiguousarray(denoise.transpose(2,0,1))

#sigma=1, beta=0, alpha=0, level=5
#sigma=0, beta=1, alpha=0, level=5
#sigma=0, beta=0, alpha=0,0.25,0.5,0.75,1, level=3,4,5,6
print('h')
alpha_list =  np.linspace(0,0.1,4)
sigma_list =  [0]
noise_thresh_list = [0] #np.arange(0,1,0.25)
beta_list =  [0]
levels_list = [3]#np.arange(2,8)
exposure=4
gllf_type='g'
cols=len(alpha_list)

image_size=448
local_ckpt_dir = '/home/mohammad/cvgutils/cvgutils/nn/lpips_tf2/weights/keras'
server_ckpt_dir = '/mshvol2/users/mohammad/cvgutils/cvgutils/nn/lpips_tf2/weights/keras'
ckpt_dir = local_ckpt_dir if os.path.exists(local_ckpt_dir) else server_ckpt_dir
vgg_ckpt_fn = os.path.join(ckpt_dir, 'vgg', 'exported.weights.h5')
lin_ckpt_fn = os.path.join(ckpt_dir, 'lin', 'exported.weights.h5')
lpips_net, lpips_lin = load_perceptual_models(image_size, vgg_ckpt_fn, lin_ckpt_fn)
lpips = learned_perceptual_metric_model(lpips_net, lpips_lin, image_size)
wlpips = learned_perceptual_metric_model(lpips_net, lpips_lin, image_size, 'wlpips')

def psnr(pred, gt):
    pred = pred.clip(0., 1.)
    gt = gt.clip(0., 1.)
    mse = np.mean((pred-gt)**2.0)
    psnr = -10. * np.log10(mse)
    return psnr

denoise_dict = {}
lbls_dict = {}
annotations = {}
for alpha in alpha_list:
    for sigma in sigma_list:
        for beta in beta_list:
            for levels in levels_list:
                for noise_thresh in noise_thresh_list:
                    guided_local_laplacian_Mullapudi2016(noisy_flash, denoise, levels, alpha, sigma, noise_thresh, beta, output_gllf)
                    output_gllf_out = np.ascontiguousarray(output_gllf.transpose(1,2,0))
                    noisy_flash_out = np.ascontiguousarray(noisy_flash.transpose(1,2,0))
                    denoise_out = np.ascontiguousarray(denoise.transpose(1,2,0))
                    output_gllf_out = np.ascontiguousarray(output_gllf.transpose(1,2,0))

                    wlpips_err = wlpips([output_gllf_out[None,...], np.array(ambient_gt)])[0]
                    lpips_err = lpips([output_gllf_out[None,...], np.array(ambient_gt)])[0]
                    psnr_err = psnr(output_gllf_out[None,...], np.array(ambient_gt))
                    
                    key = 'alpha_%.03f_level_%i_sigma_%.03f_noise_thresh_%.03f' % (alpha, levels, sigma, noise_thresh)
                    denoise_dict[key] = np.abs(np.copy(output_gllf_out) * 2**exposure)
                    lbls_dict[key] = '$\\Large\\alpha=%.02f, \\gamma=%.02f, \\beta=%.02f, exp=%i$' % (alpha,sigma,beta,exposure)
                    annotation = '<br>PSNR:%.3f<br>LPIPS:%.3f<br>WLPIPS:%.3f'%(psnr_err,lpips_err,wlpips_err)
                    annotations[key] = annotation


images = {'flash':noisy_flash_out, 'noisy':noisy_ambient, 'ambient':ambient_gt, 'denoised_deepfnf':denoise_out}
lbls =   {'flash':'Flash','noisy':'Noisy','ambient':'Ambient (ref.)','denoised_deepfnf':'DeepFnF'}
denoise_dict['ambient'] = ambient_gt
lbls_dict['ambient'] = 'Ambient (ref.)'
cols=len(lbls_dict.keys())
logger.addImage(images, lbls,'gllf_%s_beta_%.02f'%(gllf_type, beta),image_filename=str(data_gt['filename']), font_size_scale=2,vertical_spacing_scale=3)
logger.addImage(denoise_dict, lbls_dict,'gllf_%s_beta_%.02f_gllf_out'%(gllf_type, beta),annotation=None,cols=cols,image_filename=str(data_gt['filename']), font_size_scale=2,vertical_spacing_scale=3)
