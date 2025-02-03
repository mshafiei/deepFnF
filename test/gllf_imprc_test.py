from arguments_deepfnf import parse_arguments_deepfnf
parser = parse_arguments_deepfnf()
opts = parser.parse_args()
import numpy as np
import tensorflow as tf
import Viz as viz
import CLIUtils as cliuils
import gllf.gllf_layer as gllf_layer
from gllf.gllf_utils import *
tf.config.run_functions_eagerly(True)

logger = viz.logger(opts)
suffix=''
suffix += 'nostd' if opts.std_input == False else ''
suffix += 'clamp' if opts.clamp_dataset == True else ''
overfit_example_gt_data_fn = './overfit_example_data_gt%s.pkl' % suffix
overfit_example_noisy_data_fn = './overfit_example_data_noisy%s.pkl' % suffix

print('loaded example from file')
data = {}
data_gt = viz.load_pickle(overfit_example_gt_data_fn)
data_noisy = viz.load_pickle(overfit_example_noisy_data_fn)
net_input = data_noisy['net_input']
alpha = data_noisy['alpha']
noisy_flash = data_noisy['noisy_flash']
noisy_ambient = data_noisy['noisy_ambient']

noisy_ambient = tfu.camera_to_rgb(
            noisy_ambient / alpha, data_gt['color_matrix'], data_gt['adapt_matrix'],do_gamma_correct=False)

noisy_flash = tfu.camera_to_rgb(
            noisy_flash, data_gt['color_matrix'], data_gt['adapt_matrix'],do_gamma_correct=False)

data.update(data_noisy)
data.update(data_gt)

# net_input, alpha, noisy_flash, noisy_ambient = prepare_input(data, clamp=False, std_input=True)

layer = cliuils.CreateClassObjectWithOptions(opts, gllf_layer.gllf_layer_radial)
denoised, intensity_images = layer.gllf_diffable_1d([noisy_ambient],debug_reconstruct_all_remapping_images=True)
denoised = tfu.gamma_correct(denoised)
noisy_ambient = tfu.gamma_correct(noisy_ambient)
visualization = layer.visualize()
img = {"noisy_ambient":noisy_ambient,"denoised":denoised}
lbl = {"noisy_ambient":"noisy_ambient","denoised":"denoised"}
for i, v in enumerate(visualization):
    intensity_img = tfu.gamma_correct(intensity_images[i])
    img.update({v.key:v.image, v.key + ' image':intensity_img})
    lbl.update({v.key:v.label, v.key + ' image':v.label + ' image'})
logger.addImage(img, lbl, "w3_3",cols=2)
logger.takeStep()
#plot input image, denoised, PSNR, LPIPS, and the remapping functions

# The function varies from <0 to >1
print('hi')