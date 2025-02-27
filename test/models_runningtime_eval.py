
#evaluate unet, bpn, deepfnf, ours
#how running time changes w/ image resolution?
import tiny_unet_adjustable_fnf
import deepfnf_adjustable
import deepfnf_adjustable_bpn
import tensorflow as tf
import timeit
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import tiny_unet_alpha_gllf
import os
import cvgutils.Viz as viz
# tf.config.run_functions_eagerly(True)

# @tf.function
def exec_model(model, input):
    return model.forward(input)

def eval_latency(fn,fn_name,timing_iterations=3):
    t = timeit.Timer(fn, setup=fn)
    avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
    print("function %s took %fms" % (fn_name,(avg_time_sec * 1e3)))
    return avg_time_sec

net_ft_input = tf.ones((1,448,448,12),dtype=tf.float32)
ambient = tf.ones((1,448,448,3),dtype=tf.float32)
inpt = edict(net_ft_input=net_ft_input, ambient=ambient, alpha=tf.convert_to_tensor([1.]), 
             color_matrix=tf.convert_to_tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])[None,...], 
             adapt_matrix=tf.convert_to_tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])[None,...])
unet_output_size=3

latency_fn = './latency.json'
error_json = '/home/mohammad/Projects/deepfnftf2/across_methods_error.json'
latency = {}

channels_count_factors=[1.0, 0.9, 0.8, 0.7, 0.5,  1., 1.]
downsample_ct        = [0,    0,   0,   0,   0,   1, 3]

samples = edict()
samples.unet_gllf = []
samples.unet = []
samples.deepfnf = []
samples.bpn = []
samples.unet_gllf_error = []
samples.unet_error = []
samples.deepfnf_error = []
samples.bpn_error = []

error_type = 'psnr'
darkening_mode = 'Level 4'

errors = None
if(os.path.exists(error_json)):
    errors = viz.load_json(error_json)

if(os.path.exists(latency_fn)):
    samples = edict(viz.load_json(latency_fn))

for i, (downsample, channels_count_factor) in enumerate(zip(downsample_ct, channels_count_factors)):
    if(i < len(samples.unet_gllf)):
        continue
    key = '%s_%s' % (downsample, str(channels_count_factor))
    if(key in errors.keys()):
        error = errors[key]
    else:
        continue

    deepfnf_adjustable_bpn_error = 0
    deepfnf_adjustable_error = 0
    tiny_unet_alpha_gllf_error = 0
    tiny_unet_adjustable_fnf_error = 0
    if('tiny_unet_alpha_gllf' in errors.keys() and key in errors['tiny_unet_alpha_gllf'].keys()):
        tiny_unet_alpha_gllf_error = max(errors['tiny_unet_alpha_gllf'][key][darkening_mode][error_type],0)
    if('tiny_unet_adjustable_fnf' in errors.keys() and key in errors['tiny_unet_adjustable_fnf'].keys()):
        tiny_unet_adjustable_fnf_error = max(errors['tiny_unet_adjustable_fnf'][key][darkening_mode][error_type],0)
    if('deepfnf_adjustable' in errors.keys() and key in errors['deepfnf_adjustable'].keys()):
        deepfnf_adjustable_error = max(errors['deepfnf_adjustable'][key][darkening_mode][error_type],0)
    if('deepfnf_adjustable_bpn' in errors.keys() and key in errors['deepfnf_adjustable_bpn'].keys()):
        deepfnf_adjustable_bpn_error = max(errors['deepfnf_adjustable_bpn'][key][darkening_mode][error_type],0)

    # llf_levels, llf_intensity_levels, llf_remap_function, rbf_weights_ct, yuv_gllf, alphas, betas, sigmas, thresholds, downsample_ct, use_halide_implementation=False, img_ct=None, gaussian_weights_scale=2,gaussian_sigma_offset=3, piecewise_linear_weight_max=3, piecewise_linear_sigma=0.2, basis_ct=1,unet_output_size=6, min_intensity=0.0, max_intensity=1.0, IMSZ=448
    # downsample_ct=0, unet_output_size=3, channels_count_factor=1
    unet_gllf = tiny_unet_alpha_gllf.Net(downsample_ct=downsample, channels_count_factor=channels_count_factor, input_images=["noisy_ambient","noisy_flash", "deep_denoised"], llf_levels=4, llf_intensity_levels=4, llf_remap_function="gaussian_1d", rbf_weights_ct=8, yuv_gllf="false", alphas=1.0, betas=1.0, sigmas=1.0, thresholds=None, use_halide_implementation=True)
    unet = tiny_unet_adjustable_fnf.Net(downsample, unet_output_size=3, channels_count_factor=channels_count_factor)
    deepfnf = deepfnf_adjustable.Net(downsample_ct=downsample, unet_output_size=3, num_basis=90, ksz=15, burst_length=2, channels_count_factor=channels_count_factor)
    bpn = deepfnf_adjustable_bpn.Net(downsample_ct=downsample, unet_output_size=3, num_basis=90, ksz=15, burst_length=2, channels_count_factor=channels_count_factor)
    unet_gllf_t = eval_latency(lambda:exec_model(unet_gllf, inpt), "unet_gllf_downsample_%i" % downsample)
    bpn_t = eval_latency(lambda:exec_model(bpn, inpt), "bpn_downsample_%i" % downsample)
    deepfnf_t = eval_latency(lambda:exec_model(deepfnf, inpt), "deepfnf_downsample_%i" % downsample)
    unet_t = eval_latency(lambda:exec_model(unet, inpt), "unet_downsample_%i" % downsample)
    
    if(tiny_unet_alpha_gllf_error):
        samples.unet_gllf.append(unet_gllf_t)
        samples.unet_gllf_error.append(tiny_unet_alpha_gllf_error)
    if(tiny_unet_adjustable_fnf_error):
        samples.unet.append(unet_t)
        samples.unet_error.append(tiny_unet_adjustable_fnf_error)
    if(deepfnf_adjustable_error):
        samples.deepfnf.append(deepfnf_t)
        samples.deepfnf_error.append(deepfnf_adjustable_error)
    if(deepfnf_adjustable_error):
        samples.bpn.append(bpn_t)
        samples.bpn_error.append(deepfnf_adjustable_bpn_error)
    viz.dumpDictJson(samples, latency_fn)

# plot linearly, ensure samples are close to each other and they are dense enough
plt.plot(samples.unet_gllf, samples.unet_gllf_error, 'r')
plt.scatter(samples.unet_gllf, samples.unet_gllf_error)
plt.plot(samples.unet, samples.unet_error, 'r')
plt.scatter(samples.unet, samples.unet_error)
plt.plot(samples.deepfnf, samples.deepfnf_error, 'g')
plt.scatter(samples.deepfnf, samples.deepfnf_error)
plt.plot(samples.bpn, samples.bpn_error, 'b')
plt.scatter(samples.bpn, samples.bpn_error)
plt.legend(['unet_gllf','unet_gllf','unet','unet','deepfnf','deepfnf','bpn','bpn'])
plt.savefig('./latencies.png')