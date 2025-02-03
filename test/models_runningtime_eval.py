
#evaluate unet, bpn, deepfnf, ours
#how running time changes w/ image resolution?
import tiny_unet_adjustable_fnf
import deepfnf_adjustable
import deepfnf_adjustable_bpn
import tensorflow as tf
import timeit
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
tf.config.run_functions_eagerly(True)

@tf.function
def exec_model(model, input):
    return model.forward(input)

def eval_latency(fn,fn_name,timing_iterations=3):
    t = timeit.Timer(fn, setup=fn)
    avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
    print("function %s took %fms" % (fn_name,(avg_time_sec * 1e3)))
    return avg_time_sec

net_ft_input = tf.ones((1,448,448,3),dtype=tf.float32)
inpt = edict(net_ft_input=net_ft_input)
unet_output_size=3

channels_count_factors=[1.0, 0.9, 0.8, 0.7, 0.5,  1., 1.]
downsample_ct        = [0,    0,   0,   0,   0,   1, 3]

samples = edict()
samples.unet = []
samples.deepfnf = []
samples.bpn = []

for downsample, channels_count_factor in zip(downsample_ct, channels_count_factors):
    unet_gllf = tiny_unet_adjustable_fnf.Net(downsample, unet_output_size=3, channels_count_factor=channels_count_factor)
    unet = tiny_unet_adjustable_fnf.Net(downsample, unet_output_size=3, channels_count_factor=channels_count_factor)
    deepfnf = deepfnf_adjustable.Net(downsample_ct=downsample, unet_output_size=3, num_basis=90, ksz=15, burst_length=2, channels_count_factor=channels_count_factor)
    bpn = deepfnf_adjustable_bpn.Net(downsample_ct=downsample, unet_output_size=3, num_basis=90, ksz=15, burst_length=2, channels_count_factor=channels_count_factor)
    bpn_t = eval_latency(lambda:exec_model(bpn, inpt), "bpn_downsample_%i")
    deepfnf_t = eval_latency(lambda:exec_model(deepfnf, inpt), "deepfnf_downsample_%i")
    unet_t = eval_latency(lambda:exec_model(unet, inpt), "unet_downsample_%i")
    samples.unet.append(unet_t)
    samples.deepfnf.append(deepfnf_t)
    samples.bpn.append(bpn_t)

# plot linearly, ensure samples are close to each other and they are dense enough
plt.plot(samples.unet, [1.] * len(samples.unet), 'r')
plt.scatter(samples.unet, [1.] * len(samples.unet))
plt.plot(samples.deepfnf, [2.] * len(samples.deepfnf), 'g')
plt.scatter(samples.deepfnf, [2.] * len(samples.deepfnf))
plt.plot(samples.bpn, [3] * len(samples.bpn), 'b')
plt.scatter(samples.bpn, [3] * len(samples.bpn))
plt.legend()
plt.savefig('./latencies.png')