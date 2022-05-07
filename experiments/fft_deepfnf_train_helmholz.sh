#!/bin/bash
exp_params="\
--TLIST ./data/train.txt \
--VPATH ./data/valset/ \
--model deepfnf+fft_helmholz \
--weight_dir ./logs/fft_log_helmholz \
--logdir ./logs \
--expname fft_log_helmholz-e-3 \
--mindelta -1.0502254"
# --expname fft_log_helmholz-e-6 \
# --mindelta -0.19587028"
# --expname fft_log_helmholz-e-7 \
# --mindelta -15.942385"
# --expname fft_log_helmholz-e-4 \
# --mindelta -9.210175" 
# --expname fft_log_helmholz-e-7 \
# --min_delta -15.942385" 
# DeviceArray(-15.942385, dtype=float32)
# --min_delta -1.0502254"
# --mindelta -2.252168"
# jnp.log(jnp.exp(jnp.double(1e-1)) - 1)
# DeviceArray(-2.252168, dtype=float32)
# jnp.log(jnp.exp(jnp.double(1e-4)) - 1)
# DeviceArray(-9.210175, dtype=float32)
name=msh-deepfnf-helmholz-e-3
scriptFn="train.py $exp_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name"