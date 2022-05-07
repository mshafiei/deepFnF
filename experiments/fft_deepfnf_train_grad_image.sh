#!/bin/bash
exp_params="\
--TLIST ./data/train.txt \
--VPATH ./data/valset/ \
--model deepfnf+fft_grad_image \
--weight_dir ./logs/fft_log_grad_image \
--logdir ./logs \
--expname fft_log_grad_image --store_params"

name=msh-deepfnf-fft-train-init1
scriptFn="train.py $exp_params"

./experiments/run_local.sh "$scriptFn"
# ./experiments/run_server.sh "$scriptFn" "$name"