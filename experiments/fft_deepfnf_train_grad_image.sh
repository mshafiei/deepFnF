#!/bin/bash
exp_params="\
--TLIST ./data/train.txt \
--VPATH ./data/valset/ \
--model deepfnf+fft_grad_image \
--weight_dir ./logs/fft_grad_image \
--logdir ./logs \
--expname fft_grad_image \
--visualize_freq 50001"

priority='normal'
name=msh-deepfnf-fft-train-grad-image
scriptFn="train.py $exp_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name" $priority