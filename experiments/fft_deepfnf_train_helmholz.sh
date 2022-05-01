#!/bin/bash
exp_params="\
--TLIST ./data/train_1600.txt \
--VPATH ./data/valset/ \
--model deepfnf+fft_helmholz \
--weight_dir ./logs/fft_log_helmholz \
--logdir ./logs \
--expname fft_log_helmholz --store_params"

name=msh-deepfnf-fft-train-init1
scriptFn="train.py $exp_params"

./experiments/run_local.sh "$scriptFn"
# ./experiments/run_server.sh "$scriptFn" "$name"