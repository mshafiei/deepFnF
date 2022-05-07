#!/bin/bash
exp_params="\
--TLIST ./data/train.txt \
--VPATH ./data/valset/ \
--model deepfnf+fft \
--weight_dir ./logs/fft_log_init1 \
--logdir ./logs \
--expname fft_log_init1"

name=msh-deepfnf-fft-train-init1-org
scriptFn="train.py $exp_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name"