#!/bin/bash
exp_params="\
--TLIST ./data/train_1600.txt \
--VPATH ./data/valset/ \
--model deepfnf+fft \
--weight_dir fft_log_init1"

name=msh-deepfnf-fft-train-init1
scriptFn="train.py $exp_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name"