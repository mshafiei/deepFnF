#!/bin/bash
exp_params="\
--TLIST ./data/train.txt \
--VPATH ./data/valset/ \
--model deepfnf+fft \
--weight_dir ./logs/fft \
--logdir ./logs \
--expname fft \
--visualize_freq 50001"

priority='normal'
name=msh-deepfnf-fft-train
scriptFn="train.py $exp_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name" $priority