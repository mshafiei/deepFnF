#!/bin/bash
exp_params="\
--TLIST ./data/train.txt \
--VPATH ./data/valset/ \
--weight_dir ./logs/deepfnf \
--model deepfnf \
--logdir ./logs/ \
--expname fft_deepfnf"
priority='normal'
name=msh-deepfnf-retrain
scriptFn="train.py $exp_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name" $priority