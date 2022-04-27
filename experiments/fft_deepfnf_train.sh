#!/bin/bash
exp_params="\
--TLIST ./data/train_1600.txt \
--VPATH ./data/valset/ \
--model deepfnf+fft"

name=msh-deepfnf-fft-train
scriptFn="train.py $exp_params"

./experiments/run_local.sh "$scriptFn"
# ./experiments/run_server.sh "$scriptFn" "$name"