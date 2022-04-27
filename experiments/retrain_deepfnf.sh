#!/bin/bash
exp_params="\
--TLIST ./data/train_1600.txt \
--VPATH ./data/valset/ \
--model deepfnf \
--weight_dir wts"

name=msh-deepfnf-retrain1
scriptFn="train.py $exp_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name"