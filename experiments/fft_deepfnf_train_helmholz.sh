#!/bin/bash

name=msh-deepfnf-helmholz-phie2-psie1
lambda="--expname fft_log_helmholz-phie2-psie1 --min_lmbda_phi 0.01 --min_lmbda_psi 1."
# name=msh-deepfnf-helmholz-phi1-psie2e1
# lambda="--expname fft_log_helmholz-phie1-psie2 --min_lmbda_phi 1. --min_lmbda_psi .01"
# name=msh-deepfnf-helmholz-phi1-psi1-fixed
# lambda="--expname fft_log_helmholz-phie1-psie1-fixed --min_lmbda_phi 1. --min_lmbda_psi 1. --fixed_lambda"
# name=msh-deepfnf-helmholz-phi1-psi1
# lambda="--expname fft_log_helmholz-phie1-psie1 --min_lmbda_phi 1. --min_lmbda_psi 1."

priority='normal'

exp_params="\
--TLIST ./data/train.txt \
--VPATH ./data/valset/ \
--model deepfnf+fft_helmholz \
--weight_dir ./logs/fft_log_helmholz-fixed \
--logdir ./logs $lambda"

scriptFn="train.py $exp_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name" $priority