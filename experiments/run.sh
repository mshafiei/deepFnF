#!/bin/bash
#conda create -n deepfnf39 python=3.9
#conda init
#conda activate deepfnf39
#python3 -m pip install tensorflow[and-cuda]
#pip3 install natsort kaleido Jinja2 IPython pynvml h5py scikit-image clu matplotlib wandb piq
#pip3 install plotly pandas pdflatex opencv-python
#sudo /root/miniconda3/etc/profile.d/conda.sh

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
cd /mshvol2/users/mohammad/optimization/deepfnf_fork
# conda activate deepfnf
# pip3 install scikit-image
python3 -c """import imageio
imageio.plugins.freeimage.download()
"""
#pip3 install pynvml IPython
#pip install --upgrade "jax[cpu]"
#pip3 install -U jax[cuda12_cudnn89] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#pip3 install -U jaxlib[cuda12.cudnn89] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

export PYTHONPATH=`pwd`:/mshvol2/users/mohammad/cvgutils/:/mshvol2/users/mohammad/optimization/deepfnf_fork/lpips-tensorflow/
#pip3 install pynvml
echo command:
echo python3 $@
python3 $@ | tee /mshvol2/users/mohammad/optimization/deepfnf_fork/output.txt