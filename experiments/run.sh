#!/bin/bash
cd /mshvol2/users/mohammad/optimization/deepfnf_fork
cp /root/ssh_mount/id_rsa* /root/.ssh/
chmod 400 ~/.ssh/id_rsa
apt update
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt install -y python3.7 python3.7-distutils python3.7-venv exiftool
# source ./venv/bin/activate
# pip install --upgrade pip
python3.7 -m pip install --upgrade pip
python3.7 -m pip install setuptools 
python3.7 -m pip install imageio tensorflow-gpu==1.13.1 scikit-image==0.16.2 tqdm PyExifTool piq lpips plotly==5.6.0 pandas kaleido
python3.7 -c """import imageio
imageio.plugins.freeimage.download()
"""
export PYTHONPATH=`pwd`:/mshvol2/users/mohammad/cvgutils/
echo command:
echo python3.7 $@
python3.7 $@