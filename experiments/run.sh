#!/bin/bash
cd /mshvol2/users/mohammad/optimization/deepfnf_fork
cp /root/ssh_mount/id_rsa* /root/.ssh/
chmod 400 ~/.ssh/id_rsa
apt update
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt install -y python3.7
source ./venv/bin/activate
apt-get install python3.7-distutils
pip install --upgrade pip
pip3 install imageio tensorflow==1.13.1 tensorflow-gpu==1.13.1 scikit-image==0.15.0 tqdm
sudo apt-get -y install exiftool
pip3 install PyExifTool piq lpips plotly==5.6.0 pandas kaleido
python3 -c """import imageio
imageio.plugins.freeimage.download()
"""
export PYTHONPATH=`pwd`:/mshvol2/users/mohammad/cvgutils/
echo command:
echo $@
python3 $@