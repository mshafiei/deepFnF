#!/bin/bash
cp /root/ssh_mount/id_rsa* /root/.ssh/
chmod 400 ~/.ssh/id_rsa
source ./venv/bin/activate
pip install --upgrade pip
pip3 install imageio tensorflow==1.13.1 tensorflow-gpu==1.13.1 scikit-image tqdm
sudo apt-get -y install exiftool
pip3 install PyExifTool piq lpips plotly==5.6.0 pandas kaleido
python3 -c """import imageio
imageio.plugins.freeimage.download()
"""
cd /mshvol2/users/mohammad/optimization/DifferentiableSolver
export PYTHONPATH=`pwd`:/mshvol2/users/mohammad/cvgutils/
echo command:
echo $@