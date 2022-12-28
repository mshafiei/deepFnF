#!/bin/bash
cd /mshvol2/users/mohammad/optimization/deepfnf_fork
# conda activate deepfnf
# pip3 install scikit-image
python3 -c """import imageio
imageio.plugins.freeimage.download()
"""
export PYTHONPATH=`pwd`:/mshvol2/users/mohammad/cvgutils/
echo command:
echo python3 $@
python3 $@