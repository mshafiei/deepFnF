export LC_ALL=C.UTF-8
export LANG=C.UTF-8
cd /mshvol2/users/mohammad/optimization/deepfnf_fork
# conda activate deepfnf
# pip3 install scikit-image
python3 -c """import imageio
imageio.plugins.freeimage.download()
"""
export PYTHONPATH=`pwd`:/mshvol2/users/mohammad/cvgutils/
pip3 install pynvml
echo command:
echo python3 $@
python3 $@