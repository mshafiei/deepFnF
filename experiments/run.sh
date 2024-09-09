export LC_ALL=C.UTF-8
export LANG=C.UTF-8
cd /mshvol2/users/mohammad/optimization/deepfnf_fork
pip3 install imageio
python3 -c """import imageio
imageio.plugins.freeimage.download()
"""

export PYTHONPATH=`pwd`:/mshvol2/users/mohammad/cvgutils/:/mshvol2/users/mohammad/optimization/deepfnf_fork/lpips-tensorflow/:/mshvol2/users/mohammad/optimization/halide_mirror/build/python_bindings/apps
echo command:
echo python3 $sh_params
python3 $sh_params | tee /mshvol2/users/mohammad/optimization/deepfnf_fork/output.txt