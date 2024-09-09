#!/bin/bash
. "/miniconda3/etc/profile.d/conda.sh"
conda activate deepfnf
sh_params=$@
echo executing run.sh with arguments: $sh_params

while [ $# -ge 1 ]; do
        case "$1" in
                --)
                    # No more options left.
                    shift
                    break
                   ;;
                --use_gpu)
                        use_gpu="$2"
                        shift
                        ;;
        esac
        shift
done

echo use_gpu in terminal $use_gpu

if [[ $use_gpu == "False" ]]; then
echo disabling gpu
export CUDA_VISIBLE_DEVICES=""
fi

cd /mshvol2/users/mohammad/optimization/deepfnf_fork
pip3 install imageio
python3 -c """import imageio
imageio.plugins.freeimage.download()
"""

export PYTHONPATH=`pwd`:/mshvol2/users/mohammad/cvgutils/:/mshvol2/users/mohammad/optimization/deepfnf_fork/lpips-tensorflow/:/mshvol2/users/mohammad/optimization/halide_mirror/build/python_bindings/apps
echo command:
echo python3 $sh_params
python3 $sh_params | tee /mshvol2/users/mohammad/optimization/deepfnf_fork/output.txt