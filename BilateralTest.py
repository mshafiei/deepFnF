from skimage.io import imread, imsave
import os
from bilateral import bilateralFilter, bilateralSolve
import numpy as np




reference = imread('/home/mohammad/Projects/deepFnF/test_imgs/reference.png')
target = imread('/home/mohammad/Projects/deepFnF/test_imgs/target.png').astype(np.double) / (pow(2,16)-1)
# target = np.stack([target,target,target],axis=-1)
# output_filter = bilateralFilter(target, reference)
output_filter = bilateralSolve(target, reference)

imsave('/home/mohammad/Projects/deepFnF/test_imgs/output.png', (output_filter * (pow(2,16)-1)).astype(np.uint16))
print('hi')