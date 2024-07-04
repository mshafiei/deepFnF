#!/usr/bin/env python3

from arguments_deepfnf import parse_arguments_deepfnf
parser = parse_arguments_deepfnf()
opts = parser.parse_args()
import os

import numpy as np
import tensorflow as tf

from utils.dataset_prefetch import TrainSet as TrainSet_prefetch
from utils.dataset import Dataset
import utils.tf_utils as tfu
import utils.utils as ut
import cvgutils.Viz as Viz

TLIST = opts.TLIST
VPATH = opts.VPATH
BSZ = 1
IMSZ = 448
LR = 1e-4
DROP = (1.1e6, 1.25e6) # Learning rate drop
MAXITER = 1.5e6
displacement = opts.displacement
VALFREQ = opts.val_freq
SAVEFREQ = opts.save_freq

logger = Viz.logger(opts,opts.__dict__)

# # pick any image that is not in the training set
# trainfn = '/home/mohammad/Projects/deepfnftf2/data/train_2.txt'
# with open(trainfn, 'r') as fd:
#     train_lines = fd.readlines()

# dr = '/home/mohammad/Projects/DifferentiableSolver/data/merged'
# fns = os.listdir(dr)
# fullfns = []

# for i in range(len(fns)):
#     if('Shelves_057' in fns[i]):
#         continue
#     if('_flash.png' in fns[i]):
#         filename = os.path.join(dr,fns[i]).replace('_flash.png','\n')
#         if(not(filename in train_lines)):
#             fullfns.append(filename)

# valfn = '/home/mohammad/Projects/deepfnftf2/data/val.txt'
# with open(valfn, 'w') as fd:
#     fd.writelines(fullfns)

dr_testset = '/home/mohammad/Projects/DifferentiableSolver/data/testsets_nojitter_extreme_noise'
#generate the validation set
#read from the dataset
# example = dataset.get_next()

alphas = [0.001,0.002,0.004,0.01,0.02,.04,0.08]
for i, alpha in enumerate(alphas):
    dataset = TrainSet_prefetch(VPATH, bsz=BSZ, psz=IMSZ, ngpus=opts.ngpus, nthreads=4 * opts.ngpus,jitter=opts.displacement,min_scale=opts.min_scale,max_scale=opts.max_scale,theta=opts.max_rotate)
    testset_dir = os.path.join(dr_testset,str(i))
    if(not os.path.exists(testset_dir)):
        os.makedirs(testset_dir)
    for j, example in enumerate(dataset.iterator):
        if(j == dataset.length):
            break
    # for j in range(length):
        # example = dataset.get_next()
        #add noise
        example['alpha'] = alpha
        dimmed_ambient, _ = tfu.dim_image(
            example['ambient'], alpha=alpha)
        dimmed_warped_ambient, _ = tfu.dim_image(
            example['warped_ambient'], alpha=alpha)

        dimmed_ambient = tf.cast(dimmed_ambient,np.float32)
        dimmed_warped_ambient = tf.cast(dimmed_warped_ambient,np.float32)
        # Make the flash brighter by increasing the brightness of the
        # flash-only image.
        flash = example['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
        warped_flash = example['warped_flash_only'] * \
            ut.FLASH_STRENGTH + dimmed_warped_ambient

        sig_read = example['sig_read']
        sig_shot = example['sig_shot']
        example['noisy_ambient'], _, _ = tfu.add_read_shot_noise(
            dimmed_ambient, sig_read=sig_read, sig_shot=sig_shot)
        example['noisy_warped_flash'], _, _ = tfu.add_read_shot_noise(
            warped_flash, sig_read=sig_read, sig_shot=sig_shot)
        
        noisy_ambient_wb = tfu.camera_to_rgb(
            example['noisy_ambient']/alpha, example['color_matrix'], example['adapt_matrix'])
        ambient_wb = tfu.camera_to_rgb(
            example['ambient'], example['color_matrix'], example['adapt_matrix'])
        noisy_warped_flash_wb = tfu.camera_to_rgb(
            example['noisy_warped_flash'], example['color_matrix'], example['adapt_matrix'])

        if(j < 2):
            im = {'noisy_ambient':noisy_ambient_wb[0],'ambient':ambient_wb[0],'noisy_warped_flash':noisy_warped_flash_wb[0]}
            lbl = {'noisy_ambient':r'$noisy_ambient_{%.3f}$'%alpha,'ambient':r'$ambient$','noisy_warped_flash':r'$noisy_warped_flash$'}
            logger.addImage(im,lbl,'deepfnf',dim_type='HWC',addinset=False,ltype='Jupyter',mode='test')
            logger.takeStep()
        
        
        fn = os.path.join(testset_dir,str(j)+'.npz')
        np.savez(fn,**example)

    
        


#write to npz files