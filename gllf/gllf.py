from arguments_deepfnf import parse_arguments_deepfnf
parser = parse_arguments_deepfnf()
opts = parser.parse_args()
import tensorflow as tf
import imageio
import cv2
import cvgutils.Viz as viz
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
import timeit
import tensorflow.experimental.numpy as tnp
import tqdm
import os
# tf.config.run_functions_eagerly(True)
logr = viz.logger(opts=opts)
def _downsample(im):
    """Downsample with a 1 3 3 1 filter"""
    #im is padded with [1, 2], [1, 2] with constants
    assert len(im.shape) == 4
    h, w = im.shape[1], im.shape[2]
    x0,x1 = 1,w
    y0,y1 = 1,h
    im = tf.pad(im,[[0,0],[1,2],[1,2],[0,0]],mode='REFLECT')
    downx = (
        im[:, :, x0-1:x1:2, :] +
        3.0 * (im[:, :, x0:x1:2, :] + im[:, :, x0+1:x1+1:2, :]) +
        im[:, :, x0+2:x1+2:2, :]
    ) / 8.0

    downy = (
        downx[:, y0-1:y1:2, :, :] +
        3.0 * (downx[:, y0:y1:2, :, :] + downx[:, y0+1:y1+1:2, :, :]) +
        downx[:, y0+2:y1+2:2, :, :]
    ) / 8.0

    return downy


def _upsample(im):
    """Upsample using bilinear interpolation"""
    assert len(im.shape) == 4
    h, w = im.shape[1], im.shape[2]

    im = tf.image.resize(
        im,
        (h*2,w*2),
        method=tf.image.ResizeMethod.BILINEAR,
        preserve_aspect_ratio=True,
        antialias=False,
        name=None
    )

    return im

def _resize(im, target_size):
    h, _ = im.shape[1:3]
    th, _ = target_size
    ratio = np.log2(h//th) if h >= th else np.log2(th//h)
    upsample = h < th
    if(upsample):
        for i in range(int(ratio)):
            im = _upsample(im)
    else:
        for i in range(int(ratio)):
            im = _downsample(im)
    
    return im

def LaplacianPyramid(gPyramid):
    lPyramid = [0] * len(gPyramid)
    J = len(gPyramid)
    lPyramid[J - 1] = gPyramid[J - 1]
    for j in range(J - 2, -1, -1):
        lPyramid[j] = gPyramid[j] - _upsample(gPyramid[j + 1])
    return lPyramid

def GaussianPyramid(im, J):
    gPyramid = [im]
    for j in range(1, J):
        gPyramid.append(_downsample(gPyramid[j - 1]))
    return gPyramid

def slice(l_i_i, l_g_i, l, IMSZ=448):
    x, y, c = np.meshgrid(np.arange(IMSZ),np.arange(IMSZ),np.arange(3))
    l_i_i = tf.squeeze(l_i_i)
    l_g_i = tf.squeeze(l_g_i)
    idx = tf.stack((l_i_i,l_g_i,y,x,c),axis=-1)[None, None]
    l = tf.squeeze(l)
    return tf.gather_nd(l, idx)

@tf.function
def gllf(im_i, im_g, im_i_pyramid, im_g_pyramid, max_levels, max_discrete_levels, alpha_h, alpha_i, beta=1, sigma=1, IMSZ=448):
    im_i_pyramid = tf.stop_gradient(im_i_pyramid)
    im_g_pyramid = tf.stop_gradient(im_g_pyramid)
    #input and guide pyramids
    G_i = GaussianPyramid(im_i_pyramid, max_levels)
    G_g = GaussianPyramid(im_g_pyramid, max_levels)
    
    idx_guide = im_g #* (max_discrete_levels - 1) #* 256.0
    # idx_guide = tfp.math.clip_by_value_preserve_gradient(idx_guide, 0, (max_discrete_levels - 1) * 256)
    # idx_guide = tf.cast(idx_guide, dtype=tf.float32)
    # compute remapped images and its pyramids
    lpyramid = []
    for k_i in range(max_discrete_levels):
        l_i = []
        for k_g in range(max_discrete_levels):
            r_i_j = remapping_2d(im_i, idx_guide, k_i, k_g, max_discrete_levels, sigma, beta, alpha_h, alpha_i)
            #compute gaussian and laplacian of remapped images
            f_i_j_g = GaussianPyramid(r_i_j, max_levels)
            
            f_i_j_l = LaplacianPyramid(f_i_j_g)

            for i in range(len(f_i_j_l)):
                f_i_j_l[i] = _resize(f_i_j_l[i], im_i.shape[1:3])
            
            l_g_images = tf.stack(f_i_j_l, axis=0)
            l_i.append(l_g_images) #l x 1 x h x w x 3
        
        lpyramid.append(tf.stack(l_i,axis=1))#l x k_g x 1 x h x w x 3
    lpyramid = tf.stack(lpyramid,axis=1) #l x k_i x k_g x 1 x h x w x 3
    
    outLPyramid = []
    G_i_resized = [_resize(i * (max_discrete_levels - 1),(IMSZ, IMSZ)) for i in G_i]
    G_g_resized = [_resize(i * (max_discrete_levels - 1),(IMSZ, IMSZ)) for i in G_g]
    for i in range(max_levels):
        #fetch i and g pixels and discretize
        l_i = G_i_resized[i]
        l_i_i = tf.clip_by_value(tf.cast(l_i,dtype=tf.int32),0, max_discrete_levels-2)
        l_i_f = l_i - tf.cast(l_i_i,dtype=tf.float32)
        l_g = G_g_resized[i]
        l_g_i = tf.clip_by_value(tf.cast(l_g,dtype=tf.int32), 0, max_discrete_levels-2)
        l_g_f = l_g - tf.cast(l_g_i,dtype=tf.float32)

        # make laplacian pyramid by interpolation
        outLPyramid_i = (1 - l_i_f) * (1 - l_g_f) * slice(l_i_i,     l_g_i,     lpyramid[i],IMSZ=IMSZ)        + \
                        (    l_i_f) * (1 - l_g_f) * slice(l_i_i + 1, l_g_i,     lpyramid[i],IMSZ=IMSZ)        + \
                        (1 - l_i_f) * (    l_g_f) * slice(l_i_i,     l_g_i + 1, lpyramid[i],IMSZ=IMSZ)        + \
                        (    l_i_f) * (    l_g_f) * slice(l_i_i + 1, l_g_i + 1, lpyramid[i],IMSZ=IMSZ)

        outLPyramid.append(tf.squeeze(outLPyramid_i))
    
    # return outLPyramid
    #collapse pyramid
    g = [0] * max_levels
    g[max_levels - 1] = outLPyramid[max_levels - 1]
    for j in range(max_levels - 2, -1, -1):
        g[j] = g[j + 1] + outLPyramid[j]
    
    #clip for visualization
    # for j in range(max_levels):
    #     g[j] = tf.abs(tfp.clip_by_value(g[j] * 1,0,1))
    
    return g[0][None,...]

def remap(fx, alpha):
    # fx = fx * 256
    return alpha * fx * tf.exp(-fx * fx / 2.0)

def remapping_2d(i, g, k_i, k_g, n_levels, sigma, beta, alpha_h, alpha_i):
    level_input = k_i / (n_levels - 1)
    level_guide = k_g / (n_levels - 1)
    return sigma * level_input + beta * (i - level_input) + remap(g - level_guide, alpha_h) + remap(i - level_input, alpha_i)


@tf.function
def gllf_diffable_2d(im_i, im_g, max_levels, max_discrete_levels, alpha_i, alpha_h, beta=1, sigma=1, IMSZ=448):
    lpyramid = images_to_lookup_2d(im_i, im_g, max_levels, max_discrete_levels, alpha_h, alpha_i, beta=1, sigma=1, IMSZ=448)
        
    #input and guide pyramids
    G_i = GaussianPyramid(im_i, max_levels)
    G_g = GaussianPyramid(im_g, max_levels)
    outLPyramid = []
    for i in range(max_levels):
        outLPyramid_i = diffable_slice_2d(G_i[i], G_g[i], lpyramid[i], max_discrete_levels,IMSZ)
        outLPyramid.append(outLPyramid_i)
    
    #collapse pyramid
    g = [0] * max_levels
    g[max_levels - 1] = outLPyramid[max_levels - 1]
    for j in range(max_levels - 2, -1, -1):
        g[j] = _upsample(g[j + 1]) + outLPyramid[j]
    
    return g[0]

def resize_pyramid(im_i_pyramids, max_levels, max_discrete_levels, IMSZ):
    G_is = []
    for im_i_pyramid in im_i_pyramids:
        G_i = GaussianPyramid(im_i_pyramid, max_levels)
        G_is.append(tf.stack([_resize(i,(IMSZ, IMSZ)) for i in G_i]))
    return tf.stack(G_is,axis=0) #n, L, 1, h, w, c

def reconstruct_Laplacian(outLPyramid, max_levels):
    #collapse pyramid
    g = [0] * max_levels
    g[max_levels - 1] = outLPyramid[max_levels - 1]
    for j in range(max_levels - 2, -1, -1):
        g[j] = _upsample(g[j + 1]) + outLPyramid[j]
    return g[0]


def remapping_1d(i, k, sigma, beta, alpha, threshold, n_levels,min_intensity=0.0, max_intensity=1.0):
    level = k / (n_levels - 1)
    level = level * (max_intensity - min_intensity) + min_intensity
    diff = i - level
    if(threshold is None):
        return sigma * level + beta * diff + remap(diff, alpha)
    else:
        # compress = sigma * level + beta * diff
        compress = sigma * level + tf.sign(diff) * (beta * (tf.abs(diff)-threshold)+threshold)
        # compress = sigma * level + tf.sign(diff) * threshold * tf.pow(tf.abs(diff)/threshold, 1/alpha)
        details = sigma * level + tf.sign(diff) * threshold * tf.pow(tf.abs(diff)/threshold, alpha)
        # details = sigma * level + remap(diff, alpha)
        return tf.where(tf.abs(diff) < threshold, details, compress)

def images_to_lookup_1d(im_is, max_levels, max_discrete_levels, alphas, betas, sigmas, min_intensity, max_intensity, IMSZ=448):
    # compute remapped images and its pyramids
    lpyramid = []
    for im_i, alpha, beta, sigma in zip(im_is, alphas, betas, sigmas):
        l_i = []
        for k_i in range(max_discrete_levels):
            r_i_j = remapping_1d(im_i, k_i, sigma, beta, alpha, max_discrete_levels,min_intensity=min_intensity, max_intensity=max_intensity)
            f_i_j_g = GaussianPyramid(r_i_j, max_levels)
            f_i_j_l = LaplacianPyramid(f_i_j_g)
            for i in range(len(f_i_j_l)):
                f_i_j_l[i] = _resize(f_i_j_l[i], im_i.shape[1:3])
            l_g_images = tf.stack(f_i_j_l, axis=0)
            l_i.append(l_g_images) #l x 1 x h x w x 3
        lpyramid.append(tf.stack(l_i,axis=1))#l x L x 1 x h x w x 3
    lpyramid = tf.stack(lpyramid,axis=0) #k_i, L x L x 1 x h x w x 3

    return lpyramid

@tf.function
def images_to_lookup_1d_noresize(im_is, max_levels, max_discrete_levels, alphas, betas, sigmas, min_intensity, max_intensity, thresholds=None, IMSZ=448):
    #in this function
    #K is intensity sample count max_discrete_levels
    #L is level count max_levels
    # compute remapped images and its pyramids
    if(thresholds is None):
        thresholds = [None] * len(im_is)
    lpyramid = [[[] for _ in range(len(im_is))] for _ in range(max_levels)]
    for k_i in range(max_discrete_levels):
        for im_idx in range(len(im_is)):
            im_i, alpha, beta, sigma, threshold = im_is[im_idx], alphas[im_idx][k_i], betas[im_idx][k_i], sigmas[im_idx][k_i], thresholds[im_idx]
            r_i_j = remapping_1d(im_i, k_i, sigma, beta, alpha, threshold, max_discrete_levels,min_intensity=min_intensity, max_intensity=max_intensity)
            f_i_j_g = GaussianPyramid(r_i_j, max_levels)
            f_i_j_l = LaplacianPyramid(f_i_j_g)# K, 1, h, w, c
            for f_i_j_l_i, f_i_j_l_v in enumerate(f_i_j_l):
                lpyramid[f_i_j_l_i][im_idx].append(f_i_j_l_v) #{L}, I, K, 1, h, w, c
    
    lpyramid = [tf.stack(lpyramid[i],axis=0) for i in range(max_levels)]#{L}, I, K, 1, h, w, c
    lpyramid = [tf.transpose(i,(2,0,1,3,4,5)) for i in lpyramid]#{L}, 1, I, K, h, w, c
    return lpyramid

def remapping_1d_threshold_source(i, k, sigma, beta, alpha, n_levels, threshold):
    level = k / (n_levels - 1)
    diff = i - level
    result = tf.zeros_like(i)
    compress = sigma * level + tf.sign(diff) * (beta * (tf.abs(diff)-threshold)+threshold)
    details = sigma * level + tf.sign(diff) * threshold * tf.pow(tf.abs(diff)/threshold, alpha)
    result = tf.where(diff < threshold, details, compress)
    return result

def remapping_1d_threshold_guide(i, k, sigma, beta, alpha, n_levels, threshold):
    level = k / (n_levels - 1)
    diff = i - level
    result = tf.zeros_like(i)
    compress = sigma * level + tf.sign(diff) * (beta * (tf.abs(diff)-threshold)+threshold)
    details = sigma * level + tf.sign(diff) * threshold * tf.pow(tf.abs(diff)/threshold, alpha)
    result = tf.where(diff < threshold, details, compress*0)
    return result



def images_to_lookup_1d_noresize_threshold(im_is, max_levels, max_discrete_levels, alphas, betas, sigmas, threshold, remappings_1d, IMSZ=448):
    #in this function
    #K is intensity sample count max_discrete_levels
    #L is level count max_levels
    # compute remapped images and its pyramids
    lpyramid = [[[] for _ in range(len(im_is))] for _ in range(max_levels)]
    for k_i in range(max_discrete_levels):
        for im_idx, (im_i, alpha, beta, sigma, remapping_1d) in enumerate(zip(im_is, alphas, betas, sigmas, remappings_1d)):
            r_i_j = remapping_1d(im_i, k_i, sigma, beta, alpha, max_discrete_levels, threshold)
            f_i_j_g = GaussianPyramid(r_i_j, max_levels)
            f_i_j_l = LaplacianPyramid(f_i_j_g)# K, 1, h, w, c
            for f_i_j_l_i, f_i_j_l_v in enumerate(f_i_j_l):
                lpyramid[f_i_j_l_i][im_idx].append(f_i_j_l_v) #{L}, I, K, 1, h, w, c
    
    lpyramid = [tf.stack(lpyramid[i],axis=0) for i in range(max_levels)]#{L}, I, K, 1, h, w, c
    lpyramid = [tf.transpose(i,(2,0,1,3,4,5)) for i in lpyramid]#{L}, 1, I, K, h, w, c
    return lpyramid

# @tf.function
def images_to_lookup_2d(im_i, im_g, max_levels, max_discrete_levels, alpha_h, alpha_i, beta=1, sigma=1, IMSZ=448):    
    # compute remapped images and its pyramids
    lpyramid = [[[[] for _ in range(max_levels)] for _ in range(max_discrete_levels)] for _ in range(max_discrete_levels)]
    for k_i in range(max_discrete_levels):
        for k_g in range(max_discrete_levels):
            r_i_j = remapping_2d(im_i, im_g, k_i, k_g, max_discrete_levels, sigma, beta, alpha_h, alpha_i)
            #compute gaussian and laplacian of remapped images
            f_i_j_g = GaussianPyramid(r_i_j, max_levels)
            f_i_j_l = LaplacianPyramid(f_i_j_g)# K, 1, h, w, c
            for i, l in enumerate(f_i_j_l):
                lpyramid[i][k_i][k_g] = l
    #lpyramid shape is {L}, K, K, 1, h, w, c
    lpyramid = [[tf.stack(lpyramid[i][j],axis=0) for j in range(max_discrete_levels)] for i in range(max_levels)]
    lpyramid = [tf.stack(lpyramid[i],axis=0) for i in range(max_levels)]
    #{L}, K, K, 1, h, w, c -> {L}, 1, K, K, h, w, c
    lpyramid = [tf.transpose(lpyramid[i],(2,0,1,3,4,5)) for i in range(max_levels)]
    return lpyramid
    
@tf.function
def gllf_diffable_1d(im_is, alphas, max_levels, max_discrete_levels, betas, sigmas,thresholds=None,min_intensity=0.0, max_intensity=1.0, IMSZ=448):
    outLPyramids = images_to_lookup_1d_noresize(im_is, max_levels, max_discrete_levels, alphas, thresholds=thresholds,IMSZ=IMSZ, betas=betas, sigmas=sigmas,min_intensity=min_intensity, max_intensity=max_intensity)
    G_is = [[[] for _ in range(len(im_is))] for _ in range(max_levels)]
    for im_idx in range(len(im_is)):
        im_i_pyramid = im_is[im_idx]
    # for im_idx, im_i_pyramid in enumerate(im_is):
        G_i = GaussianPyramid(im_i_pyramid, max_levels)
        for i, g in enumerate(G_i):
            G_is[i][im_idx] = g
    G_is = [tf.stack(i,axis=0) for i in G_is] # {L}, I, 1, h, w, c
    G_is = [tf.transpose(i,(1,0,2,3,4)) for i in G_is] # {L}, 1, I, h, w, c
    
    outLPyramid = []
    for i in range(max_levels):
        #outLPyramids n, L, L, 1, h, w, c
        #G_is         n, L, 1, h, w, c
        outLPyramid_slice = diffable_slice_separable(G_is[i], outLPyramids[i], max_discrete_levels,IMSZ,min_intensity, max_intensity)
        outLPyramid.append(outLPyramid_slice)
    return reconstruct_Laplacian(outLPyramid, max_levels)

@tf.function
def gllf_diffable_1d_threshold(im_is, alphas, max_levels, max_discrete_levels, betas, sigmas, threshold, remappings_1d, IMSZ=448):
    # outLPyramids = images_to_lookup_1d(im_is, max_levels, max_discrete_levels, alphas, IMSZ=IMSZ, betas=betas, sigmas=sigmas)
    outLPyramids = images_to_lookup_1d_noresize_threshold(im_is, max_levels, max_discrete_levels, alphas, IMSZ=IMSZ, betas=betas, sigmas=sigmas, threshold=threshold, remappings_1d=remappings_1d)
    # G_is = resize_pyramid(im_is, max_levels, max_discrete_levels, IMSZ)
    G_is = [[[] for _ in range(len(im_is))] for _ in range(max_levels)]
    for im_idx, im_i_pyramid in enumerate(im_is):
        G_i = GaussianPyramid(im_i_pyramid, max_levels)
        for i, g in enumerate(G_i):
            G_is[i][im_idx] = g
    G_is = [tf.stack(i,axis=0) for i in G_is] # {L}, I, 1, h, w, c
    G_is = [tf.transpose(i,(1,0,2,3,4)) for i in G_is] # {L}, 1, I, h, w, c
    
    outLPyramid = []
    for i in range(max_levels):
        #outLPyramids n, L, L, 1, h, w, c
        #G_is         n, L, 1, h, w, c
        outLPyramid_slice = diffable_slice_separable(G_is[i], outLPyramids[i], max_discrete_levels,IMSZ)
        outLPyramid.append(outLPyramid_slice)
    return reconstruct_Laplacian(outLPyramid, max_levels)


def smooth_strict_clip(x, clip_min, clip_max, smoothing=1.0):
    """
    Smoothly clips values to the range [clip_min, clip_max] in a differentiable way
    with strict enforcement of boundaries.

    Parameters:
    - x: The input tensor.
    - clip_min: The lower boundary for clipping.
    - clip_max: The upper boundary for clipping.
    - smoothing: Controls the smoothness of the transition (higher = smoother).

    Returns:
    - A tensor with values strictly clipped between clip_min and clip_max.
    """
    # Scale x to the range [-1, 1] before applying tanh
    scale = 2.0 / (clip_max - clip_min)
    offset = (clip_max + clip_min) / 2.0
    x_scaled = (x - offset) * scale

    # Apply tanh for smooth, strict clipping and scale back to [clip_min, clip_max]
    clipped = tf.tanh(x_scaled / smoothing)
    output = (clipped / scale) + offset
    
    return output

# @tf.function
# def set_inner_slice_1d(l_i_i, l, IMSZ=448, ones=False):
#     #l shape is b, I, L, h, w, c
#     #l_i_i shape is b, I, h, w, c
    
#     b, img_ct, h, w, c_ct = l_i_i.shape
#     l_i_i_shape = l_i_i.shape
#     l_shape = l.shape
#     x = tf.convert_to_tensor(np.arange(w,dtype=np.int32))
#     y = tf.convert_to_tensor(np.arange(h,dtype=np.int32))
#     c = tf.convert_to_tensor(np.arange(c_ct,dtype=np.int32))
#     x, y, c = tnp.meshgrid(x, y, c)
#     x = tf.repeat(x[None],axis=0,repeats=img_ct*b)
#     y = tf.repeat(y[None],axis=0,repeats=img_ct*b)
#     c = tf.repeat(c[None],axis=0,repeats=img_ct*b)
#     l_i_i = tf.reshape(l_i_i,[b*img_ct]+l_i_i_shape[2:])
#     idx = tf.stack((l_i_i,y,x,c),axis=-1)
#     l = tf.reshape(l, [b * img_ct] + l_shape[2:])
#     # gathered = tf.gather_nd(l, idx,batch_dims=1)
#     # gathered_reshape = tf.reshape(gathered,(b,img_ct)+gathered.shape[1:])

#     zero_l = tf.zeros(l.shape)
#     gathered_values = tf.gather_nd(zero_l, idx, batch_dims=1)
#     update = tf.ones_like(gathered_values)
#     zero_l_update = []
#     for i in range(img_ct):
#         zero_l_update.append(tf.scatter_nd(idx[i], update[i], zero_l[i].shape))
#     zero_l_update = tf.stack(zero_l_update,axis=0)
#     zero_l_update = tf.reshape(zero_l_update, l_shape)
#     return zero_l_update
    
# @tf.function
# def inner_slice_1d(l_i_i, l, IMSZ=448):
#     #l shape is b, I, L, h, w, c
#     #l_i_i shape is b, I, h, w, c
    
#     b, img_ct, h, w, c_ct = l_i_i.shape
#     l_i_i_shape = l_i_i.shape
#     l_shape = l.shape
#     x = tf.convert_to_tensor(np.arange(w,dtype=np.int32))
#     y = tf.convert_to_tensor(np.arange(h,dtype=np.int32))
#     c = tf.convert_to_tensor(np.arange(c_ct,dtype=np.int32))
#     x, y, c = tnp.meshgrid(x, y, c)
#     x = tf.repeat(x[None],axis=0,repeats=img_ct*b)
#     y = tf.repeat(y[None],axis=0,repeats=img_ct*b)
#     c = tf.repeat(c[None],axis=0,repeats=img_ct*b)
#     l_i_i = tf.reshape(l_i_i,[b*img_ct]+l_i_i_shape[2:])
#     idx = tf.stack((l_i_i,y,x,c),axis=-1)
#     l = tf.reshape(l, [b * img_ct] + l_shape[2:])
#     gathered = tf.gather_nd(l, idx,batch_dims=1)
#     gathered_reshape = tf.reshape(gathered,(b,img_ct)+gathered.shape[1:])
#     return gathered_reshape

@tf.function
def inner_slice(l_i_i, l_g_i, l):
    #l shape is b, I, L, h, w, c
    #l_i_i shape is b, I, h, w, c
    b, h, w, c_ct = l_i_i.shape
    x = tf.convert_to_tensor(np.arange(w,dtype=np.int32))
    y = tf.convert_to_tensor(np.arange(h,dtype=np.int32))
    c = tf.convert_to_tensor(np.arange(c_ct,dtype=np.int32))
    x, y, c = tnp.meshgrid(x, y, c)
    x = tf.repeat(x[None],axis=0,repeats=b)
    y = tf.repeat(y[None],axis=0,repeats=b)
    c = tf.repeat(c[None],axis=0,repeats=b)
    idx = tf.stack((l_i_i,l_g_i,y,x,c),axis=-1)
    gathered = tf.gather_nd(l, idx,batch_dims=1)
    return  gathered

@tf.function
def set_inner_slice(l_i_i, l_g_i, l):
    #l shape is b, I, L, h, w, c
    #l_i_i shape is b, I, h, w, c
    b, h, w, c_ct = l_i_i.shape
    x = tf.convert_to_tensor(np.arange(w,dtype=np.int32))
    y = tf.convert_to_tensor(np.arange(h,dtype=np.int32))
    c = tf.convert_to_tensor(np.arange(c_ct,dtype=np.int32))
    x, y, c = tnp.meshgrid(x, y, c)
    x = tf.repeat(x[None],axis=0,repeats=b)
    y = tf.repeat(y[None],axis=0,repeats=b)
    c = tf.repeat(c[None],axis=0,repeats=b)
    idx = tf.stack((l_i_i,l_g_i,y,x,c),axis=-1)
    zero_l = tf.zeros(l.shape)
    gathered_values = tf.gather_nd(zero_l, idx, batch_dims=1)
    update = tf.ones_like(gathered_values)
    zero_l_update = []
    for i in range(b):
        zero_l_update.append(tf.scatter_nd(idx[i], update[i], zero_l[i].shape))
    zero_l_update = tf.stack(zero_l_update,axis=0)
    return zero_l_update

@tf.function
def inner_slice_1d(l_i_i, l, IMSZ=448):
    #l shape is b, I, L, h, w, c
    #l_i_i shape is b, I, h, w, c
    
    b, img_ct, h, w, c_ct = l_i_i.shape
    l_i_i_shape = l_i_i.shape
    l_shape = l.shape
    x = tf.convert_to_tensor(np.arange(w,dtype=np.int32))
    y = tf.convert_to_tensor(np.arange(h,dtype=np.int32))
    c = tf.convert_to_tensor(np.arange(c_ct,dtype=np.int32))
    x, y, c = tnp.meshgrid(x, y, c)
    x = tf.repeat(x[None],axis=0,repeats=img_ct*b)
    y = tf.repeat(y[None],axis=0,repeats=img_ct*b)
    c = tf.repeat(c[None],axis=0,repeats=img_ct*b)
    l_i_i = tf.reshape(l_i_i,[b*img_ct]+l_i_i_shape[2:])
    idx = tf.stack((l_i_i,y,x,c),axis=-1)
    l = tf.reshape(l, [b * img_ct] + l_shape[2:])
    gathered = tf.gather_nd(l, idx,batch_dims=1)
    gathered_reshape = tf.reshape(gathered,(b,img_ct)+gathered.shape[1:])
    return gathered_reshape

@tf.function
def set_inner_slice_1d(l_i_i, l, IMSZ=448, ones=False):
    #l shape is b, I, L, h, w, c
    #l_i_i shape is b, I, h, w, c
    
    b, img_ct, h, w, c_ct = l_i_i.shape
    l_i_i_shape = l_i_i.shape
    l_shape = l.shape
    x = tf.convert_to_tensor(np.arange(w,dtype=np.int32))
    y = tf.convert_to_tensor(np.arange(h,dtype=np.int32))
    c = tf.convert_to_tensor(np.arange(c_ct,dtype=np.int32))
    x, y, c = tnp.meshgrid(x, y, c)
    x = tf.repeat(x[None],axis=0,repeats=img_ct*b)
    y = tf.repeat(y[None],axis=0,repeats=img_ct*b)
    c = tf.repeat(c[None],axis=0,repeats=img_ct*b)
    l_i_i = tf.reshape(l_i_i,[b*img_ct]+l_i_i_shape[2:])
    idx = tf.stack((l_i_i,y,x,c),axis=-1)
    l = tf.reshape(l, [b * img_ct] + l_shape[2:])
    # gathered = tf.gather_nd(l, idx,batch_dims=1)
    # gathered_reshape = tf.reshape(gathered,(b,img_ct)+gathered.shape[1:])

    zero_l = tf.zeros(l.shape)
    gathered_values = tf.gather_nd(zero_l, idx, batch_dims=1)
    update = tf.ones_like(gathered_values)
    zero_l_update = []
    for i in range(img_ct):
        zero_l_update.append(tf.scatter_nd(idx[i], update[i], zero_l[i].shape))
    zero_l_update = tf.stack(zero_l_update,axis=0)
    zero_l_update = tf.reshape(zero_l_update, l_shape)
    return zero_l_update
    
@tf.custom_gradient
def diffable_slice_2d(l_i, l_g, lpyramid, max_discrete_levels,IMSZ):
    l_r,l_c, _, _, _, _ = lpyramid.shape
    #fetch i and g pixels and discretize
    max_discrete_levels_ft = tf.cast(max_discrete_levels,tf.float32)
    l_i_i = tf.clip_by_value(tf.cast(l_i * (max_discrete_levels_ft - 1),tf.int32),0, max_discrete_levels-2)
    l_i_f = l_i * (max_discrete_levels_ft - 1) - tf.cast(l_i_i,dtype=tf.float32)
    l_g_i = tf.clip_by_value(tf.cast(l_g * (max_discrete_levels_ft - 1),tf.int32),0, max_discrete_levels-2)
    l_g_f = l_g * (max_discrete_levels_ft - 1) - tf.cast(l_g_i,dtype=tf.float32)

    l_i_i_0_l_g_i_0 = inner_slice(l_i_i,     l_g_i,     lpyramid)
    l_i_i_1_l_g_i_0 = inner_slice(l_i_i + 1, l_g_i,     lpyramid)
    l_i_i_0_l_g_i_1 = inner_slice(l_i_i,     l_g_i + 1, lpyramid)
    l_i_i_1_l_g_i_1 = inner_slice(l_i_i + 1, l_g_i + 1, lpyramid)

    # make laplacian pyramid by interpolation
    outLPyramid_i = (1 - l_i_f) * (1 - l_g_f) * l_i_i_0_l_g_i_0        + \
                    (    l_i_f) * (1 - l_g_f) * l_i_i_1_l_g_i_0        + \
                    (1 - l_i_f) * (    l_g_f) * l_i_i_0_l_g_i_1        + \
                    (    l_i_f) * (    l_g_f) * l_i_i_1_l_g_i_1
    # Define the custom gradient
    def grad_fn(dy):

        # make laplacian pyramid by interpolation
        d_i =   (  - 1) * (1 - l_g_f) * l_i_i_0_l_g_i_0        + \
                (    1) * (1 - l_g_f) * l_i_i_1_l_g_i_0        + \
                (  - 1) * (    l_g_f) * l_i_i_0_l_g_i_1        + \
                (    1) * (    l_g_f) * l_i_i_1_l_g_i_1

        d_g =   (1  - l_i_f) * ( - 1) * l_i_i_0_l_g_i_0        + \
                (     l_i_f) * ( - 1) * l_i_i_1_l_g_i_0        + \
                (1  - l_i_f) * (   1) * l_i_i_0_l_g_i_1        + \
                (     l_i_f) * (   1) * l_i_i_1_l_g_i_1
        
        l_i_i_0_l_g_i_0_ones = (1  - l_i_f) * (1 - l_g_f) * set_inner_slice(l_i_i,     l_g_i,     lpyramid)
        l_i_i_1_l_g_i_0_ones = (     l_i_f) * (1 - l_g_f) * set_inner_slice(l_i_i + 1, l_g_i,     lpyramid)
        l_i_i_0_l_g_i_1_ones = (1  - l_i_f) * (    l_g_f) * set_inner_slice(l_i_i,     l_g_i + 1, lpyramid)
        l_i_i_1_l_g_i_1_ones = (     l_i_f) * (    l_g_f) * set_inner_slice(l_i_i + 1, l_g_i + 1, lpyramid)
        dy_d_l = l_i_i_0_l_g_i_0_ones + l_i_i_1_l_g_i_0_ones + l_i_i_0_l_g_i_1_ones + l_i_i_1_l_g_i_1_ones
        

        return dy * d_i, dy * d_g, dy[None,None,...] * dy_d_l, None, None


    return outLPyramid_i, grad_fn

@tf.custom_gradient
def diffable_slice_separable(l_i, lpyramids, max_discrete_levels,IMSZ, min_intensity=0, max_intensity=1):
    #fetch i and g pixels and discretize
    # assert min_intensity < max_intensity
    l_i = (l_i - min_intensity) / (max_intensity - min_intensity)
    max_discrete_levels_ft = tf.cast(max_discrete_levels,tf.float32)
    l_i_is = tf.clip_by_value(tf.cast(l_i * (max_discrete_levels_ft - 1), tf.int32), 0, max_discrete_levels-2)
    l_i_fs = l_i * (max_discrete_levels_ft - 1) - tf.cast(l_i_is, dtype=tf.float32)
    
    l_i_i_0_l_g_i_0s = inner_slice_1d(l_i_is,     lpyramids, IMSZ=IMSZ)
    l_i_i_1_l_g_i_0s = inner_slice_1d(l_i_is + 1, lpyramids, IMSZ=IMSZ)
    
    # make laplacian pyramid by interpolation
    outLPyramids = (1 - l_i_fs) * l_i_i_0_l_g_i_0s + l_i_fs * l_i_i_1_l_g_i_0s
    outLPyramids = tf.reduce_sum(outLPyramids,axis=1)
    
    # Define the custom gradient
    def grad_fn(dy):
        # make laplacian pyramid by interpolation
        d_i = (  - 1) * l_i_i_0_l_g_i_0s + (    1) * l_i_i_1_l_g_i_0s
        l_i_i_0_l_g_i_0_ones = (1  - l_i_fs[:,:,None,...]) * set_inner_slice_1d(l_i_is,       lpyramids, IMSZ=IMSZ, ones=True)
        l_i_i_1_l_g_i_0_ones = (     l_i_fs[:,:,None,...]) * set_inner_slice_1d(l_i_is+1,     lpyramids, IMSZ=IMSZ, ones=True)
        dy_d_l = l_i_i_0_l_g_i_0_ones + l_i_i_1_l_g_i_0_ones
        return dy[None,...] * d_i, dy[None,None,...] * dy_d_l, None, None, None, None

    return outLPyramids, grad_fn

def deriv_slice(l_i, l_g, i, lpyramid, max_discrete_levels,IMSZ):
    #fetch i and g pixels and discretize
    l_i_i = tf.clip_by_value(tf.cast(l_i,dtype=tf.int32),0, max_discrete_levels-2)
    l_i_f = l_i - tf.cast(l_i_i,dtype=tf.float32)
    l_g_i = tf.clip_by_value(tf.cast(l_g,dtype=tf.int32), 0, max_discrete_levels-2)
    l_g_f = l_g - tf.cast(l_g_i,dtype=tf.float32)

    l_i_i_0_l_g_i_0 = inner_slice(l_i_i,     l_g_i,     lpyramid[i],IMSZ=IMSZ)
    l_i_i_1_l_g_i_0 = inner_slice(l_i_i + 1, l_g_i,     lpyramid[i],IMSZ=IMSZ)
    l_i_i_0_l_g_i_1 = inner_slice(l_i_i,     l_g_i + 1, lpyramid[i],IMSZ=IMSZ)
    l_i_i_1_l_g_i_1 = inner_slice(l_i_i + 1, l_g_i + 1, lpyramid[i],IMSZ=IMSZ)

    # make laplacian pyramid by interpolation
    d_i =   (  - 1) * (1 - l_g_f) * l_i_i_0_l_g_i_0        + \
            (    1) * (1 - l_g_f) * l_i_i_1_l_g_i_0        + \
            (  - 1) * (    l_g_f) * l_i_i_0_l_g_i_1        + \
            (    1) * (    l_g_f) * l_i_i_1_l_g_i_1

    d_g =   (1  - l_i_f) * ( - 1) * l_i_i_0_l_g_i_0        + \
            (     l_i_f) * ( - 1) * l_i_i_1_l_g_i_0        + \
            (1  - l_i_f) * (   1) * l_i_i_0_l_g_i_1        + \
            (     l_i_f) * (   1) * l_i_i_1_l_g_i_1
    return d_i, d_g

# dense function
def dense_fn(image):
    return tf.reduce_mean(image) * image

def finite_derivative(fn, input_image, eps=0.01):
    """
    Takes an function fn with input h,w,c and output h,w,c and computes the finite difference of it
    a jacobian tensor with shape h,w,c,h,w,c 
    it returns a 6d jacobian, first 3 dimensions correspond to the output and last to the input
    """
    
    h, w, c = input_image.shape
    fn_i = fn(input_image)
    tensor_list = []
    for ii in tqdm.trange(h):
        for ij in range(w):
            for ik in range(c):
                perturbed_image = tf.identity(input_image)
                perturbed_image = tf.tensor_scatter_nd_add(
                            perturbed_image,
                            indices=[[ii, ij, ik]],
                            updates=[-eps]
                        )
                tensor_list.append((fn_i - fn(perturbed_image)) / eps)

    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0) #ihiwic,oh,ow,oc 
    autodiff_per_pixel_gradients = tf.reshape(autodiff_per_pixel_gradients,(h,w,c,h,w,c))
    return tf.transpose(autodiff_per_pixel_gradients,(3,4,5,0,1,2))#oh,ow,oc,ih,iw,ic

def finite_derivative_l(fn, input_image, eps=0.01):
    """
    Takes an function fn with input h,w,c and output h,w,c and computes the finite difference of it
    a jacobian tensor with shape h,w,c,h,w,c 
    it returns a 6d jacobian, first 3 dimensions correspond to the output and last to the input
    """
    
    ro, co, _, h, w, c = input_image.shape
    fn_i = fn(input_image)
    tensor_list = []
    print('finite diff')
    for ir in tqdm.trange(ro):
        for ic in range(co):
            for ii in range(h):
                for ij in range(w):
                    for ik in range(c):
                        perturbed_image = tf.identity(input_image)
                        perturbed_image = tf.tensor_scatter_nd_add(
                                    perturbed_image,
                                    indices=[[ir,ic,0,ii, ij, ik]],
                                    updates=[-eps]
                                )
                        tensor_list.append((fn_i - fn(perturbed_image)) / eps)

    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0)#ihxiwxicxroxco,oh,ow,oc 
    autodiff_per_pixel_gradients = tf.reshape(autodiff_per_pixel_gradients,(ro,co,h,w,c,h,w,c))
    return tf.transpose(autodiff_per_pixel_gradients,(5,6,7,0,1,2,3,4))#oh,ow,oc,ro,co,ih,iw,ic

def gradient_6d(fn, input_image):
    """takes a function fn that takes an input image and returns an image
    it returns a 6d jacobian, first 3 dimensions correspond to the output and last to the input
    Args:
        f (_type_): _description_
        var (_type_): _description_
    """
    h, w, c = input_image.shape
    tensor_list = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_image)
        output_image = fn(input_image)
    
        # Compute gradients per output pixel
        for i in tqdm.trange(h):
            for j in range(w):
                for k in range(c):
                    pixel_value = output_image[i, j, k]
                    pixel_gradient = tape.gradient(pixel_value, input_image)
                    tensor_list.append(pixel_gradient)
    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0)#ohxowxocx,ro,co,ih,iw,ic 
    return tf.reshape(autodiff_per_pixel_gradients,(h,w,c,h,w,c))
    
def gradient_l(fn, input_image):
    """takes a function fn that takes an input image and returns an image
    it returns a 6d jacobian, first 3 dimensions correspond to the output and last to the input
    Args:
        f (_type_): _description_
        var (_type_): _description_
    """
    ro,co,_,h, w, c = input_image.shape
    tensor_list = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_image)
        output_image = fn(input_image)
        print('autodiff')
        # Compute gradients per output pixel

        for i in range(h):
            for j in range(w):
                for k in range(c):
                    pixel_value = output_image[i, j, k]
                    pixel_gradient = tape.gradient(pixel_value, input_image)
                    tensor_list.append(pixel_gradient)
    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0)
    return tf.reshape(autodiff_per_pixel_gradients,(h, w, c, ro, co, h, w, c))#oh,ow,oc,ro,co,ih,iw,ic 

def visualize_grad_6d(fn_result, gradients, finite_differences, exponent=1, scale=1):
    gradients_mean = tf.reduce_sum(gradients,axis=(0,1,2)) ** exponent * scale
    finite_differences_mean = tf.reduce_sum(finite_differences,axis=(0,1,2)) ** exponent * scale
    g = {'fn_result':fn_result}
    g.update({'gradients_mean':tf.clip_by_value(gradients_mean,0,1), 'finite_differences_mean':tf.clip_by_value(finite_differences_mean,0,1)})
    g.update({'gradients_5,8,0':tf.clip_by_value(gradients[1,2,0,...],0,1), 'finite_differences_5,8,0':tf.clip_by_value(finite_differences[1,2,0,...],0,1)})
    
    lbl = {'fn_result':'fn_result'}
    lbl.update({'gradients_mean':'gradients_mean', 'finite_differences_mean':'finite_differences_mean'})
    lbl.update({'gradients_5,8,0':'gradients_5,8,0', 'finite_differences_5,8,0':'finite_differences_5,8,0'})

    for k,v in g.items():
        g[k] = cv2.resize(v.numpy(),(448,448))
    logr.addImage(g, lbl, 'train')
    logr.takeStep()

def visualize_grad_7d(fn_result, gradients, finite_differences, exponent=1, scale=1):
    gradients_mean = tf.reduce_mean(gradients,axis=(0,1,2,3)) ** exponent * scale
    finite_differences_mean = tf.reduce_mean(finite_differences,axis=(0,1,2,3)) ** exponent * scale
    g = {'fn_result':fn_result}
    g.update({'gradients_mean':tf.clip_by_value(gradients_mean,0,1), 'finite_differences_mean':tf.clip_by_value(finite_differences_mean,0,1)})
    g.update({'gradients_5,8,0':tf.clip_by_value(gradients[1,2,0,0,...],0,1), 'finite_differences_5,8,0':tf.clip_by_value(finite_differences[1,2,0,0,...],0,1)})
    
    lbl = {'fn_result':'fn_result'}
    lbl.update({'gradients_mean':'gradients_mean', 'finite_differences_mean':'finite_differences_mean'})
    lbl.update({'gradients_5,8,0':'gradients_5,8,0', 'finite_differences_5,8,0':'finite_differences_5,8,0'})

    for k,v in g.items():
        g[k] = cv2.resize(v.numpy(),(448,448))
    logr.addImage(g, lbl, 'train')
    logr.takeStep()

def compare_grad_gllf():
    input_fn = '/home/mohammad/Downloads/fft_combine/blurred.png'
    guide_fn = '/home/mohammad/Downloads/fft_combine/flash.png'

    max_levels = 3
    max_discrete_levels = 3
    alpha = 1
    beta  = 1.0
    sigma = 1.0
    IMSZ = 448
    im_i = imageio.imread(input_fn).astype(np.float32) / 255.0
    im_g = imageio.imread(guide_fn).astype(np.float32) / 255.0
    # im_i = tf.convert_to_tensor(im_i[None,128:128+IMSZ,128:128+IMSZ,:])
    # im_g = tf.convert_to_tensor(im_g[None,128:128+IMSZ,128:128+IMSZ,:])
    im_i = cv2.resize(im_i, (IMSZ, IMSZ))
    im_g = cv2.resize(im_g, (IMSZ, IMSZ))
    im_i = tf.convert_to_tensor(im_i[None,...])
    im_g = tf.convert_to_tensor(im_g[None,...])
    alpha_h = 0.5#tf.random.uniform(im_i.shape,0,1)
    alpha_i = 0#tf.random.uniform(im_i.shape,0,1)

    fn = lambda diffable_image: gllf_diffable_2d(diffable_image[None,...], im_g, max_levels, max_discrete_levels, alpha_h, alpha_i, IMSZ=IMSZ, beta=beta, sigma=sigma)[0,...]
    fn_result = tf.clip_by_value(fn(im_i[0]),0,1)
    g = {'fn_result':fn_result}
    lbl = {'fn_result':'fn_result'}

    for k,v in g.items():
        g[k] = cv2.resize(v.numpy(),(448,448))
    logr.addImage(g, lbl, 'train')
    logr.addIndividualImages(g, lbl, 'title', format='exr')
    logr.takeStep()

    def test_grad(fn, img):
        fn_result = fn(img)
        gradients = tf.abs(gradient_6d(fn, img))
        finite_differences = tf.abs(finite_derivative(fn, img, eps=0.001))
        return fn_result, gradients, finite_differences

    
    fn_result, gradients, finite_differences = test_grad(fn, im_i[0])
            
    visualize_grad_6d(fn_result, gradients, finite_differences, 1, 1)
    


def compare_grad_slice():
    input_fn = '/home/mohammad/Downloads/fft_combine/blurred.png'
    guide_fn = '/home/mohammad/Downloads/fft_combine/flash.png'

    max_levels = 4
    max_discrete_levels = 4
    alpha = 1
    beta  = 1.0
    sigma = 1.0
    IMSZ = 32
    im_i = imageio.imread(input_fn).astype(np.float32) / 255.0
    im_g = imageio.imread(guide_fn).astype(np.float32) / 255.0
    im_i = tf.convert_to_tensor(im_i[None,128:128+IMSZ,128:128+IMSZ,:])
    im_g = tf.convert_to_tensor(im_g[None,128:128+IMSZ,128:128+IMSZ,:])
    alpha_h = tf.random.uniform(im_i.shape,0,1)
    alpha_i = tf.random.uniform(im_i.shape,0,1)
    
    fn = lambda diffable_image: gllf_diffable_2d(diffable_image[None,...], im_g, max_levels, max_discrete_levels, alpha_h, alpha_i, IMSZ=IMSZ, beta=beta, sigma=sigma)[0,...]


    slice_inp_i, slice_inp_g, slice_inp_l = images_to_lookup_2d(im_i, im_g,im_i, im_g, max_levels, max_discrete_levels, alpha_h, alpha_i, IMSZ=IMSZ, beta=beta, sigma=sigma)
    # slice(slice_inp_i[1], slice_inp_g[1], slice_inp_p[1])
    # fn = lambda alpha_h: images_to_lookup_2d(im_i, im_g,im_i, im_g, max_levels, max_discrete_levels, alpha_h[None,...], alpha_i, IMSZ=IMSZ, beta=beta, sigma=sigma)[0,...]
    # fn = dense_fn
    i = 0
    fn_source = lambda diffable_img:diffable_slice_2d(diffable_img[None,...], slice_inp_g[i], slice_inp_l[i], max_discrete_levels,IMSZ)[0,...]
    fn_guide = lambda diffable_img:diffable_slice_2d(slice_inp_i[i], diffable_img[None,...], slice_inp_l[i], max_discrete_levels,IMSZ)[0,...]
    fn_intensity = lambda diffable_img:diffable_slice_2d(slice_inp_i[i], slice_inp_g[i], diffable_img, max_discrete_levels,IMSZ)[0,...]
        
    def test_grad_l(fn,img,i0, j0):
        fn_result = fn(img) * 5
        gradients = tf.abs(gradient_l(fn, img))
        
        filename = './fd_l.npy'
        import os
        if(not os.path.exists(filename)):
            finite_differences = tf.abs(finite_derivative_l(fn, img, eps=0.0001))
            np.save(filename, finite_differences)
        else:
            finite_differences = np.load(filename)
        return fn_result, gradients, finite_differences
        

    # fn_result, gradients, finite_differences = test_grad(fn_source, slice_inp_i[i][0])
    # visualize_grad(fn_result, gradients, finite_differences)
    # fn_result, gradients, finite_differences = test_grad(fn_guide, slice_inp_g[i][0])
    # visualize_grad(fn_result, gradients, finite_differences)
    fn_result, gradients, finite_differences = test_grad_l(fn_intensity, slice_inp_l[i],0,0)
            
    # pyramid = cv2.resize(slice_inp_l[i][0,0,0,...].numpy(),(448,448))
    # logr.addImage({'pyramid':pyramid}, {'pyramid':'pyramid'}, 'train')
    # logr.takeStep()
    finite_differences_tmp = tf.reduce_mean(tf.reduce_mean(finite_differences,axis=3),axis=3)
    gradients_tmp = tf.reduce_mean(tf.reduce_mean(gradients,axis=3),axis=3)
    # gradients_tmp = tf.transpose(gradients_tmp,(3,4,5,0,1,2))
    visualize_grad_6d(fn_result, gradients_tmp, finite_differences_tmp)
    

def finite_derivative_7d(fn, input_image, input_i, level_i, eps=0.01):
    """
    Takes an function fn with input h,w,c and output h,w,c and computes the finite difference of it
    a jacobian tensor with shape h,w,c,h,w,c 
    it returns a 6d jacobian, first 3 dimensions correspond to the output and last to the input
    """
    
    _, _, ro, _, h, w, c = input_image.shape
    fn_i = fn(input_image)
    tensor_list = []
    print('finite diff')
    for ir in tqdm.trange(ro):
        for ii in range(h):
            for ij in range(w):
                for ik in range(c):
                    perturbed_image = tf.identity(input_image)
                    perturbed_image = tf.tensor_scatter_nd_add(
                                perturbed_image,
                                indices=[[input_i, level_i, ir,0,ii, ij, ik]],
                                updates=[-eps]
                            )
                    tensor_list.append((fn_i - fn(perturbed_image)) / eps)

    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0)#ihxiwxicxro,oh,ow,oc 
    autodiff_per_pixel_gradients = tf.reshape(autodiff_per_pixel_gradients,(ro,h,w,c,h,w,c))
    return tf.transpose(autodiff_per_pixel_gradients,(4,5,6,0,1,2,3))#oh,ow,oc,ro,ih,iw,ic

def gradient_7d(fn, input_image, input_i, level_i):
    """takes a function fn that takes an input image and returns an image
    it returns a 6d jacobian, first 3 dimensions correspond to the output and last to the input
    Args:
        f (_type_): _description_
        var (_type_): _description_
    """
    _, _, ro, _, h, w, c = input_image.shape
    # ro,_,h, w, c = input_image.shape
    tensor_list = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_image)
        output_image = fn(input_image)
        print('autodiff')
        # Compute gradients per output pixel

        for i in tqdm.trange(h):
            for j in range(w):
                for k in range(c):
                    pixel_value = output_image[0, i, j, k]
                    pixel_gradient = tape.gradient(pixel_value, input_image)
                    tensor_list.append(pixel_gradient)
    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0)
    return tf.reshape(autodiff_per_pixel_gradients[:,input_i, level_i,:,0,...],(h, w, c, ro, h, w, c))#oh,ow,oc,ro,ih,iw,ic 


def compare_grad_slice_1d():
    input_fn = '/home/mohammad/Downloads/fft_combine/blurred.png'
    guide_fn = '/home/mohammad/Downloads/fft_combine/flash.png'

    max_levels = 4
    max_discrete_levels = 4
    alpha = 1
    beta  = 1.0
    sigma = 1.0
    IMSZ = 32
    im_i = imageio.imread(input_fn).astype(np.float32) / 255.0
    im_g = imageio.imread(guide_fn).astype(np.float32) / 255.0
    im_i = tf.convert_to_tensor(im_i[None,128:128+IMSZ,128:128+IMSZ,:])
    im_g = tf.convert_to_tensor(im_g[None,128:128+IMSZ,128:128+IMSZ,:])
    # im_i = cv2.resize(im_i, (IMSZ, IMSZ))
    # im_g = cv2.resize(im_g, (IMSZ, IMSZ))
    # im_i = tf.convert_to_tensor(im_i[None,...])
    # im_g = tf.convert_to_tensor(im_g[None,...])

    alpha_h = 1#tf.random.uniform(im_i.shape,0,1)
    alpha_i = 0#tf.random.uniform(im_i.shape,0,1)
    

    #show forward gllf
    fn_result = gllf_diffable_1d([im_i, im_g], [im_i, im_g], [alpha_i, alpha_h], max_levels, max_discrete_levels, betas=[beta,0], sigmas=[sigma,0], IMSZ=IMSZ)[0,...]
    g = {'fn_result':fn_result}
    lbl = {'fn_result':'fn_result'}

    for k,v in g.items():
        g[k] = cv2.resize(v.numpy(),(448,448))
    logr.addImage(g, lbl, 'train')
    logr.takeStep()

    #gradient of slice
    outLPyramids = images_to_lookup_1d([im_i, im_g], max_levels, max_discrete_levels, [alpha_i, alpha_h], IMSZ=IMSZ, betas=[beta,0], sigmas=[sigma,0])
    G_is = []
    for im_i_pyramid in [im_i, im_g]:
        G_i = GaussianPyramid(im_i_pyramid, max_levels)
        G_is.append(tf.stack([_resize(i * (max_discrete_levels - 1),(IMSZ, IMSZ)) for i in G_i]))
    G_is = tf.stack(G_is,axis=0) #n, L, 1, h, w, c
    i = 1
    #outLPyramids n, L, L, 1, h, w, c
    #G_is         n, L, 1, h, w, c
    fn = lambda diffable_img:diffable_slice_separable(G_is[:,i,...], diffable_img[:,i,...], max_discrete_levels, IMSZ)
    input_i, level_i = 0, i
    fn_grad = 'grad.npy'
    fn_fd = 'fd.npy'
    if(os.path.exists(fn_grad)):
        grad_diff = np.load(fn_grad)
    else:
        grad_diff = gradient_7d(fn, outLPyramids,input_i, level_i)
        np.save(fn_grad, grad_diff)

    if(os.path.exists(fn_fd)):
        finite_diff = np.load(fn_fd)
    else:
        finite_diff = finite_derivative_7d(fn, outLPyramids,input_i, level_i, eps=0.0001)
        np.save(fn_fd, finite_diff)
    visualize_grad_7d(fn_result, tf.abs(grad_diff)*1000, tf.abs(finite_diff)*1000)

    # outLPyramid_slice = diffable_slice_separable(G_is[:,i,...], outLPyramids[:,i,...], max_discrete_levels,IMSZ)
    # fn_intensity = lambda diffable_img:diffable_slice_separable(l_i_is[:,i,...], l_i_fs[:,i,...], diffable_img, max_discrete_levels,IMSZ)[0,...]

    #6d gradient
    fn_intensity = lambda diffable_img:gllf_diffable_1d([im_i, im_g], [im_i, diffable_img[None,...]], [alpha_i, alpha_h], max_levels, max_discrete_levels, betas=[beta,0], sigmas=[sigma,0], IMSZ=IMSZ)[0,...]
    
    def test_grad_l(fn,img):
        fn_result = fn(img) * 5
        gradients = tf.abs(gradient_6d(fn, img))
        
        filename = './fd_l.npy'
        import os
        if(not os.path.exists(filename)):
            finite_differences = tf.abs(finite_derivative(fn, img, eps=0.0001))
            np.save(filename, finite_differences)
        else:
            finite_differences = np.load(filename)
        return fn_result, gradients, finite_differences
        
    fn_result, gradients, finite_differences = test_grad_l(fn_intensity, im_g[0])
            
    # finite_differences_tmp = tf.reduce_mean(tf.reduce_mean(finite_differences,axis=3),axis=3)
    # gradients_tmp = tf.reduce_mean(tf.reduce_mean(gradients,axis=3),axis=3)
    # # gradients_tmp = tf.transpose(gradients_tmp,(3,4,5,0,1,2))
    visualize_grad_6d(fn_result, tf.abs(gradients), tf.abs(finite_differences))
    # fn_result = tf.abs(fn_intensity(slice_inp_ls[:,i,...])) * 5
    print('hi')

    
def test_compare_grad_slice_1d():
    input_fn = '/home/mohammad/Downloads/fft_combine/blurred.png'
    guide_fn = '/home/mohammad/Downloads/fft_combine/flash.png'

    max_levels = 4
    max_discrete_levels = 4
    alpha = 1
    beta  = 1.0
    sigma = 1.0
    IMSZ = 32
    im_i = imageio.imread(input_fn).astype(np.float32) / 255.0
    im_g = imageio.imread(guide_fn).astype(np.float32) / 255.0
    im_i = tf.convert_to_tensor(im_i[None,128:128+IMSZ,128:128+IMSZ,:])
    im_g = tf.convert_to_tensor(im_g[None,128:128+IMSZ,128:128+IMSZ,:])
    # im_i = cv2.resize(im_i, (IMSZ, IMSZ))
    # im_g = cv2.resize(im_g, (IMSZ, IMSZ))
    # im_i = tf.convert_to_tensor(im_i[None,...])
    # im_g = tf.convert_to_tensor(im_g[None,...])

    alpha_h = 1#tf.random.uniform(im_i.shape,0,1)
    alpha_i = 0#tf.random.uniform(im_i.shape,0,1)
    

    #show forward gllf
    fn_result = gllf_diffable_1d([im_i, im_g], [im_i, im_g], [alpha_i, alpha_h], max_levels, max_discrete_levels, betas=[beta,0], sigmas=[sigma,0], IMSZ=IMSZ)[0,...]
    g = {'fn_result':fn_result}
    lbl = {'fn_result':'fn_result'}

    for k,v in g.items():
        g[k] = cv2.resize(v.numpy(),(448,448))
    logr.addImage(g, lbl, 'train')
    logr.takeStep()

    #gradient of slice
    outLPyramids = images_to_lookup_1d([im_i, im_g], max_levels, max_discrete_levels, [alpha_i, alpha_h], IMSZ=IMSZ, betas=[beta,0], sigmas=[sigma,0])
    G_is = []
    for im_i_pyramid in [im_i, im_g]:
        G_i = GaussianPyramid(im_i_pyramid, max_levels)
        G_is.append(tf.stack([_resize(i * (max_discrete_levels - 1),(IMSZ, IMSZ)) for i in G_i]))
    G_is = tf.stack(G_is,axis=0) #n, L, 1, h, w, c
    i = 1
    #outLPyramids n, L, L, 1, h, w, c
    #G_is         n, L, 1, h, w, c
    fn = lambda diffable_img:diffable_slice_separable(G_is[:,i,...], diffable_img[:,i,...], max_discrete_levels, IMSZ)
    input_i, level_i = 0, i
    fn_grad = 'grad.npy'
    fn_fd = 'fd.npy'
    if(os.path.exists(fn_grad)):
        grad_diff = np.load(fn_grad)
    else:
        grad_diff = gradient_7d(fn, outLPyramids,input_i, level_i)
        np.save(fn_grad, grad_diff)

    if(os.path.exists(fn_fd)):
        finite_diff = np.load(fn_fd)
    else:
        finite_diff = finite_derivative_7d(fn, outLPyramids,input_i, level_i, eps=0.0001)
        np.save(fn_fd, finite_diff)
    visualize_grad_7d(fn_result, tf.abs(grad_diff)*1000, tf.abs(finite_diff)*1000)

    # outLPyramid_slice = diffable_slice_separable(G_is[:,i,...], outLPyramids[:,i,...], max_discrete_levels,IMSZ)
    # fn_intensity = lambda diffable_img:diffable_slice_separable(l_i_is[:,i,...], l_i_fs[:,i,...], diffable_img, max_discrete_levels,IMSZ)[0,...]

    #6d gradient
    fn_intensity = lambda diffable_img:gllf_diffable_1d([im_i, im_g], [im_i, diffable_img[None,...]], [alpha_i, alpha_h], max_levels, max_discrete_levels, betas=[beta,0], sigmas=[sigma,0], IMSZ=IMSZ)[0,...]
    
    def test_grad_l(fn,img):
        fn_result = fn(img) * 5
        gradients = tf.abs(gradient_6d(fn, img))
        
        filename = './fd_l.npy'
        import os
        if(not os.path.exists(filename)):
            finite_differences = tf.abs(finite_derivative(fn, img, eps=0.0001))
            np.save(filename, finite_differences)
        else:
            finite_differences = np.load(filename)
        return fn_result, gradients, finite_differences
        
    fn_result, gradients, finite_differences = test_grad_l(fn_intensity, im_g[0])
            
    # finite_differences_tmp = tf.reduce_mean(tf.reduce_mean(finite_differences,axis=3),axis=3)
    # gradients_tmp = tf.reduce_mean(tf.reduce_mean(gradients,axis=3),axis=3)
    # # gradients_tmp = tf.transpose(gradients_tmp,(3,4,5,0,1,2))
    visualize_grad_6d(fn_result, tf.abs(gradients), tf.abs(finite_differences))
    # fn_result = tf.abs(fn_intensity(slice_inp_ls[:,i,...])) * 5
    print('hi')


def run_gllf():
    #read guide and input
    input_fn = '/home/mohammad/Downloads/fft_combine/blurred.png'
    # input_fn = '/home/mohammad/Downloads/fft_combine/flash.png'
    guide_fn = '/home/mohammad/Downloads/fft_combine/flash.png'
    max_levels = 4
    max_discrete_levels = 4
    alpha = 1
    beta  = 1.0
    sigma = 1.0
    IMSZ = 448
    im_i = imageio.imread(input_fn).astype(np.float32) / 255.0
    im_g = imageio.imread(guide_fn).astype(np.float32) / 255.0
    im_i = cv2.resize(im_i, (IMSZ, IMSZ))[None,...]
    im_g = cv2.resize(im_g, (IMSZ, IMSZ))[None,...]

    

    timing_iterations = 2
    fn = lambda input, guide, alpha_h, alpha_i, beta_: gllf_diffable_2d(input, guide,input, guide, max_levels, max_discrete_levels, alpha_h, alpha_i, IMSZ=448, beta=beta_, sigma=sigma)
    alpha_i_map = alpha * np.ones_like(im_i)
    alpha_j_map = alpha * np.ones_like(im_i)
    im_i_var = tf.Variable(im_i)
    im_g_var = tf.Variable(im_g)
    alpha_i_var = tf.Variable(alpha_i_map)
    alpha_j_var = tf.Variable(alpha_j_map)
    beta_var = tf.Variable(beta * tf.ones_like(im_i))
    variables = {'im_i':im_i_var, 'im_g':im_g_var, 'alpha_i':alpha_i_var, 'alpha_j':alpha_j_var, 'beta':beta_var}
    g = fn(variables['im_i'], variables['im_g'], variables['alpha_i'], variables['alpha_j'], variables['beta'])
    start = (128, 128)
    length = (64, 64)

    d_alpha_i = finite_derivative(fn, variables, 'alpha_i',start,length)
    with tf.GradientTape() as tape:
        g = fn(variables['im_i'], variables['im_g'], variables['alpha_i'], variables['alpha_j'], variables['beta'])
        gradients = tape.gradient(g, variables.values())

    # eps = 0.001
    # g_fd = np.zeros_like(im_i)
    # import tqdm
    # for i in tqdm.trange(64):
    #     for j in tqdm.trange(64):
    #         alpha_p = alpha_map
    #         alpha_n = alpha_map
    #         alpha_p[0,i,j,:] += eps
    #         alpha_n[0,i,j,:] -= eps
    #         alpha_p = tf.convert_to_tensor(alpha_p)
    #         alpha_n = tf.convert_to_tensor(alpha_n)
    #         g_p = fn(variables['im_i'], variables['im_g'], alpha_p, variables['beta'])
    #         g_n = fn(variables['im_i'], variables['im_g'], alpha_n, variables['beta'])
    #         g_fd_ij = (g_p[0] - g_n[0]) / (eps * 2)
    #         g_fd[0] += g_fd_ij.numpy()
    timeit_fn = lambda : fn(im_i, im_g, alpha, beta)
    t = timeit.Timer(timeit_fn, setup=timeit_fn)
    avg_time_sec = t.timeit(number=timing_iterations) / timing_iterations
    print("time: %fms" % (avg_time_sec * 1e3))
    # g = fn()

    im = {'%i' % i:im for i, im in enumerate(g)}
    lbl = {'%i' % i:'%i' % i for i in range(len(g))}
    # im.update({'gradients_alpha':gradients[2], 'fd':g_fd})
    # lbl.update({'gradients_alpha':'$\\partial O\\partial \\alpha $','fd':'$(O(\\alpha) - O(\\alpha-0.001))/0.001 $'})
    logr.addImage(im, lbl, 'train')

# compare_grad_gllf()
# compare_grad_slice_1d()