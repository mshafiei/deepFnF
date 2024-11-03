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
# tf.config.run_functions_eagerly(True)

def _downsample(im):
    """Downsample with a 1 3 3 1 filter"""
    #im is padded with [1, 2], [1, 2] with constants
    assert len(im.shape) == 4
    h, w = im.shape[1], im.shape[2]
    x0,x1 = 1,w
    y0,y1 = 1,h
    im = tf.pad(im,[[0,0],[1,2],[1,2],[0,0]])
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

def remap(fx, alpha):
    return alpha * fx * tf.exp(-fx * fx / 2.0)

def remapping(i, idx_guide, k_i, k_g, n_levels, sigma, beta, alpha_h, alpha_i):
    level_input = k_i / (n_levels - 1)
    level_guide = k_g / (n_levels - 1)
    return sigma * level_input + beta * (i - level_input) + remap(idx_guide - level_guide, alpha_h) + remap(idx_guide - level_guide, alpha_i)

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
            r_i_j = remapping(im_i, idx_guide, k_i, k_g, max_discrete_levels, sigma, beta, alpha_h, alpha_i)
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

@tf.function
def gllf_diffable(im_i, im_g, max_levels, max_discrete_levels, alpha_h, alpha_i, beta=1, sigma=1, IMSZ=448):
    # im_i_pyramid = tf.stop_gradient(im_i_pyramid)
    # im_g_pyramid = tf.stop_gradient(im_g_pyramid)
    #input and guide pyramids
    G_i = GaussianPyramid(im_i, max_levels)
    G_g = GaussianPyramid(im_g, max_levels)
    
    idx_guide = im_g #* (max_discrete_levels - 1) #* 256.0
    # idx_guide = tfp.math.clip_by_value_preserve_gradient(idx_guide, 0, (max_discrete_levels - 1) * 256)
    # idx_guide = tf.cast(idx_guide, dtype=tf.float32)
    # compute remapped images and its pyramids
    lpyramid = []
    for k_i in range(max_discrete_levels):
        l_i = []
        for k_g in range(max_discrete_levels):
            r_i_j = remapping(im_i, idx_guide, k_i, k_g, max_discrete_levels, sigma, beta, alpha_h, alpha_i)
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
        outLPyramid_i = diffable_slice(G_i_resized[i], G_g_resized[i], lpyramid[i], max_discrete_levels,IMSZ)

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
def images_to_lookup(im_i, im_g, im_i_pyramid, im_g_pyramid, max_levels, max_discrete_levels, alpha_h, alpha_i, beta=1, sigma=1, IMSZ=448):
    # im_i_pyramid = tf.stop_gradient(im_i_pyramid)
    # im_g_pyramid = tf.stop_gradient(im_g_pyramid)
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
            r_i_j = remapping(im_i, idx_guide, k_i, k_g, max_discrete_levels, sigma, beta, alpha_h, alpha_i)
            #compute gaussian and laplacian of remapped images
            f_i_j_g = GaussianPyramid(r_i_j, max_levels)
            
            f_i_j_l = LaplacianPyramid(f_i_j_g)

            for i in range(len(f_i_j_l)):
                f_i_j_l[i] = _resize(f_i_j_l[i], im_i.shape[1:3])
                
                # f_i_j_l[i] = _resize(f_i_j_g[i], im_i.shape[1:3])
            
            l_g_images = tf.stack(f_i_j_l, axis=0)
            l_i.append(l_g_images) #l x 1 x h x w x 3
        
        lpyramid.append(tf.stack(l_i,axis=1))#l x k_g x 1 x h x w x 3
    lpyramid = tf.stack(lpyramid,axis=1) #l x k_i x k_g x 1 x h x w x 3
    
    outLPyramid = []
    G_i_resized = [_resize(i * (max_discrete_levels - 1),(IMSZ, IMSZ)) for i in G_i]
    G_g_resized = [_resize(i * (max_discrete_levels - 1),(IMSZ, IMSZ)) for i in G_g]
    slice_inp_i = []
    slice_inp_g = []
    slice_inp_p = []
    for i in range(max_levels):

        slice_inp_i.append(G_i_resized[i])
        slice_inp_g.append(G_g_resized[i])
        slice_inp_p.append(lpyramid[i])
    
    return slice_inp_i, slice_inp_g, slice_inp_p

@tf.function
def inner_slice(l_i_i, l_g_i, l, IMSZ=448):
    x = tf.convert_to_tensor(np.arange(448,dtype=np.int32))
    y = tf.convert_to_tensor(np.arange(448,dtype=np.int32))
    c = tf.convert_to_tensor(np.arange(3,dtype=np.int32))
    x, y, c = tnp.meshgrid(x, y, c)
    l_i_i = tf.squeeze(l_i_i)
    l_g_i = tf.squeeze(l_g_i)
    idx = tf.stack((l_i_i,l_g_i,y,x,c),axis=-1)[None, None]
    l = tf.squeeze(l)
    return tf.gather_nd(l, idx)
    
@tf.custom_gradient
def diffable_slice(l_i, l_g, lpyramid, max_discrete_levels,IMSZ):
    #fetch i and g pixels and discretize
    l_i_i = tf.clip_by_value(tf.cast(l_i,dtype=tf.int32),0, max_discrete_levels-2)
    l_i_f = l_i - tf.cast(l_i_i,dtype=tf.float32)
    l_g_i = tf.clip_by_value(tf.cast(l_g,dtype=tf.int32), 0, max_discrete_levels-2)
    l_g_f = l_g - tf.cast(l_g_i,dtype=tf.float32)

    l_i_i_0_l_g_i_0 = tf.squeeze(inner_slice(l_i_i,     l_g_i,     lpyramid,IMSZ=IMSZ))
    l_i_i_1_l_g_i_0 = tf.squeeze(inner_slice(l_i_i + 1, l_g_i,     lpyramid,IMSZ=IMSZ))
    l_i_i_0_l_g_i_1 = tf.squeeze(inner_slice(l_i_i,     l_g_i + 1, lpyramid,IMSZ=IMSZ))
    l_i_i_1_l_g_i_1 = tf.squeeze(inner_slice(l_i_i + 1, l_g_i + 1, lpyramid,IMSZ=IMSZ))

    # Define the custom gradient
    def grad_fn(dy):
        #fetch i and g pixels and discretize
        l_i_i = tf.clip_by_value(tf.cast(l_i,dtype=tf.int32),0, max_discrete_levels-2)
        l_i_f = l_i - tf.cast(l_i_i,dtype=tf.float32)
        l_g_i = tf.clip_by_value(tf.cast(l_g,dtype=tf.int32), 0, max_discrete_levels-2)
        l_g_f = l_g - tf.cast(l_g_i,dtype=tf.float32)

        # make laplacian pyramid by interpolation
        d_i =   (  - 1) * (1 - l_g_f) * l_i_i_0_l_g_i_0        + \
                (    1) * (1 - l_g_f) * l_i_i_1_l_g_i_0        + \
                (  - 1) * (    l_g_f) * l_i_i_0_l_g_i_1        + \
                (    1) * (    l_g_f) * l_i_i_1_l_g_i_1

        d_g =   (1  - l_i_f) * ( - 1) * l_i_i_0_l_g_i_0        + \
                (     l_i_f) * ( - 1) * l_i_i_1_l_g_i_0        + \
                (1  - l_i_f) * (   1) * l_i_i_0_l_g_i_1        + \
                (     l_i_f) * (   1) * l_i_i_1_l_g_i_1
                
        return dy * d_i, dy * d_g, None, None, None

    # make laplacian pyramid by interpolation
    outLPyramid_i = (1 - l_i_f) * (1 - l_g_f) * l_i_i_0_l_g_i_0        + \
                    (    l_i_f) * (1 - l_g_f) * l_i_i_1_l_g_i_0        + \
                    (1 - l_i_f) * (    l_g_f) * l_i_i_0_l_g_i_1        + \
                    (    l_i_f) * (    l_g_f) * l_i_i_1_l_g_i_1
    return outLPyramid_i, grad_fn

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
    for ii in range(h):
        for ij in range(w):
            for ik in range(c):
                perturbed_image = tf.identity(input_image)
                perturbed_image = tf.tensor_scatter_nd_add(
                            perturbed_image,
                            indices=[[ii, ij, ik]],
                            updates=[-eps]
                        )
                tensor_list.append((fn_i - fn(perturbed_image)) / eps)

    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0)
    autodiff_per_pixel_gradients = tf.reshape(autodiff_per_pixel_gradients,(h,w,c,h,w,c))
    return tf.transpose(autodiff_per_pixel_gradients,(3,4,5,0,1,2))

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
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    pixel_value = output_image[i, j, k]
                    pixel_gradient = tape.gradient(pixel_value, input_image)
                    tensor_list.append(pixel_gradient)
    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0)
    return tf.reshape(autodiff_per_pixel_gradients,(h,w,c,h,w,c))
    

def compare_grad():
    logr = viz.logger(opts=opts)
    input_fn = '/home/mohammad/Downloads/fft_combine/blurred.png'
    guide_fn = '/home/mohammad/Downloads/fft_combine/flash.png'

    max_levels = 3
    max_discrete_levels = 3
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
    
    # fn = lambda diffable_image: gllf_diffable(diffable_image[None,...], im_g, max_levels, max_discrete_levels, alpha_h, alpha_i, IMSZ=IMSZ, beta=beta, sigma=sigma)[0,...]

    slice_inp_i, slice_inp_g, slice_inp_p = images_to_lookup(im_i, im_g,im_i, im_g, max_levels, max_discrete_levels, alpha_h, alpha_i, IMSZ=IMSZ, beta=beta, sigma=sigma)
    # slice(slice_inp_i[1], slice_inp_g[1], slice_inp_p[1])
    # fn = lambda alpha_h: images_to_lookup(im_i, im_g,im_i, im_g, max_levels, max_discrete_levels, alpha_h[None,...], alpha_i, IMSZ=IMSZ, beta=beta, sigma=sigma)[0,...]
    # fn = dense_fn
    i = 0
    fn = lambda diffable_img:diffable_slice(diffable_img[None,...], slice_inp_g[i], slice_inp_p[i], max_discrete_levels,IMSZ)[0,...]
    alpha_h = slice_inp_i[i]

    fn_result = fn(alpha_h[0,...]) * 5
    gradients = gradient_6d(fn, alpha_h[0,...])
    finite_differences = finite_derivative(fn, alpha_h[0,...])
    g = {'fn_result':fn_result}
    g.update({'gradients_0,0,0':gradients[0,0,0,...], 'finite_differences_0,0,0':finite_differences[0,0,0,...]})
    g.update({'gradients_5,8,0':gradients[5,8,0,...], 'finite_differences_5,8,0':finite_differences[5,8,0,...]})
    
    lbl = {'fn_result':'fn_result'}
    lbl.update({'gradients_0,0,0':'gradients_0,0,0', 'finite_differences_0,0,0':'finite_differences_0,0,0'})
    lbl.update({'gradients_5,8,0':'gradients_5,8,0', 'finite_differences_5,8,0':'finite_differences_5,8,0'})

    for k,v in g.items():
        g[k] = cv2.resize(v.numpy(),(448,448))
    logr.addImage(g, lbl, 'train')
    print('hi')


def run_gllf():
    logr = viz.logger(opts=opts)
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
    fn = lambda input, guide, alpha_h, alpha_i, beta_: gllf_diffable(input, guide,input, guide, max_levels, max_discrete_levels, alpha_h, alpha_i, IMSZ=448, beta=beta_, sigma=sigma)
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

# compare_grad()