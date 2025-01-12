import tensorflow as tf
import numpy as np
import tensorflow
import utils.tf_utils as tfu
import utils.utils as ut
@tf.function
def prepare_input(example, clamp=False, std_input=True):
    alpha = example['alpha']
    dimmed_ambient, _ = tfu.dim_image(
        example['ambient'], alpha=alpha)
    dimmed_warped_ambient, _ = tfu.dim_image(
        example['warped_ambient'], alpha=alpha)

    # Make the flash brighter by increasing the brightness of the
    # flash-only image.
    flash = example['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
    warped_flash = example['warped_flash_only'] * \
        ut.FLASH_STRENGTH + dimmed_warped_ambient

    sig_read = example['sig_read']
    sig_shot = example['sig_shot']
    noisy_ambient, _, _ = tfu.add_read_shot_noise(
        dimmed_ambient, sig_read=sig_read, sig_shot=sig_shot)
    noisy_flash, _, _ = tfu.add_read_shot_noise(
        warped_flash, sig_read=sig_read, sig_shot=sig_shot)

    if(clamp):
        noisy_ambient = tf.maximum(noisy_ambient,0)
        noisy_flash = tf.maximum(noisy_flash,0)
    
    noisy = tf.concat([noisy_ambient, noisy_flash], axis=-1)
    if(std_input):
        
        noise_std = tfu.estimate_std(noisy, sig_read, sig_shot)
        net_input = tf.concat([noisy, noise_std], axis=-1)
    else:
        net_input = noisy

    return net_input, alpha, noisy_flash, noisy_ambient
    
def radial_basis(i, level, w_i, sigma_i, basis_type):
    i_shape = i.shape
    i = tf.reshape(i, (1, -1))
    diff = i - level
    
    tf.assert_equal(len(w_i)%2-1,0,"number of weights should be odd")
    if(len(w_i) == 1):
        c = tf.cast(tf.convert_to_tensor([0.]), tf.float32) #weights_ct, 1
    else:
        c = tf.cast(tf.linspace(-1,1,len(w_i))[:,None],tf.float32) #weights_ct, 1
    x = diff #1, intensities
    if(basis_type == 'piecewise_linear'):
        rbf_val = piecewise_rbf(w_i, x, c, sigma_i)
    elif(basis_type == 'gaussian_1d'):
        rbf_val = gaussian_basis(w_i, x, c, sigma_i)
    else:
        print('Basis function ', basis_type, ' undefined')
        
    # result = 1 * level + 1 * (diff) + rbf_val
    return tf.reshape(rbf_val, i_shape)
    
def gaussian_basis(w, x, c, sigma):
    centered_x = (x - c)
    return tf.reduce_sum(x + w * centered_x * tf.exp(- sigma * centered_x ** 2),axis=0)

def piecewise_rbf(w, x, c, sigma,w_eps=0.01):
    centered_x = (x - c)
    y_large = tf.where(centered_x > sigma, w * sigma + 1/tf.maximum(w,w_eps) * (centered_x - sigma),0)
    y_small = tf.where(centered_x < -sigma, w * (-sigma) + 1/tf.maximum(w,w_eps) * (centered_x + sigma),0)
    y = tf.where((centered_x)**2 <= sigma **2, w * centered_x, y_large + y_small)
    return tf.reduce_sum(y,axis=0)

def rbf_debug(w, x, c, sigma):
    return w *  tf.exp(- sigma * (x - c)**2)
    
def reconstruct_Laplacian(outLPyramid, max_levels):
    #collapse pyramid
    g = [0] * max_levels
    g[max_levels - 1] = outLPyramid[max_levels - 1]
    for j in range(max_levels - 2, -1, -1):
        g[j] = _upsample(g[j + 1]) + outLPyramid[j]
    return g[0]


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