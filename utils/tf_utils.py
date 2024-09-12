import numpy as np
import tensorflow as tf
import time
import math
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple
from six.moves import range
import imageio
import cv2
import os
CONVERSION_MATRICES = {
    "xyz_to_rgb": np.array(
        (
            (3.24071, -1.53726, -0.498571),
            (-0.969258, 1.87599, 0.0415557),
            (0.0556352, -0.203996, 1.05707),
        ), dtype=np.float32
    ),
    "rgb_to_xyz": np.array(
        (
            (0.412424, 0.357579, 0.180464),
            (0.212656, 0.715158, 0.0721856),
            (0.0193324, 0.119193, 0.950444),
        ), dtype=np.float32
    ),
}


def dim_image(
        ambient, min_alpha=0.02, max_alpha=0.2, alpha=None):
    bsz = tf.shape(ambient)[0]
    if alpha is None:
        alpha = tf.pow(10., tf.random.uniform(
            [bsz, 1, 1, 1], np.log10(min_alpha), np.log10(max_alpha)))
    return alpha * ambient, alpha


def add_read_shot_noise(
    imgs, sig_read=None, sig_shot=None, 
    min_read=-3., max_read=-2, min_shot=-2., max_shot=-1.3):
    if sig_read is None or sig_shot is None:
        bsz = tf.shape(imgs)[0]
        sig_read = tf.pow(10., tf.random.uniform(
            [bsz, 1, 1, 1], min_read, max_read))
        sig_shot = tf.pow(10., tf.random.uniform(
            [bsz, 1, 1, 1], min_shot, max_shot))
    read = sig_read * tf.random.normal(tf.shape(imgs))
    shot = tf.sqrt(imgs) * sig_shot * tf.random.normal(tf.shape(imgs))
    noisy = imgs + shot + read
    return noisy, sig_read, sig_shot


def estimate_std(noisy, sig_read, sig_shot):
    return tf.sqrt(sig_read**2 + tf.maximum(0., noisy) * sig_shot**2)


def reverse_gamma(imgs, gamma=1. / 2.2):
    return imgs**(1. / gamma)


def gamma_correct(x):
    b = .0031308
    gamma = 1. / 2.4
    a = .055
    k0 = 12.92
    def gammafn(x): return (1 + a) * tf.pow(tf.maximum(x, b), gamma) - a
    srgb = tf.where(x < b, k0 * x, gammafn(x))
    k1 = (1 + a) * gamma
    srgb = tf.where(x > 1, k1 * x - k1 + 1, srgb)
    return srgb


def camera_to_rgb(imgs, color_matrix, adapt_matrix):
    b, c = tf.shape(imgs)[0], tf.shape(imgs)[-1]
    imsp = tf.shape(imgs)
    imgs = tf.reshape(tf.cast(imgs,np.float32), [b, -1, c])
    imgs = tf.transpose(imgs, [0, 2, 1])

    xyz = tf.linalg.solve(color_matrix, imgs)
    xyz = tf.linalg.matmul(adapt_matrix, xyz)
    rgb = tf.linalg.matmul(CONVERSION_MATRICES['xyz_to_rgb'][None,...], xyz)
    rgb = gamma_correct(rgb)

    rgb = tf.transpose(rgb, [0, 2, 1])
    rgb = tf.reshape(rgb, imsp)
    return rgb
def gamma_correct_np(x):
    b = .0031308
    gamma = 1. / 2.4
    a = .055
    k0 = 12.92
    def gammafn(x): return (1 + a) * np.power(np.maximum(x, b), gamma) - a
    srgb = np.where(x < b, k0 * x, gammafn(x))
    k1 = (1 + a) * gamma
    srgb = np.where(x > 1, k1 * x - k1 + 1, srgb)
    return srgb

def camera_to_rgb_np(imgs, color_matrix, adapt_matrix):
    b, c = imgs.shape[0], imgs.shape[-1]
    imsp = imgs.shape
    imgs = imgs.reshape(b, -1, c)
    imgs = imgs.transpose(0, 2, 1)

    xyz = np.linalg.solve(color_matrix, imgs)
    xyz = np.matmul(adapt_matrix, xyz)
    rgb = np.matmul(CONVERSION_MATRICES['xyz_to_rgb'][None,...], xyz)
    rgb = gamma_correct_np(rgb)

    rgb = np.transpose(rgb, [0, 2, 1])
    rgb = rgb.reshape(*imsp)
    return rgb

def get_gradient(imgs):
    return tf.concat([
        .5 * (imgs[:, 1:, :-1, :] - imgs[:, :-1, :-1, :]),
        .5 * (imgs[:, :-1, 1:, :] - imgs[:, :-1, :-1, :])], axis=-1)

@tf.function
def gradient_loss(pred, gt):
    return l1_loss(get_gradient(pred), get_gradient(gt))

def dx_tf(x):
    #x is b,h,w,c
    return tf.roll(x, [1], axis=[2]) - x
def dy_tf(x):
    #x is b,h,w,c
    return tf.roll(x, [1], axis=[1]) - x

def l1_loss(pred, gt):
    return tf.reduce_mean(tf.abs(pred - gt))

@tf.function
def l2_loss(pred, gt):
    return tf.reduce_mean(tf.square(pred - gt))

def screen_poisson(lambda_d, img,grad_x,grad_y,IMSZ):
    img_freq = tf.signal.fft2d(tf.dtypes.complex(img,tf.zeros_like(img)))
    grad_x_freq = tf.signal.fft2d(tf.dtypes.complex(grad_x,tf.zeros_like(grad_x)))
    grad_y_freq = tf.signal.fft2d(tf.dtypes.complex(grad_y,tf.zeros_like(grad_y)))
    sz = tf.complex(float(IMSZ),0.)
    tf_fftfreq_even = lambda n : tf.concat((tf.range(n/2),tf.range(-n/2,0)),0) / n
    tf_fftfreq_odd = lambda n : tf.concat((tf.range((n-1)/2+1),tf.range(-(n-1)/2,0)),0) / n
    tf_fftfreq = lambda n : tf_fftfreq_even(n) if n%2==0 else tf_fftfreq_odd(n)
    sx = tf.complex(tf_fftfreq(IMSZ),0.)
    sx = tf.reshape(sx, [-1,1])
    sx = tf.tile(sx, [1, int(IMSZ)])
    sx = tf.transpose(sx)
    sy = tf.complex(tf_fftfreq(IMSZ),0.)
    sy = tf.reshape(sy, [-1,1])
    sy = tf.tile(sy, [1, int(IMSZ)])

    # Fourier transform of shift operators
    Dx_freq = 2 * math.pi * (tf.exp(-1j * sx) - 1)
    Dy_freq = 2 * math.pi * (tf.exp(-1j * sy) - 1)
    Dx_freq = Dx_freq[None,None,...]
    Dy_freq = Dy_freq[None,None,...]
    # my_grad_x_freq = Dx_freq * img_freqs)
    # my_grad_x_freq & my_grad_y_freq should be the same as grad_x_freq & grad_y_freq
    lambda_complex = tf.dtypes.complex(lambda_d,tf.zeros_like(lambda_d))
    lambda_comp_complex = tf.dtypes.complex(1-lambda_d,tf.zeros_like(lambda_d))

    recon_freq = (lambda_complex * img_freq + lambda_comp_complex * tf.math.conj(Dx_freq) * grad_x_freq + tf.math.conj(Dy_freq) * grad_y_freq) / \
                (lambda_complex + lambda_comp_complex * (tf.math.conj(Dx_freq) * Dx_freq + tf.math.conj(Dy_freq) * Dy_freq))
    return tf.math.real(tf.signal.ifft2d(recon_freq))

def fft_combine(alpha, high_im, low_im):
    high_freq = tf.signal.fft2d(tf.dtypes.complex(img,tf.zeros_like(high_im)))
    low_freq = tf.signal.fft2d(tf.dtypes.complex(img,tf.zeros_like(low_im)))

def sigmoid(x0,k,shape):
    dim1 = tf.linspace(-1.0, 1.0, shape[-1])
    dim2 = tf.linspace(-1.0, 1.0, shape[-2])
    x, y = tf.meshgrid(dim1,dim2)
    dist = tf.sqrt(x **2 + y**2)
    kernel = 1 - 1 / (1 + tf.exp(-k*(dist-x0))) 
    kernel_complex = tf.dtypes.complex(kernel, kernel)
    inv_kernel_complex = tf.dtypes.complex(1-kernel, 1-kernel)
    return tf.signal.fftshift(kernel_complex), tf.signal.fftshift(inv_kernel_complex)

def combineFNFInFT(flash,denoise,x0,k):
    #flash and dnoise in bxcxhxw
    kernel_ft, inv_kernel_ft = sigmoid(x0,k,flash.shape)
    # inv_kernel_ft = tf.maximum(tf.dtypes.complex(0.0,0.0), tf.dtypes.complex(1.0,1.0) - kernel_ft)

    # convolve
    flash_ft = tf.signal.fft2d(tf.dtypes.complex(flash,tf.zeros_like(flash)))
    blurred_ft = tf.signal.fft2d(tf.dtypes.complex(denoise,tf.zeros_like(denoise)))

    # the 'newaxis' is to match to color direction
    blurred_ft_convolved = kernel_ft[None, None, :, :] * blurred_ft
    blurred_ift = tf.real(tf.signal.ifft2d(blurred_ft_convolved))

    flash_ft_convolved = inv_kernel_ft[None, None, :, :] * flash_ft
    flash_ift = tf.real(tf.signal.ifft2d(flash_ft_convolved))

    ft_combined = blurred_ft_convolved + flash_ft_convolved
    combined = tf.real(tf.signal.ifft2d(ft_combined))
    return combined, flash_ift, blurred_ift, kernel_ft


def get_psnr(pred, gt):
    pred = tf.clip_by_value(pred, 0., 1.)
    gt = tf.clip_by_value(gt, 0., 1.)
    mse = tf.reduce_mean((pred - gt)**2.0, axis=[1, 2, 3])
    psnr = tf.reduce_mean(-10. * tf.math.log(mse) / tf.math.log(10.))
    return psnr


def apply_filtering(imgs, kernels):
    b, h, w, c = imgs.get_shape().as_list()
    burst_length = c // 3
    b = tf.shape(imgs)[0]
    ksz = int(np.sqrt(kernels.get_shape().as_list()[-1] / burst_length / 3))
    padding = (ksz - 1) // 2
    imgs = tf.pad(imgs, [[0, 0], [padding, padding], [
                  padding, padding], [0, 0]], 'REFLECT')
    patches = tf.image.extract_patches(
        imgs, [1, ksz, ksz, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')
    patches = tf.reshape(patches, [b, h, w, ksz * ksz, burst_length * 3])
    kernels = tf.reshape(kernels, [b, h, w, ksz * ksz, burst_length * 3])
    framewise = tf.reduce_sum(patches * kernels, axis=-2)
    framewise = tf.reshape(framewise, [b, h, w, burst_length, 3])
    out = tf.reduce_sum(framewise, axis=-2)
    return out


def apply_dilated_filtering(imgs, kernels, dilation=1):
    b, h, w, c = imgs.get_shape().as_list()
    burst_length = c // 3
    b = tf.shape(imgs)[0]
    ksz = int(np.sqrt(kernels.get_shape().as_list()[-1] / burst_length / 3))
    padding = (ksz - 1) * dilation // 2
    imgs = tf.pad(
        imgs, [[0, 0], [padding, padding],
        [padding, padding], [0, 0]], 'REFLECT')
    patches = tf.image.extract_patches(
        imgs, [1, ksz, ksz, 1], [1, 1, 1, 1],
        [1, dilation, dilation, 1], 'VALID')
    patches = tf.reshape(patches, [b, h, w, ksz * ksz, burst_length * 3])
    kernels = tf.reshape(kernels, [b, h, w, ksz * ksz, burst_length * 3])
    framewise = tf.reduce_sum(patches * kernels, axis=-2)
    framewise = tf.reshape(framewise, [b, h, w, burst_length, 3])
    out = tf.reduce_sum(framewise, axis=-2)
    return out

def bilinear_filter_precompute(ksz,nchannel=3):
    if ksz == 3:
        kernel = np.array([0.5, 1., 0.5], dtype=np.float32).reshape([3, 1])
    elif ksz == 7:
        kernel = np.array([0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25],
                        dtype=np.float32).reshape([7, 1])
    kernel = np.matmul(kernel, kernel.T)
    kernel = kernel / np.sum(kernel)
    kernel = tf.tile(kernel[..., None, None], [1, 1, nchannel, 1])
    return kernel

def bilinear_filter(image, ksz, kernel=None):
    # time1 = time.time_ns() /1000000
    if(kernel is None):
        if ksz == 3:
            kernel = np.array([0.5, 1., 0.5], dtype=np.float32).reshape([3, 1])
        elif ksz == 7:
            kernel = np.array([0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25],
                            dtype=np.float32).reshape([7, 1])
        kernel = np.matmul(kernel, kernel.T)
        kernel = kernel / np.sum(kernel)
        kernel = tf.tile(kernel[..., None, None], [1, 1, tf.shape(image)[-1], 1])
    # time2 = time.time_ns() /1000000
    image = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
    # time3 = time.time_ns() /1000000
    # print('bilinear constant ', time2 - time1, ' depthwise ', time3 - time2)
    return image


def _downsample(image,
                kernel) -> tf.Tensor:
  """Downsamples the image using a convolution with stride 2.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    kernel: A tensor of shape `[H_k, W_k, C, C]`, where `H_k` and `W_k` are the
      height and width of the kernel.

  Returns:
    A tensor of shape `[B, H_d, W_d, C]`, where `H_d` and `W_d` are the height
    and width of the downsampled image.

  """
  return tf.nn.conv2d(
      input=image, filters=kernel, strides=[1, kernel.shape[0]//2, kernel.shape[1]//2, 1], padding="SAME")


def _binomial_kernel(num_channels: int,
                     dtype: tf.DType = tf.float32) -> tf.Tensor:
  """Creates a 5x5 binomial kernel.

  Args:
    num_channels: The number of channels of the image to filter.
    dtype: The type of an element in the kernel.

  Returns:
    A tensor of shape `[5, 5, num_channels, num_channels]`.
  """
  kernel = np.array((1., 4., 6., 4., 1.), dtype=dtype.as_numpy_dtype)
  kernel = np.outer(kernel, kernel)
  kernel /= np.sum(kernel)
  kernel = kernel[:, :, np.newaxis, np.newaxis]
  return tf.constant(kernel, dtype=dtype) * tf.eye(num_channels, dtype=dtype)


def _build_pyramid(image, sampler,
                   num_levels: int) -> List[tf.Tensor]:
  """Creates the different levels of the pyramid.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    sampler: A function to execute for each level (_upsample or _downsample).
    num_levels: The number of levels.

  Returns:
    A list containing `num_levels` tensors of shape `[B, H_i, W_i, C]`, where
    `H_i` and `W_i` are the height and width of the image for the level i.
  """
  kernel = _binomial_kernel(tf.shape(input=image)[3], dtype=image.dtype)
  levels = [image]
  for _ in range(num_levels):
    image = sampler(image, kernel)
    levels.append(image)
  return levels


def _split(image,
           kernel) -> Tuple[tf.Tensor, tf.Tensor]:
  """Splits the image into high and low frequencies.

  This is achieved by smoothing the input image and substracting the smoothed
  version from the input.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    kernel: A tensor of shape `[H_k, W_k, C, C]`, where `H_k` and `W_k` are the
      height and width of the kernel.

  Returns:
    A tuple of two tensors of shape `[B, H, W, C]` and `[B, H_d, W_d, C]`, where
    the first one contains the high frequencies of the image and the second one
    the low frequencies. `H_d` and `W_d` are the height and width of the
    downsampled low frequency image.

  """
  low = _downsample(image, kernel)
  high = image - _upsample(low, kernel, tf.shape(input=image))
  return high, low


def _upsample(image,
              kernel,
              output_shape = None
              ) -> tf.Tensor:
  """Upsamples the image using a transposed convolution with stride 2.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    kernel: A tensor of shape `[H_k, W_k, C, C]`, where `H_k` and `W_k` are the
      height and width of the kernel.
    output_shape: The output shape.

  Returns:
    A tensor of shape `[B, H_u, W_u, C]`, where `H_u` and `W_u` are the height
    and width of the upsampled image.
  """
  if output_shape is None:
    output_shape = tf.shape(input=image)
    output_shape = (output_shape[0], output_shape[1] * 2, output_shape[2] * 2,
                    output_shape[3])
  return tf.nn.conv2d_transpose(
      image,
      kernel * 4.0,
      output_shape=output_shape,
      strides=[1, kernel.shape[0]//2, kernel.shape[1]//2, 1],
      padding="SAME")


def downsample(image,
               num_levels: int,
               name: str = "pyramid_downsample") -> List[tf.Tensor]:
  """Generates the different levels of the pyramid (downsampling).

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    num_levels: The number of levels to generate.
    name: A name for this op that defaults to "pyramid_downsample".

  Returns:
    A list containing `num_levels` tensors of shape `[B, H_i, W_i, C]`, where
    `H_i` and `W_i` are the height and width of the downsampled image for the
    level i.

  Raises:
    ValueError: If the shape of `image` is not supported.
  """
  with tf.name_scope(name):
    image = tf.convert_to_tensor(value=image)

    # shape.check_static(tensor=image, tensor_name="image", has_rank=4)

    return _build_pyramid(image, _downsample, num_levels)


def merge(levels,
          name: str = "pyramid_merge") -> tf.Tensor:
  """Merges the different levels of the pyramid back to an image.

  Args:
    levels: A list containing tensors of shape `[B, H_i, W_i, C]`, where `B` is
      the batch size, H_i and W_i are the height and width of the image for the
      level i, and `C` the number of channels of the image.
    name: A name for this op that defaults to "pyramid_merge".

  Returns:
      A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.

  Raises:
    ValueError: If the shape of the elements of `levels` is not supported.
  """
  with tf.name_scope(name):
    levels = [tf.convert_to_tensor(value=level) for level in levels]

    for index, level in enumerate(levels):
      shape.check_static(
          tensor=level, tensor_name="level {}".format(index), has_rank=4)

    image = levels[-1]
    kernel = _binomial_kernel(tf.shape(input=image)[3], dtype=image.dtype)
    for level in reversed(levels[:-1]):
      image = _upsample(image, kernel, tf.shape(input=level)) + level
    return image


def split(image,
          num_levels: int,
          name: str = "pyramid_split") -> List[tf.Tensor]:
  """Generates the different levels of the pyramid.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    num_levels: The number of levels to generate.
    name: A name for this op that defaults to "pyramid_split".

  Returns:
    A list containing `num_levels` tensors of shape `[B, H_i, W_i, C]`, where
    `H_i` and `W_i` are the height and width of the image for the level i.

  Raises:
    ValueError: If the shape of `image` is not supported.
  """
  with tf.name_scope(name):
    image = tf.convert_to_tensor(value=image)

    # shape.check_static(tensor=image, tensor_name="image", has_rank=4)

    kernel = _binomial_kernel(tf.shape(input=image)[3], dtype=image.dtype)
    low = image
    levels = []
    for _ in range(num_levels):
      high, low = _split(low, kernel)
      levels.append(high)
    levels.append(low)
    return levels


def upsample(image,
             num_levels: int,
             name: str = "pyramid_upsample") -> List[tf.Tensor]:
  """Generates the different levels of the pyramid (upsampling).

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    num_levels: The number of levels to generate.
    name: A name for this op that defaults to "pyramid_upsample".

  Returns:
    A list containing `num_levels` tensors of shape `[B, H_i, W_i, C]`, where
    `H_i` and `W_i` are the height and width of the upsampled image for the
    level i.

  Raises:
    ValueError: If the shape of `image` is not supported.
  """
  with tf.name_scope(name):
    image = tf.convert_to_tensor(value=image)

    # shape.check_static(tensor=image, tensor_name="image", has_rank=4)

    return _build_pyramid(image, _upsample, num_levels)

def MergeLaplacian(laplacian):
    coarse = laplacian[-1]
    for im in (laplacian[::-1])[1:]:
        coarse = im + upsample(coarse, 1)[-1]
    
    return coarse

def laplacian_interpolation_linear(x, source, target):
   return x * source + (1-x) * target

def laplacian_interpolation_product(x, source, target):
   intersect = source * target
   return (1+x)*source

def laplacian_interpolation_factor(x,n,alpha, intensity):
  #  return 1/(1+tf.exp(-intensity*(x/n-alpha)))
   if(type(x) == int):
    if(x == 0):
        return 0
    elif(x == 1):
        return 0
    elif(x == 2):
        return 0
    elif(x == 3):
        return 0
    elif(x == 4):
        return 0
    elif(x == 5):
        return 0
    else:
        return 0
   else:
    x1 = []
    for x0 in x:
      if(x0 == 0):
          x1.append(0)
      elif(x0 == 1):
          x1.append(0)
      elif(x0 == 2):
          x1.append(0)
      elif(x0 == 3):
          x1.append(0)
      elif(x0 == 4):
          x1.append(0)
      elif(x0 == 5):
          x1.append(0)
      else:
          x1.append(0)
    return x1
   
def InterpolateLaplacian(source, target, alpha, intensity, interpolation_type='linear'):
    #source and target are find (index 0) to coarse (last index) pyramids
    assert len(target) == len(source), "lengths of source and target do not match"
    n = float(len(source))
    output = [0]*len(source)
    output[-1] = source[-1]
    for k, (source_laplacian, target_laplacian) in enumerate(zip(source[:-1], target[:-1])):
        beta = laplacian_interpolation_factor(k, n, alpha, intensity)
        if(interpolation_type == 'linear'):
          output[k] = laplacian_interpolation_linear(beta, source_laplacian, target_laplacian)
        if(interpolation_type == 'product'):
           output[k] = laplacian_interpolation_product(beta, source_laplacian, target_laplacian)

        
    return output

def InterpolateLaplacianPixelWise(source, target, x0):
    #source and target are find (index 0) to coarse (last index) pyramids
    assert len(target) == len(source), "lengths of source and target do not match"
    n = float(len(source))
    output = [0]*len(source)
    output[-1] = source[-1]
    for k, (i, j) in enumerate(zip(source[:-1], target[:-1])):
        beta = 1/ (1+tf.exp(-x0[k]))
        output[k] = beta * i + (1-beta) * j
        
    return output

def dumpPyramid(pyramid,dr):
    for j, i in enumerate(pyramid):
        im = np.abs(np.array(i[0,:,:,:])*10)
        cv2.imwrite(os.path.join(dr,'%i.png'%j), (np.clip(im,0,1)*255).astype(np.uint8))

def interpolated_laplacian(flash,denoise,x0,k,n):
    laplacianflash = split(flash, n)
    laplacianblur = split(denoise, n)
    return InterpolateLaplacian(laplacianblur, laplacianflash, x0,k)

def combineFNFInLaplacian(flash,denoise,x0,k,n):
   import tensorflow_graphics.image.pyramid as tfgip
    # laplacianflash = split(flash, n)
    # laplacianblur = split(denoise, n)
    # interpolatedLaplacian = InterpolateLaplacian(laplacianblur, laplacianflash, x0,k)
    # return MergeLaplacian(laplacianblur)
   laplacianflash = tfgip.split(flash,n)
   laplacianblur = tfgip.split(denoise,n)
   interpolatedLaplacian = InterpolateLaplacian(laplacianblur, laplacianflash, x0,k)
   return tfgip.merge(interpolatedLaplacian)


def combineFNFInLaplacianPixelWise(flash,denoise,x0):
    laplacianflash = split(flash, len(x0))
    laplacianblur = split(denoise, len(x0))
    interpolatedLaplacian = InterpolateLaplacianPixelWise(laplacianblur, laplacianflash, x0)
    return MergeLaplacian(interpolatedLaplacian)