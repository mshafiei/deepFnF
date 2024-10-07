from guided_local_laplacian_color_Mullapudi2016 import guided_local_laplacian_color_Mullapudi2016 as gllf
from guided_local_laplacian_color_grad_Mullapudi2016 import guided_local_laplacian_color_grad_Mullapudi2016 as gllf_g
import numpy as np
import cv2
import halide as hl
import halide.imageio as io
import os
import cv2
#inputs start
levels = 2
beta = 1.
alpha = 0.5 #0 close to denoise image, 1 close to flash
eps = 0.1
lr = 0.01
fn_flash = '/home/mohammad/Downloads/fft_combine/flash.png'
fn_denoise = '/home/mohammad/Downloads/fft_combine/blurred.png'
do_store_image = True
do_store_image_fd = do_store_image
#inputs end
wh = 1024

flash_fn = '/home/mohammad/Downloads/fft_combine/flash.png'
blur_fn = '/home/mohammad/Downloads/fft_combine/blurred.png'

i_flash = (hl.imageio.imread(flash_fn) / 255).astype(np.float32).transpose(1,2,0)
i_denoise = (hl.imageio.imread(blur_fn) / 255).astype(np.float32).transpose(1,2,0)

# i_flash = np.ones((wh,wh,3), dtype=np.float32)*0.6
# # i_denoise = np.ones((wh,wh,3), dtype=np.float32)*0.4
# # i_flash[100:200,:,:] = 0.2
# i_flash[100:200,100:200,:] = 0.3
# # i_denoise = i_flash

# i_flash = cv2.imread(fn_flash) / 255.
# i_denoise = cv2.imread(fn_denoise) / 255.
i_denoise = cv2.resize(i_denoise,(i_flash.shape[1],i_flash.shape[0]))

i_flash = np.ascontiguousarray(i_flash.transpose(2,0,1),dtype=np.float32)
i_denoise = np.ascontiguousarray(i_denoise.transpose(2,0,1),dtype=np.float32)

def store_image(img, fn):
    if('png' in fn):
        img_u8 = (img * 255).astype(np.uint8)
        io.imwrite(fn, img_u8)
    elif('exr' in fn):
        cv2.imwrite(fn, img.transpose(1,2,0))

def gllf_call(i_flash, i_denoise, levels, alpha, beta):
    combined = np.empty_like(i_flash)
    intensity_max = 1#max(i_flash.max(), i_denoise.max())
    intensity_min = 0#min(i_flash.min(), i_denoise.min())
    denoised_np = (i_denoise - intensity_min) / (intensity_max - intensity_min)
    flash_np = (i_flash - intensity_min) / (intensity_max - intensity_min)
    gllf(flash_np, denoised_np, levels, alpha / (levels-1), beta, combined)
        
    return combined

def gllf_g_call(i_flash, i_denoise, levels, alpha, beta):
    combined_g = np.empty_like(i_flash)
    intensity_max = 1#max(i_flash.max(), i_denoise.max())
    intensity_min = 0#min(i_flash.min(), i_denoise.min())
    denoised_np = (i_denoise - intensity_min) / (intensity_max - intensity_min)
    flash_np = (i_flash - intensity_min) / (intensity_max - intensity_min)
    gllf_g(flash_np, denoised_np, levels, alpha / (levels-1), beta, combined_g)
        
    return combined_g

def f(f_alpha, I):
    return f_alpha - I

def gllf_alpha_delta(i_flash, i_denoise, levels, alpha_i, eps, beta):
    #finite difference w.r.t. alpha
    f_alpha_1 = gllf_call(i_flash, i_denoise, levels, alpha_i, beta)
    f_alpha_n1 = gllf_call(i_flash, i_denoise, levels, alpha_i - eps, beta)
    f_alpha_g = gllf_g_call(i_flash, i_denoise, levels, alpha_i, beta)
    j = (f_alpha_1 - f_alpha_n1) / (eps)

    f_val = f(f_alpha_1, i_flash)
    return -np.matmul(j.reshape(-1),f_val.reshape(-1)) / np.matmul(j.reshape(-1),j.reshape(-1)), np.abs(f_val).sum(), f_alpha_1, f_alpha_n1, j * 0.5 + 0.5, f_alpha_g * 0.5 + 0.5



for i in range(1):
    d, err, f_alpha_1, f_alpha_n1, f_fd, f_g = gllf_alpha_delta(i_flash, i_denoise, levels, alpha, eps, beta)
    if(do_store_image):
        fn_1 = './deriv_output/image_%03i_alpha_1_%.03f.exr' % (i, alpha)
        fn_n1 = './deriv_output/image_%03i_alpha_n1_%.03f.exr' % (i, alpha)
        fn_fd = './deriv_output/image_%03i_alpha_%.03f_fd.exr' % (i, alpha)
        fn_g = './deriv_output/image_%03i_alpha_%.03f_g.exr' % (i, alpha)
        fn_flash = './deriv_output/image_flash.png'
        fn_blur = './deriv_output/image_blur.png'
        os.makedirs(os.path.dirname(os.path.abspath(fn_1)), exist_ok=True)
        store_image(f_alpha_1, fn_1)
        store_image(f_alpha_n1, fn_n1)
        store_image(f_fd, fn_fd)
        store_image(f_g, fn_g)
        if(not os.path.exists(fn_flash)):
            store_image(i_flash, fn_flash)
        if(not os.path.exists(fn_blur)):
            store_image(i_denoise, fn_blur)
    print('d, ', d, 'alpha ', alpha, ' error ', err)
    alpha += lr * d

print('hi')