from gllf import *
from gllf import _resize
from easydict import EasyDict as edict
from guided_local_laplacian_color_local_alpha_Mullapudi2016 import guided_local_laplacian_color_local_alpha_Mullapudi2016 as guided_local_laplacian_color
def setup(testname,crop=False):
    input_fn = '/home/mohammad/Downloads/fft_combine/blurred.png'
    guide_fn = '/home/mohammad/Downloads/fft_combine/flash.png'

    max_levels = 4
    max_discrete_levels = 4
    alpha = 1
    beta  = 1.0
    sigma = 1.0
    alpha_h = 1.#tf.random.uniform(im_i.shape,0,1)
    alpha_i = 0.#tf.random.uniform(im_i.shape,0,1)
    IMSZ = 32
    im_i = imageio.imread(input_fn).astype(np.float32) / 255.0
    im_g = imageio.imread(guide_fn).astype(np.float32) / 255.0
    if(crop):
        IMSZ = 32
        im_i = tf.convert_to_tensor(im_i[None,128:128+IMSZ,128:128+IMSZ,:])
        im_g = tf.convert_to_tensor(im_g[None,128:128+IMSZ,128:128+IMSZ,:])
    else:
        IMSZ = im_i.shape[1]
        im_i = cv2.resize(im_i, (IMSZ, IMSZ))
        im_g = cv2.resize(im_g, (IMSZ, IMSZ))
        im_i = tf.convert_to_tensor(im_i[None,...])
        im_g = tf.convert_to_tensor(im_g[None,...])
    if(testname == 'test_compare_grad_slice_1d'):
        fn_grad = 'grad_test_compare_grad_slice_1d.npy'
        fn_fd = 'fd_test_compare_grad_slice_1d.npy'
    elif(testname == 'images_to_lookup_1d_grad'):
        fn_grad = 'grad_images_to_lookup_1d_grad.npy'
        fn_fd = 'fd_images_to_lookup_1d_grad.npy'
    else:
        print('unknown test')
        exit(0)
    if(os.path.exists(fn_grad)):
        grad_diff = np.load(fn_grad)
    else:
        grad_diff = None

    if(os.path.exists(fn_fd)):
        grad_fd = np.load(fn_fd)
    else:
        grad_fd = None
    return edict(im_i=im_i, im_g=im_g, max_levels=max_levels, max_discrete_levels=max_discrete_levels, alpha_h=alpha_h, alpha_i=alpha_i, beta=beta, sigma=sigma, IMSZ=IMSZ, grad_diff=grad_diff, grad_fd=grad_fd, fn_grad=fn_grad, fn_fd=fn_fd)

def tear_down():
    pass
def halide_gllf(denoised_np, flash_np, levels, alpha, beta,sigma):
    h, w, c = flash_np.shape
    flash_np = tf.transpose(flash_np, [2,0,1])
    denoised_np = tf.transpose(denoised_np, [2,0,1])
    aw = 2
    ah = 2
    alpha = np.ones([ah, aw], dtype=np.float32) * alpha
    llf_out = np.empty([3, h, w], dtype=np.float32)
    guided_local_laplacian_color(flash_np, denoised_np, levels, alpha, beta, sigma, aw, ah, w, h, llf_out)
    return tf.transpose(llf_out, [1,2,0])[None,...]
#6d gradient and finite derivative and visualization
def finite_derivative(fn, input_image,input_idx=[],output_idx=[], eps=0.01):
    """
    Takes an function fn with input h,w,c and output h,w,c and computes the finite difference of it
    a jacobian tensor with shape h,w,c,h,w,c 
    it returns a 6d jacobian, first 3 dimensions correspond to the output and last to the input
    """
    
    h, w, c = input_image.shape[-3:]
    fn_i = fn(input_image)
    assert len(input_idx) + 3 == len(input_image.shape)
    assert len(output_idx) + 3 == len(fn_i.shape)
    tensor_list = []
    for ii in tqdm.trange(h):
        for ij in range(w):
            for ik in range(c):
                perturbed_image = tf.identity(input_image)
                perturbed_image = tf.tensor_scatter_nd_add(
                            perturbed_image,
                            indices=[input_idx+[ii, ij, ik]],
                            updates=[-eps]
                        )
                tensor_list.append((tf.gather_nd(fn_i,output_idx) - tf.gather_nd(fn(perturbed_image),output_idx)) / eps)

    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0) #ihiwic,oh,ow,oc 
    autodiff_per_pixel_gradients = tf.reshape(autodiff_per_pixel_gradients,(h,w,c,h,w,c))
    return tf.transpose(autodiff_per_pixel_gradients,(3,4,5,0,1,2))#oh,ow,oc,ih,iw,ic
def gradient_6d(fn, input_image,input_idx=[],output_idx=[]):
    """takes a function fn that takes an input image and returns an image
    it returns a 6d jacobian, first 3 dimensions correspond to the output and last to the input
    Args:
        f (_type_): _description_
        var (_type_): _description_
    """
    h, w, c = input_image.shape[-3:]
    assert len(input_idx) + 3 == len(input_image.shape)
    tensor_list = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_image)
        output_image = fn(input_image)
        assert len(output_idx) + 3 == len(output_image.shape)
        # Compute gradients per output pixel
        for i in tqdm.trange(h):
            if(i == 29):
                break
            for j in range(w):
                for k in range(c):
                    pixel_value = output_image[output_idx+[i,j,k]]
                    pixel_gradient = tape.gradient(pixel_value, input_image)
                    tensor_list.append(pixel_gradient[input_idx])
    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0)#ohxowxocx,ro,co,ih,iw,ic 
    return tf.reshape(autodiff_per_pixel_gradients,(h - (h - 29),w,c,h,w,c))
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
def test_grad_6d(fn, img, gradients, finite_differences, fn_grad, fn_fd, input_idx=[],output_idx=[]):
    fn_result = fn(img)
    if(finite_differences is None):
        finite_differences = tf.abs(finite_derivative(fn, img, input_idx=input_idx, output_idx=output_idx, eps=0.0001))
        np.save(fn_fd, finite_differences)
    if(gradients is None):
        gradients = tf.abs(gradient_6d(fn, img, input_idx=input_idx, output_idx=output_idx))
        np.save(fn_grad, gradients)
    return fn_result, gradients, finite_differences

#gradient tests
def images_to_lookup_1d_grad(im_i, im_g, max_levels, max_discrete_levels, alpha_h, alpha_i, beta, sigma, IMSZ, grad_diff, grad_fd):
    imgs = tf.stack([im_i, im_g],axis=0)
    alphas = tf.stack([alpha_i, alpha_h],axis=0)
    fn = lambda diffable_imgs: images_to_lookup_1d(diffable_imgs, max_levels, max_discrete_levels, alphas, IMSZ=IMSZ, betas=[beta,0], sigmas=[sigma,0])
    input_idx=[0,0]
    output_idx=[0,0,0,0]
    fn_result, gradients, finite_differences = test_grad_6d(fn, imgs, grad_diff, grad_fd, input_idx=input_idx, output_idx=output_idx)
    visualize_grad_6d(fn_result[0,0,0,0]*10, gradients, finite_differences, 1, 1)
    maximum_relative = (tf.abs(finite_differences - gradients) / (tf.abs(finite_differences)+0.0001)).numpy().max()
    assert maximum_relative < 0.01
def resize_pyramid_grad():
    pass
def reconstruct_Laplacian_grad():
    pass
def test_compare_grad_slice_1d(configs):
    imgs = tf.stack([configs.im_i, configs.im_g],axis=0)
    alphas = tf.stack([configs.alpha_i, configs.alpha_h],axis=0)
    fn = lambda diffable_imgs: gllf_diffable_1d(diffable_imgs, alphas, configs.max_levels, configs.max_discrete_levels, betas=[configs.beta,0], sigmas=[configs.sigma,0], IMSZ=configs.IMSZ)
    input_idx=[0,0]
    output_idx=[0]
    fn_result, gradients, finite_differences = test_grad_6d(fn, imgs, configs.grad_diff, configs.grad_fd, configs.fn_grad, configs.fn_fd, input_idx=input_idx, output_idx=output_idx)
    visualize_grad_6d(fn_result[output_idx], gradients/2, finite_differences[:-3]/2, 1, 1)
    maximum_relative = (tf.abs(finite_differences[:-3] - gradients) / (tf.abs(finite_differences[:-3])+0.0001)).numpy().max()
    assert maximum_relative < 0.01

def test_compare_grad_slice_1d_intensity(configs):
    imgs = tf.stack([configs.im_g, configs.im_g],axis=0)
    alphas = tf.stack([configs.alpha_i, configs.alpha_h],axis=0)
    fn = lambda diffable_imgs: gllf_diffable_1d(diffable_imgs, alphas, configs.max_levels, configs.max_discrete_levels, betas=[configs.beta,0], sigmas=[configs.sigma,0], IMSZ=configs.IMSZ)
    fn_result = gllf_diffable_2d(imgs[0], imgs[1], configs.max_levels, configs.max_discrete_levels, configs.alpha_h, configs.alpha_i, beta=configs.beta, sigma=configs.sigma, IMSZ=448)[0]
    # fn_result = fn(imgs)[0]
    fn_result_halide = halide_gllf(imgs[0,0], imgs[1,0], configs.max_levels, configs.alpha_h, configs.beta, configs.sigma)[0]
    g = {'fn_result_3':tf.abs(fn_result)*3,'halide_3':tf.abs(fn_result_halide)*3,'fn_result':tf.abs(fn_result),'halide':tf.abs(fn_result_halide)}
    lbl = {'fn_result_3':'fn_result_3','halide_3':'halide_3','fn_result':'fn_result','halide':'halide'}
    for k,v in g.items():
        g[k] = cv2.resize(v.numpy(),(448,448))
    logr.addImage(g, lbl, 'train')
    logr.takeStep()

    assert True
configs = setup('test_compare_grad_slice_1d',crop=False)
# images_to_lookup_1d_grad(configs)
# test_compare_grad_slice_1d(configs)
test_compare_grad_slice_1d_intensity(configs)
print('hi')