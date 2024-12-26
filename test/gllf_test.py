from gllf import *
# from memory_profiler import profile, memory_usage
from gllf import _resize
from easydict import EasyDict as edict
from gllf_halide import halide_gllf
tf.config.run_functions_eagerly(True)
def setup(testname,crop=False):
    input_fn = '/home/mohammad/Downloads/fft_combine/blurred.png'
    guide_fn = '/home/mohammad/Downloads/fft_combine/flash.png'

    max_levels = 4
    max_discrete_levels = 4
    alpha = 1
    beta  = 1.0
    sigma = 1.0
    IMSZ = 448
    im_i = imageio.imread(input_fn).astype(np.float32) / 255.0
    im_g = imageio.imread(guide_fn).astype(np.float32) / 255.0
    im_i[100:110,100:110,:] = -0.05
    alpha_h = tf.Variable(0.0)#np.array(tf.random.uniform(im_i.shape,0,1))
    alpha_i = tf.Variable(0.0)#np.array(tf.random.uniform(im_i.shape,0,1))
    if(crop):
        IMSZ = 32
        im_i = tf.convert_to_tensor(im_i[None,128:128+IMSZ,128:128+IMSZ,:])
        im_g = tf.convert_to_tensor(im_g[None,128:128+IMSZ,128:128+IMSZ,:])
    else:
        # IMSZ = im_i.shape[1]
        im_i = cv2.resize(im_i, (IMSZ, IMSZ))
        im_g = cv2.resize(im_g, (IMSZ, IMSZ))
        # alpha_i = cv2.resize(alpha_i, (IMSZ, IMSZ))
        # alpha_h = cv2.resize(alpha_h, (IMSZ, IMSZ))
        im_i = tf.convert_to_tensor(im_i[None,...])
        im_g = tf.convert_to_tensor(im_g[None,...])
        # alpha_i = tf.convert_to_tensor(alpha_i[None,...])
        # alpha_h = tf.convert_to_tensor(alpha_h[None,...])
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

#6d gradient and finite derivative and visualization
def finite_derivative(fn, input_image,input_size,output_size,input_idx=[],output_idx=[], eps=0.005):
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
    for ii in tqdm.trange(input_size[0]):
        for ij in range(input_size[1]):
            for ik in range(input_size[2]):
                perturbed_image = tf.identity(input_image)
                perturbed_image = tf.tensor_scatter_nd_add(
                            perturbed_image,
                            indices=[input_idx+[ii, ij, ik]],
                            updates=[-eps]
                        )
                tensor_list.append((tf.gather_nd(fn_i,output_idx) - tf.gather_nd(fn(perturbed_image),output_idx)) / eps)

    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0) #ihiwic,oh,ow,oc 
    autodiff_per_pixel_gradients = tf.reshape(autodiff_per_pixel_gradients,(*input_size,h,w,c))
    return tf.transpose(autodiff_per_pixel_gradients,(3,4,5,0,1,2))[:output_size[0],:output_size[1],:output_size[2],...]#oh,ow,oc,ih,iw,ic

def gradient_6d(fn, input_image,input_size,output_size,input_idx=[],output_idx=[]):
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
        for i in tqdm.trange(output_size[0]):
            for j in range(output_size[1]):
                for k in range(output_size[2]):
                    pixel_value = output_image[output_idx+[i,j,k]]
                    pixel_gradient = tape.gradient(pixel_value, input_image)
                    tensor_list.append(pixel_gradient[input_idx])
    autodiff_per_pixel_gradients = tf.stack(tensor_list,axis=0)#ohxowxocx,ro,co,ih,iw,ic 
    return tf.reshape(autodiff_per_pixel_gradients,(*output_size,h,w,c))[...,:input_size[0],:input_size[1],:input_size[2]]

def gradient_0d(fn, input_image,input_idx=[],output_idx=[]):
    """takes a function fn that takes an input image and returns an image
    it returns a 6d jacobian, first 3 dimensions correspond to the output and last to the input
    Args:
        f (_type_): _description_
        var (_type_): _description_
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_image)
        output_image = fn(input_image)
    return tape.jacobian(output_image, input_image)
        
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
def test_grad_6d(fn, img, gradients, finite_differences, fn_grad, fn_fd, input_size, output_size, input_idx=[],output_idx=[]):
    fn_result = fn(img)
    if(gradients is None):
        gradients = tf.abs(gradient_6d(fn, img, input_size=input_size, output_size=output_size, input_idx=input_idx, output_idx=output_idx))
        # np.save(fn_grad, gradients)
    if(finite_differences is None):
        finite_differences = tf.abs(finite_derivative(fn, img, input_size=input_size, output_size=output_size, input_idx=input_idx, output_idx=output_idx, eps=0.03))
        # np.save(fn_fd, finite_differences)
    return fn_result, gradients, finite_differences

def finite_derivative_0d(fn, input_image,input_idx=[],output_idx=[], eps=0.005):
    """
    Takes an function fn with input h,w,c and output h,w,c and computes the finite difference of it
    a jacobian tensor with shape h,w,c,h,w,c 
    it returns a 6d jacobian, first 3 dimensions correspond to the output and last to the input
    """
    
    fn_i = fn(input_image)
    fn_i_n1 = fn(input_image - eps)
    return (fn_i - fn_i_n1) / eps
    
def test_grad_0d(fn, img, gradients, finite_differences, fn_grad, fn_fd, input_size, output_size, input_idx=[],output_idx=[]):
    fn_result = fn(img)
    if(gradients is None):
        gradients = tf.abs(gradient_0d(fn, img, input_idx=input_idx, output_idx=output_idx))
        # np.save(fn_grad, gradients)
    if(finite_differences is None):
        finite_differences = tf.abs(finite_derivative_0d(fn, img, input_idx=input_idx, output_idx=output_idx, eps=0.001))
        # np.save(fn_fd, finite_differences)
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
    input_size = [8,8,1]
    output_size = [8,8,1]
    fn_result, gradients, finite_differences = test_grad_6d(fn, imgs, configs.grad_diff, configs.grad_fd, configs.fn_grad, configs.fn_fd, input_size=input_size, output_size=output_size, input_idx=input_idx, output_idx=output_idx)
    visualize_grad_6d(fn_result[output_idx], gradients/2, finite_differences[:-3]/2, 1, 1)
    maximum_relative = (tf.abs(finite_differences[:-3] - gradients) / (tf.abs(finite_differences[:-3])+0.0001)).numpy().max()
    assert maximum_relative < 0.01

def test_compare_alpha_grad_slice_1d(configs):
    imgs = tf.stack([configs.im_i, configs.im_g],axis=0)
    alphas = tf.stack([configs.alpha_i, configs.alpha_h],axis=0)
    fn = lambda diffable_alpha: gllf_diffable_1d(imgs, diffable_alpha, configs.max_levels, configs.max_discrete_levels, betas=[configs.beta,0], sigmas=[configs.sigma,0], IMSZ=configs.IMSZ)
    input_idx=[0,0]
    output_idx=[0]
    input_size = [16,16,1]
    output_size = [16,16,1]
    fn_result, gradients, finite_differences = test_grad_6d(fn, alphas, configs.grad_diff, configs.grad_fd, configs.fn_grad, configs.fn_fd, input_size=input_size, output_size=output_size, input_idx=input_idx, output_idx=output_idx)
    visualize_grad_6d(fn_result[output_idx], gradients/2, finite_differences[:-3]/2, 1, 1)
    maximum_relative = (tf.abs(finite_differences[:-3] - gradients) / (tf.abs(finite_differences[:-3])+0.0001)).numpy().max()
    assert maximum_relative < 0.01
def test_compare_alpha_0d_grad_slice_1d(configs):
    imgs = tf.stack([configs.im_i, configs.im_g],axis=0)
    # alphas = tf.stack([configs.alpha_i, configs.alpha_h],axis=0)
    alphas = [configs.alpha_i, configs.alpha_h]
    fn = lambda diffable_alpha: gllf_diffable_1d(imgs, [diffable_alpha, configs.alpha_h], configs.max_levels, configs.max_discrete_levels, betas=[configs.beta,0], sigmas=[configs.sigma,0], IMSZ=configs.IMSZ)
    input_idx=[0,0]
    output_idx=[0]
    input_size = [16,16,1]
    output_size = [16,16,1]
    fn_result, gradients, finite_differences = test_grad_0d(fn, configs.alpha_i, configs.grad_diff, configs.grad_fd, configs.fn_grad, configs.fn_fd, input_size=input_size, output_size=output_size, input_idx=input_idx, output_idx=output_idx)
    visualize_grad_6d(fn_result[output_idx], gradients/2, finite_differences[:-3]/2, 1, 1)
    maximum_relative = (tf.abs(finite_differences[:-3] - gradients) / (tf.abs(finite_differences[:-3])+0.0001)).numpy().max()
    assert maximum_relative < 0.01
#gradient numerical tests
# def test_compare_grad_slice_1d(configs):
    # imgs = tf.stack([configs.im_i, configs.im_g],axis=0)
    # alphas = tf.stack([configs.alpha_i, configs.alpha_h],axis=0)
    # fn = lambda diffable_imgs: gllf_diffable_1d(diffable_imgs, alphas, configs.max_levels, configs.max_discrete_levels, betas=[configs.beta,0], sigmas=[configs.sigma,0], IMSZ=configs.IMSZ)
    # input_idx=[0,0]
    # output_idx=[0]
    # input_size = [16,16,1]
    # output_size = [16,16,1]
    # fn_result, gradients, finite_differences = test_grad_6d(fn, imgs, configs.grad_diff, configs.grad_fd, configs.fn_grad, configs.fn_fd, input_size=input_size, output_size=output_size, input_idx=input_idx, output_idx=output_idx)
    # visualize_grad_6d(fn_result[output_idx], gradients/20, finite_differences/20, 1, 1)
    # maximum_relative = (tf.abs(finite_differences - gradients) / (tf.abs(finite_differences)+0.0001)).numpy().max()
    # print('maximum_relative ',maximum_relative)
    # assert maximum_relative < 0.01

def test_compare_grad_slice_2d(configs):
    imgs = tf.stack([configs.im_i, configs.im_g],axis=0)
    alphas = tf.stack([configs.alpha_i, configs.alpha_h],axis=0)
    fn = lambda diffable_imgs: gllf_diffable_2d(diffable_imgs[0],diffable_imgs[1], configs.max_levels, configs.max_discrete_levels, alphas[0], alphas[1],IMSZ=configs.IMSZ)
    input_idx=[0,0]
    output_idx=[0]
    input_size = [8,8,1]
    output_size = [8,8,1]
    fn_result, gradients, finite_differences = test_grad_6d(fn, imgs, configs.grad_diff, configs.grad_fd, configs.fn_grad, configs.fn_fd, input_size=input_size, output_size=output_size, input_idx=input_idx, output_idx=output_idx)
    visualize_grad_6d(fn_result[output_idx], gradients/20, finite_differences/20, 1, 1)
    maximum_relative = (tf.abs(finite_differences - gradients) / (tf.abs(finite_differences)+0.0001)).numpy().max()
    print('maximum_relative ',maximum_relative)
    assert maximum_relative < 0.01
    
# from guided_local_laplacian_color_local_alpha_Mullapudi2016 import guided_local_laplacian_color_local_alpha_Mullapudi2016 as guided_local_laplacian_color
# def halide_gllf(denoised_np, flash_np, levels, alpha, beta,sigma):
#     h, w, c = flash_np.shape
#     flash_np = tf.transpose(flash_np, [2,0,1])
#     denoised_np = tf.transpose(denoised_np, [2,0,1])
#     aw = 2
#     ah = 2
#     alpha = np.ones([ah, aw], dtype=np.float32) * alpha
#     llf_out = np.empty([3, h, w], dtype=np.float32)
#     guided_local_laplacian_color(flash_np, denoised_np, levels, alpha, beta, sigma, aw, ah, w, h, llf_out)
#     return tf.transpose(llf_out, [1,2,0])[None,...]
#     *********
# (3, 448, 448) <dtype: 'float32'> (3, 448, 448) <dtype: 'float32'> (3, 448, 448) float32
# 4 [[1. 1.]
#  [1. 1.]] 1.0 1.0 2 2 448 448
# *********
#comparison with halide
def test_compare_gllf_2d_tf_vs_halide(configs):
    imgs = tf.stack([configs.im_i, configs.im_g],axis=0)
    alphas = tf.stack([configs.alpha_i, configs.alpha_h],axis=0)
    fn_result = gllf_diffable_2d(imgs[0], imgs[1], configs.max_levels, configs.max_discrete_levels, configs.alpha_i, configs.alpha_h, beta=configs.beta, sigma=configs.sigma, IMSZ=configs.IMSZ)[0]
    # fn_result = fn(imgs)[0]
    fn_result_halide = halide_gllf(imgs[0,0], imgs[1,0], configs.max_levels, configs.alpha_i, configs.beta, configs.sigma)[0]
    g = {'fn_result_3':tf.abs(fn_result)*3,'halide_3':tf.abs(fn_result_halide)*3,'fn_result':tf.abs(fn_result),'halide':tf.abs(fn_result_halide)}
    lbl = {'fn_result_3':'fn_result_3','halide_3':'halide_3','fn_result':'fn_result','halide':'halide'}
    for k,v in g.items():
        g[k] = cv2.resize(v.numpy(),(448,448))
    logr.addImage(g, lbl, 'train')
    logr.takeStep()

    assert True
def test_compare_gllf_1d_tf_vs_2d(configs):
    imgs = tf.stack([configs.im_i, configs.im_g],axis=0)
    alphas = tf.stack([configs.alpha_i, configs.alpha_h],axis=0)
    
    fn = lambda diffable_imgs: gllf_diffable_1d(diffable_imgs, alphas, configs.max_levels, configs.max_discrete_levels, betas=[configs.beta,0], sigmas=[configs.sigma,0], min_intensity=0.0, max_intensity=1.0, IMSZ=configs.IMSZ)
    
    
    fn_result_1d = fn(imgs)[0]
    fn_result = gllf_diffable_2d(imgs[0], imgs[1], configs.max_levels, configs.max_discrete_levels, configs.alpha_i, configs.alpha_h, beta=configs.beta, sigma=configs.sigma, IMSZ=configs.IMSZ)[0]
    # fn_result = fn(imgs)[0]
    
    g = {'fn_result_3':tf.abs(fn_result)*3,'fn_result_1_3':tf.abs(fn_result_1d)*3,'fn_result':tf.abs(fn_result),'fn_result_1':tf.abs(fn_result_1d)}
    lbl = {'fn_result_3':'fn_result_3','fn_result_1_3':'fn_result_1_3','fn_result':'fn_result','fn_result_1':'fn_result_1'}
    for k,v in g.items():
        g[k] = cv2.resize(v.numpy(),(448,448))
    logr.addImage(g, lbl, 'train')
    logr.takeStep()

    assert True
configs = setup('test_compare_grad_slice_1d',crop=False)
# images_to_lookup_1d_grad(configs)
# test_compare_grad_slice_1d(configs)
# test_compare_alpha_0d_grad_slice_1d(configs)
# test_compare_alpha_grad_slice_1d(configs)
# test_compare_grad_slice_2d(configs)
# test_compare_gllf_2d_tf_vs_halide(configs)
test_compare_gllf_1d_tf_vs_2d(configs)
print('hi')