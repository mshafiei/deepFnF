
def CreateNetwork(opts):
    
    if(opts.model == 'deepfnf_llf_diffable'):
        from net_llf_tf2_diffable import Net as netLLF
        model = netLLF(opts.llf_alpha, opts.llf_beta, opts.llf_levels, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_llf_overfit'):
        from net_llf_tf2_local_alpha_overfitting import Net as net_overfit
        model = net_overfit(alpha_width=opts.alpha_width, alpha_height=opts.alpha_height, llf_beta=opts.llf_beta, llf_levels=opts.llf_levels, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_llf_alpha_map_unet'):
        from net_llf_tf2_local_alpha_unet import Net as net_unet
        model = net_unet(alpha_width=opts.alpha_width, alpha_height=opts.alpha_height, llf_beta=opts.llf_beta, llf_levels=opts.llf_levels, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_llf_alpha_map_unet_v2'):
        from net_llf_tf2_local_alpha_unet_v2 import Net as net_unet_v2
        model = net_unet_v2(llf_sigma=opts.llf_sigma,alpha_width=opts.alpha_width, alpha_height=opts.alpha_height, llf_beta=opts.llf_beta, llf_levels=opts.llf_levels, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_llf_alpha_map_image_v2'):
        from net_llf_tf2_local_alpha_overfitting_v2 import Net as net_image_v2
        model = net_image_v2(llf_sigma=opts.llf_sigma,alpha_width=opts.alpha_width, alpha_height=opts.alpha_height, llf_beta=opts.llf_beta, llf_levels=opts.llf_levels, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_llf_alpha_map_image'):
        from net_llf_tf2_local_alpha_overfitting import Net as net_unet
        model = net_unet(alpha_width=opts.alpha_width, alpha_height=opts.alpha_height, llf_beta=opts.llf_beta, llf_levels=opts.llf_levels, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_llf_scalar_alpha_encoder'):
        from net_llf_tf2_diffable_encoder import Net as net_unet
        model = net_unet(llf_beta=opts.llf_beta, llf_levels=opts.llf_levels, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_llf_scalar_alpha'):
        from net_llf_tf2_diffable_scalar import Net as net_unet
        model = net_unet(llf_beta=opts.llf_beta, llf_levels=opts.llf_levels, ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor,lmbda=opts.lmbda)
    elif(opts.model == 'deepfnf_refine_unet'):
        from net_deepfnf_unet_refinement import Net as deepfnf_refine_net
        model = deepfnf_refine_net(ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor)

    import net
    deepfnf_model = net.Net(ksz=opts.ksz, num_basis=opts.num_basis, burst_length=2,channels_count_factor=opts.channels_count_factor)
    return model, deepfnf_model

def composite_centered_numpy(small_image, large_image, x, y):
    # Get dimensions of both images (C, H, W)
    channels, large_height, large_width = large_image.shape
    _, small_height, small_width = small_image.shape

    # Calculate the top-left corner where the large image should be placed
    top_left_x = x - large_width // 2
    top_left_y = y - large_height // 2

    # Determine the region in the small image where the large image will be pasted
    x_start_small = max(0, top_left_x)
    y_start_small = max(0, top_left_y)
    x_end_small = min(small_width, top_left_x + large_width)
    y_end_small = min(small_height, top_left_y + large_height)

    # Calculate corresponding regions in the large image
    x_start_large = max(0, -top_left_x)
    y_start_large = max(0, -top_left_y)
    x_end_large = x_start_large + (x_end_small - x_start_small)
    y_end_large = y_start_large + (y_end_small - y_start_small)

    # Paste the portion of the large image that fits into the small image
    small_image[:, y_start_small:y_end_small, x_start_small:x_end_small] = large_image[:, y_start_large:y_end_large, x_start_large:x_end_large]

    return small_image