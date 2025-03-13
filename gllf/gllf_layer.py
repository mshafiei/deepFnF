import tensorflow as tf
from gllf.gllf_utils import *
import tensorflow.experimental.numpy as tnp
from tiny_unet_adjustable import Net as tiny_unet
import utils.tf_utils as tfu
import cvgutils.Viz as viz
from easydict import EasyDict as edict
#encode: a vector per intensity layer
# each vector represents an odd basis function

@tf.numpy_function(Tout=tf.float32)
def gllf_diffable_1d_halide(im_is, IMSZ, range_weights, w_i, sigma_i, img_ct, max_levels, image_weights_0, image_weights_1, image_weights_2, image_weights_3):
    from guided_local_laplacian_color_neural_local_alpha_Mullapudi2016 import guided_local_laplacian_color_neural_local_alpha_Mullapudi2016 as guided_local_laplacian_color
    llf_out = np.empty([3, IMSZ, IMSZ], dtype=np.float32)
    _, s_h, s_w, _, _ = image_weights_0.shape
    input_contiguous = np.ascontiguousarray(im_is[0].transpose(3,2,1,0))
    guided_local_laplacian_color(input_contiguous, max_levels, 
                        np.ascontiguousarray(image_weights_0.transpose(4,3,2,1,0)), 
                        np.ascontiguousarray(image_weights_1.transpose(4,3,2,1,0)), 
                        np.ascontiguousarray(image_weights_2.transpose(4,3,2,1,0)),
                        np.ascontiguousarray(image_weights_3.transpose(4,3,2,1,0)), 
                        np.ascontiguousarray(range_weights.transpose(2,1,0)),
                        np.ascontiguousarray(w_i.transpose(1,0)), 
                        np.ascontiguousarray(sigma_i.transpose(1,0)), img_ct, s_w, s_h, IMSZ, IMSZ, llf_out)
    return llf_out.transpose(2,1,0)[None,...]
class gllf_layer_radial(tiny_unet):
    def __init__(self, llf_levels, llf_intensity_levels, llf_remap_function, rbf_weights_ct, yuv_gllf, alphas, betas, sigmas, thresholds, downsample_ct, use_halide_implementation=False, img_ct=None, gaussian_weights_scale=10,gaussian_sigma_offset=3, piecewise_linear_weight_max=3, piecewise_linear_sigma=0.2, basis_ct=1,unet_output_size=6, min_intensity=0.0, max_intensity=1.0, IMSZ=448,**kwargs):
        super().__init__(downsample_ct, unet_output_size=unet_output_size,**kwargs)
        self.max_levels = llf_levels
        self.max_discrete_levels = llf_intensity_levels
        self.min_intensity=min_intensity
        self.max_intensity=max_intensity
        self.llf_remap_function = llf_remap_function
        self.IMSZ=IMSZ
        self.scalar_alphas_options = alphas
        self.scalar_betas_options = betas
        self.scalar_sigmas_options = sigmas
        self.yuv_gllf = yuv_gllf
        self.thresholds = thresholds
        self.rbf_weights_ct = rbf_weights_ct
        self.gaussian_weights_scale=gaussian_weights_scale
        self.gaussian_sigma_offset=gaussian_sigma_offset
        self.piecewise_linear_weight_max=piecewise_linear_weight_max
        self.piecewise_linear_sigma=piecewise_linear_sigma
        self.use_halide_implementation = use_halide_implementation
        if(use_halide_implementation):
            from guided_local_laplacian_color_neural_local_alpha_Mullapudi2016 import guided_local_laplacian_color_neural_local_alpha_Mullapudi2016 as guided_local_laplacian_color
            self.halide_gllf = guided_local_laplacian_color
        if(self.thresholds is not None):
            for i in range(len(thresholds)):
                if(thresholds[i] < 0):
                    self.thresholds[i] = None
        self.alpha_count = 3*4*3
        self.basis_ct = basis_ct
        self.img_ct = img_ct

    
    def gllf(self, diffable_imgs, bottleneck, reconstruct_gllf_pyramids=False):
        self.scalar_alphas_net(bottleneck)
        tf.debugging.assert_equal(self.img_ct, len(diffable_imgs))
        imgs = []
        for im in diffable_imgs:
            if(self.yuv_gllf):
                imgs.append(tfu.rgb_to_yuv(im))
            else:
                imgs.append(im)
        
        if(reconstruct_gllf_pyramids):
            visualization = gllf_layer_radial.visualize(self,im_is=imgs)
            if(self.yuv_gllf):
                for i in range(len(visualization)):
                    if('remapping' in visualization[i].key):
                        continue
                    visualization[i].image = tfu.yuv_to_rgb(visualization[i].image)
                return visualization
            else:
                for i in range(len(visualization)):
                    if('remapping' in visualization[i].key):
                        continue
                    visualization[i].image = visualization[i].image
                return visualization
        else:
            # im = 0
            # for i in imgs:
            #     im += i
            # return im
            if(self.use_halide_implementation):
                output = gllf_diffable_1d_halide(tf.stack(imgs,axis=-1), self.IMSZ, self.range_weights, self.w_i, self.sigma_i, self.img_ct, self.max_levels, self.image_weights[0], self.image_weights[1], self.image_weights[2], self.image_weights[3])
                # return imgs[-1]
            else:
                output = self.gllf_diffable_1d(imgs)
            if(self.yuv_gllf):
                    return tfu.yuv_to_rgb(output)
            else:
                    return output


    def remap(self,fx, alpha):
        # fx = fx * 256
        return alpha * fx * tf.exp(-fx * fx / 2.0)

    def scalar_alphas_net(self, bottleneck):
        pfx='scalars_'
        
        #Nx4x4ximage_count
        if(self.llf_remap_function == 'exp_1d'):
            out_scalars, _ = self.down_block(bottleneck, self.channel_count(512), pfx + 'alpha_down5')#1,8,8,64
            out_scalars, _ = self.down_block(out_scalars, self.channel_count(256), pfx + 'alpha_down6') #1,4,4,32
            out_scalars = self.conv(pfx + 'alpha_bottleneck_1', out_scalars, self.channel_count(128),relu=False, ksz=1)#1,4,4,16
            out_scalars = self.conv(pfx + 'alpha_bottleneck_2', out_scalars, self.channel_count(64),relu=False, ksz=1)
            out_scalars = self.conv(pfx + 'alpha_bottleneck_3', out_scalars, self.channel_count(32),relu=False, ksz=1)
            out_scalars = self.conv(pfx + 'alpha_bottleneck_4', out_scalars, self.channel_count(16),relu=False, sigmoid=True, activation_name=pfx + 'bottleneck', ksz=1)
            # out_scalars = self.conv(pfx + 'alpha_bottleneck_5', out_scalars, self.channel_count(8),relu=False, ksz=1)
            # out_scalars = self.conv(pfx + 'alpha_bottleneck_6', out_scalars, self.channel_count(4),relu=False, ksz=1)# 1, 4, 4, 1
            # out_scalars = self.conv(pfx + 'alpha_bottleneck_7', out_scalars, 1,relu=False, activation_name=pfx + 'bottleneck', ksz=1) #1x4x4x1
            self.scalar_alphas = tf.reshape(out_scalars * 2 - 1,(1,2, self.max_discrete_levels, 4)) #16 basis functions
            self.alphas = self.scalar_alphas[0,:,:,0]
            # self.alphas = [[self.scalar_alphas[0,0,0,0], self.scalar_alphas[0,0,0,0], self.scalar_alphas[0,0,0,0], self.scalar_alphas[0,0,0,0]]]
            self.alphas = []
            self.betas = []
            self.sigmas = []
            for j in range(self.img_ct):
                betas_j = []
                sigmas_j = []
                alphas_j = []
                for i in range(self.max_discrete_levels):
                    alphas_j.append(self.scalar_alphas[0,j,i,0])
                    sigmas_j.append(1.)
                    if(j == 0):
                        betas_j.append(1.)
                    else:
                        betas_j.append(self.scalar_alphas[0,j,i,1])
                        # sigmas_j.append(self.scalar_alphas[0,j,i,2])
                self.alphas.append(alphas_j)
                self.betas.append(betas_j)
                self.sigmas.append(sigmas_j)
            #alphas, betas, sigmas are img_ct, discrete_intensities
            # self.alphas = [self.scalar_alphas_options[0] * self.scalar_alphas[0,0], self.scalar_alphas_options[1] * self.scalar_alphas[0,1], self.scalar_alphas_options[2] * self.scalar_alphas[0,2],  self.scalar_alphas_options[3] * self.scalar_alphas[0,3]]
            # self.betas =  [1,                                                       self.scalar_betas_options[1]  * self.scalar_alphas[0,5], self.scalar_betas_options[2]  * self.scalar_alphas[0,6],  self.scalar_betas_options[3]  * self.scalar_alphas[0,7]]
            # self.sigmas = [1,                                                       self.scalar_sigmas_options[1] * self.scalar_alphas[0,9], self.scalar_sigmas_options[2] * self.scalar_alphas[0,10], self.scalar_sigmas_options[3] * self.scalar_alphas[0,11]]
            self.range_weights      = tf.ones((1,self.img_ct, self.max_discrete_levels))
            # self.image_weights      = tf.ones((self.max_levels))
            self.image_weights = []
            for i in range(self.max_levels):
                spatial_weights = bottleneck
                for j in range(i):
                    spatial_weights, _  = self.down_block(spatial_weights,  self.channel_count(1024), pfx + 'image_basis_down%i_%i' % (i,j))
                #convolve, store and downsample
                image_weights    = self.conv(pfx + 'spatial_bottleneck_%i_1' % i, spatial_weights, self.channel_count(512//(2**i)), relu=False, ksz=1)
                image_weights    = self.conv(pfx + 'spatial_bottleneck_%i_2' % i, image_weights, self.channel_count(512//(2**(i+1))), relu=False, ksz=1)
                image_weights    = self.conv(pfx + 'spatial_bottleneck_%i_3' % i, image_weights, self.channel_count(512//(2**(i+2))), relu=False, ksz=1)
                image_weights    = self.conv(pfx + 'spatial_bottleneck_%i_4' % i, image_weights, self.img_ct * 3, relu=False, ksz=1)
                image_weights    = self.conv(pfx + 'spatial_bottleneck_%i_5' % i, image_weights, self.img_ct * 3, relu=False, ksz=1)
                image_weights    = tf.reshape(image_weights,(*image_weights.shape[:3], self.img_ct, 3))+0.5
                self.image_weights.append(image_weights)
        elif(self.llf_remap_function == 'manual_1d'):
            self.alphas = [[-1, -1, -1, -1]]
            self.betas = [[1.,1.,1.,1.]]
            self.sigmas = [[1.,1.,1.,1.]]
            self.range_weights      = tf.ones((1,self.img_ct, self.max_discrete_levels))
            self.image_weights      = tf.ones((self.max_levels))
        elif(self.llf_remap_function == 'exp_2d'):
            out_scalars, _ = self.down_block(bottleneck, self.channel_count(512), pfx + 'alpha_down5')#1,8,8,64
            out_scalars, _ = self.down_block(out_scalars, self.channel_count(256), pfx + 'alpha_down6') #1,4,4,32
            out_scalars = self.conv(pfx + 'alpha_bottleneck_1', out_scalars, self.channel_count(128),relu=False, ksz=1)#1,4,4,16
            out_scalars = self.conv(pfx + 'alpha_bottleneck_2', out_scalars, self.channel_count(64),relu=False, ksz=1)
            out_scalars = self.conv(pfx + 'alpha_bottleneck_3', out_scalars, self.channel_count(32),relu=False, ksz=1)
            out_scalars = self.conv(pfx + 'alpha_bottleneck_4', out_scalars, self.channel_count(16),relu=False, ksz=1)
            out_scalars = self.conv(pfx + 'alpha_bottleneck_5', out_scalars, self.channel_count(8),relu=False, ksz=1)
            out_scalars = self.conv(pfx + 'alpha_bottleneck_6', out_scalars, self.channel_count(4),relu=False, ksz=1)# 1, 4, 4, 1
            out_scalars = self.conv(pfx + 'alpha_bottleneck_7', out_scalars, 1,relu=False, activation_name=pfx + 'bottleneck', ksz=1) #1x4x4x1
            self.scalar_alphas = tf.reshape(out_scalars,(1,16)) #16 basis functions
            self.alphas = [self.scalar_alphas_options[0] * self.alphas[...,:3], self.scalar_alphas_options[1] * self.alphas[...,3:6],   self.scalar_alphas_options[2] * self.alphas[...,6:9],    self.scalar_alphas_options[3] * self.alphas[...,9:12]]
            self.betas =  [1,                                                   self.scalar_betas_options[1]  * self.alphas[...,12:15], self.scalar_betas_options[2]  * self.alphas[...,15:18],  self.scalar_betas_options[3]  * self.alphas[...,18:21]]
            self.sigmas = [1,                                                   self.scalar_sigmas_options[1] * self.alphas[...,21:24], self.scalar_sigmas_options[2] * self.alphas[...,24:27],  self.scalar_sigmas_options[3] * self.alphas[...,27:30]]
        elif(self.llf_remap_function == 'gaussian_1d'):
            basis_weights_size      = self.max_discrete_levels * self.basis_ct * self.rbf_weights_ct * self.img_ct * 2
            range_weights_size      = self.max_discrete_levels * self.img_ct
            total_weights_size      = basis_weights_size + range_weights_size

            #encode range and basis weights
            range_basis_weights, _  = self.down_block(bottleneck,  self.channel_count(512), pfx + 'range_basis_down5')#1,8,8,64
            range_basis_weights, _  = self.down_block(range_basis_weights, self.channel_count(256), pfx + 'range_basis_down6') #1,4,4,32
            range_basis_weights, _  = self.down_block(range_basis_weights, self.channel_count(128), pfx + 'range_basis_down7') #1,2,2,16
            range_basis_weights     = self.conv(pfx + 'range_basis_bottleneck_1', range_basis_weights, total_weights_size, relu=False, ksz=1) #1,2,2,16
            if(self.llf_remap_function_type == 'fixed_top_layer'):
                range_basis_weights     = self.conv(pfx + 'range_basis_bottleneck_2', range_basis_weights, total_weights_size, relu=False, ksz=1) #1,2,2,16
            else:
                range_basis_weights     = self.conv(pfx + 'range_basis_bottleneck_2', range_basis_weights, total_weights_size, relu=False, softplus=True, ksz=1) #1,2,2,16
            range_basis_weights     = tf.reduce_sum(range_basis_weights,axis=(1,2))
            basis_weights           = range_basis_weights[:,:basis_weights_size]
            

            if(self.llf_remap_function_type == 'fixed_top_layer'):
                #b,2,I,K
                basis_weights           = tf.reshape(basis_weights, (1, 2, self.img_ct, self.max_discrete_levels * self.basis_ct * self.rbf_weights_ct))

                experiment=False
                if(experiment):
                    self.w_i = tf.ones((self.img_ct, self.max_discrete_levels)) * -1
                    self.sigma_i = tf.ones((self.img_ct, self.max_discrete_levels)) * 5.5
                    self.range_weights = tf.ones((1, self.img_ct, self.max_discrete_levels))
                else:
                    #I,K
                    w_i_amb = tf.ones((1, self.max_discrete_levels)) * -1 #-1
                    w_i_flash = tf.nn.sigmoid(basis_weights[0,0,1:2,:self.max_discrete_levels]) * (1 + 2.3) -1 #-1 is smoothing, 2.3 is increasing details but still keeping the curve differentiable
                    
                    sigma_amb = 0.1 + tf.nn.sigmoid(basis_weights[0,1,0:1,:self.max_discrete_levels]) * 10 #~10 if low noise, ~0.1 for high noise
                    sigma_flash = tf.nn.sigmoid(basis_weights[0,1,0:1,:self.max_discrete_levels]) * 10 # can vary
                
                    self.w_i                = tf.concat([w_i_amb, w_i_flash], axis=0)
                    #I,K
                    self.sigma_i            = tf.concat([sigma_amb, sigma_flash], axis=0)
                    
                    #range weight: k * i
                    #interpolates different ranges
                    range_weights           = tf.nn.sigmoid(range_basis_weights[:,basis_weights_size:]) * 2
                    self.range_weights      = tf.reshape(range_weights,(1,self.img_ct, self.max_discrete_levels))
            else:
                #b,2,I,K
                basis_weights           = tf.reshape(basis_weights, (1, 2, self.img_ct, self.max_discrete_levels * self.basis_ct * self.rbf_weights_ct))
                #I,K
                self.w_i                = (2 * tf.nn.sigmoid(basis_weights[0,0,:,:self.max_discrete_levels]) - 1) * self.gaussian_weights_scale
                #I,K
                self.sigma_i            = basis_weights[0,1,:,:self.max_discrete_levels] ** 2 + self.gaussian_sigma_offset

                #range weight: k * i
                #interpolates different ranges
                range_weights           = range_basis_weights[:,basis_weights_size:]
                self.range_weights      = tf.reshape(range_weights,(1,self.img_ct, self.max_discrete_levels))


            # #I,K
            # self.w_i                = (2 * tf.nn.sigmoid(basis_weights[0,0,:,:self.max_discrete_levels]) - 1) * self.gaussian_weights_scale
            # #I,K
            # self.sigma_i            = basis_weights[0,1,:,:self.max_discrete_levels] ** 2 + self.gaussian_sigma_offset




            #create a small unet here
            #if downsample = 0 the unet is just a decoder
            #self.skip_gllf.d1 = self.skip.d1 ...
            #if downsample = 3 the unet is an encoder and a decoder
                #self.encode
                #self.decode
                #
            #decode gllf local interpolation
            # for i in range(self.downsample_ct):
            
            # 128
            # self.image_weights = []
            # for i in range(self.max_levels):
            #     spatial_weights = bottleneck
            #     for j in range(i):
            #         spatial_weights, _  = self.down_block(spatial_weights,  self.channel_count(1024), pfx + 'image_basis_down%i_%i' % (i,j))
            #     #convolve, store and downsample
            #     image_weights    = self.conv(pfx + 'spatial_bottleneck_%i_1' % i, spatial_weights, self.channel_count(512//(2**i)), relu=False, ksz=1)
            #     image_weights    = self.conv(pfx + 'spatial_bottleneck_%i_2' % i, image_weights, self.channel_count(512//(2**(i+1))), relu=False, ksz=1)
            #     image_weights    = self.conv(pfx + 'spatial_bottleneck_%i_3' % i, image_weights, self.channel_count(512//(2**(i+2))), relu=False, ksz=1)
            #     image_weights    = self.conv(pfx + 'spatial_bottleneck_%i_4' % i, image_weights, self.img_ct * 3, relu=False, ksz=1)
            #     image_weights    = self.conv(pfx + 'spatial_bottleneck_%i_5' % i, image_weights, self.img_ct * 3, relu=False, ksz=1)
            #     image_weights    = tf.reshape(image_weights,(*image_weights.shape[:3], self.img_ct, 3))+0.5
            #     self.image_weights.append(image_weights)

            return self.w_i, self.sigma_i
        elif(self.llf_remap_function == 'manual'):
            #range weight: k * i
            #1, img_ct, k
            self.range_weights = tf.ones((1,self.img_ct, self.max_discrete_levels))

            #interpolates different ranges
            #spatial image weight: {I},{L},s_h,s_w
            #interpolates the laplacian pyramids of different images locally
            self.image_weights = []
            for i in range(self.max_levels):
                #convolve, store and downsample
                #L, 1, h, w, img_ct, c
                self.image_weights.append(tf.ones((1,2,2,self.img_ct,3)))
            self.image_weights = tf.stack(self.image_weights, axis=0)
            # img_ct, k
            self.w_i                = tf.convert_to_tensor([[1.] * self.max_levels] * self.img_ct,dtype=tf.float32)
            self.sigma_i            = tf.convert_to_tensor([[1.] * self.max_levels] * self.img_ct,dtype=tf.float32)

        else:
            print('Remapping function ', self.llf_remap_function, ' is not implemented')
            exit(0)

        

            # #range weight: k * i
            # #interpolates different ranges
            # self.range_weights      = tf.reshape(range_weights,(1,self.img_ct, self.max_discrete_levels))
   
    
    @tf.custom_gradient
    def diffable_slice_separable_expansion(self, l_i, lpyramids):
        #lpyramids 1, I, L, h, w, c
        #l_i       , I, h, w, c
        #fetch i and g pixels and discretize
        # assert min_intensity < max_intensity
        l_i = (l_i - self.min_intensity) / (self.max_intensity - self.min_intensity)
        max_discrete_levels_ft = tf.cast(self.max_discrete_levels,tf.float32)
        l_i_is = tf.clip_by_value(tf.cast(l_i * (max_discrete_levels_ft - 1), tf.int32), 0, self.max_discrete_levels-2)
        l_i_fs = l_i * (max_discrete_levels_ft - 1) - tf.cast(l_i_is, dtype=tf.float32)
        
        l_i_i_0_l_g_i_0s = self.inner_slice_1d(l_i_is,     lpyramids)
        l_i_i_1_l_g_i_0s = self.inner_slice_1d(l_i_is + 1, lpyramids)
        
        # Sum up laplacian pyramids
        outLPyramids = (1 - l_i_fs) * l_i_i_0_l_g_i_0s + l_i_fs * l_i_i_1_l_g_i_0s #1, I, h, w, c
        # self.range_weights      = interpollation_weights[:,self.img_ct:]
        # Define the custom gradient
        def grad_fn(dy):
            # make laplacian pyramid by interpolation
            d_i = (  - 1) * l_i_i_0_l_g_i_0s + (    1) * l_i_i_1_l_g_i_0s
            l_i_i_0_l_g_i_0_ones = (1  - l_i_fs[:,:,None,...]) * self.set_inner_slice_1d(l_i_is,       lpyramids)
            l_i_i_1_l_g_i_0_ones = (     l_i_fs[:,:,None,...]) * self.set_inner_slice_1d(l_i_is+1,     lpyramids)
            dy_d_l = l_i_i_0_l_g_i_0_ones + l_i_i_1_l_g_i_0_ones
            return dy * d_i, dy[:,:,None,...] * dy_d_l

        return outLPyramids, grad_fn

    def apply_image_weights_to_pyramid(self, outLPyramids, image_weights):
        # self.image_weights      = interpollation_weights[:,:self.img_ct]
            # Extract blocks from the padded image
        #b,I,h,w,3 -> bI,h,w,3
        b_ct, i_ct, p_h, p_w, c = outLPyramids.shape
        interim_pyramid = tf.einsum('bIhwc->bIchw',outLPyramids)
        interim_pyramid = tf.reshape(interim_pyramid,(b_ct * i_ct * c, p_h, p_w))
        _, s_h, s_w,_,_ = image_weights.shape
        b_h, b_w = p_h//s_h, p_w//s_w
        #bIc,h,w,3 -> bIc,b_h,b_w,h//b_h*w//b_w,3
        blocks = tf.image.extract_patches(
            images=interim_pyramid[...,None],
            sizes=[1, b_h, b_w, 1],
            strides=[1, b_h, b_w, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        blocks = tf.reshape(blocks,(b_ct * i_ct, c, *blocks.shape[1:]))
        

        image_weights_interim = tf.einsum('bhwIc->bIchw',image_weights)
        image_weights_interim = tf.reshape(image_weights_interim,(b_ct*self.img_ct,self.per_layer_decoder_nchannels,*image_weights_interim.shape[3:]))
        interim_blocks = blocks * image_weights_interim[...,None]
        interim_blocks = tf.reshape(interim_blocks,(b_ct,i_ct,3,s_h, s_w,b_h,b_w))
        interim_blocks = tf.einsum('bIchwHW->bIhHwWc',interim_blocks)
        interim_blocks = tf.reshape(interim_blocks,(b_ct, i_ct, s_h*b_h, s_w*b_w, 3))
        outLPyramids = interim_blocks
        return outLPyramids
    
    def diffable_slice_separable(self, l_i, lpyramids, image_weights):
        outLPyramids = self.diffable_slice_separable_expansion(l_i, lpyramids * self.range_weights[...,None,None,None]) #1, I, h, w, c
        if(self.llf_remap_function == 'gaussian_1d' or self.llf_remap_function == 'exp_1d'):
            outLPyramids = self.apply_image_weights_to_pyramid(outLPyramids, image_weights)
        return tf.reduce_sum(outLPyramids,axis=1) #1, h, w, c
    
    def gllf_diffable_1d(self, im_is, debug_reconstruct_all_remapping_images=False):
        # return self.remapping_1d(im_is, 0, 0)
        outLPyramids = self.images_to_lookup_1d_noresize(im_is) #L,1,n,k,h,w,c
        G_is = [[[] for _ in range(len(im_is))] for _ in range(self.max_levels)]
        for im_idx in range(len(im_is)):
            im_i_pyramid = im_is[im_idx]
        # for im_idx, im_i_pyramid in enumerate(im_is):
            G_i = GaussianPyramid(im_i_pyramid, self.max_levels)
            for i, g in enumerate(G_i):
                G_is[i][im_idx] = g
        G_is = [tf.stack(i,axis=0) for i in G_is] # {L}, I, 1, h, w, c
        G_is = [tf.transpose(i,(1,0,2,3,4)) for i in G_is] # {L}, 1, I, h, w, c
        outLPyramid = []
        if(debug_reconstruct_all_remapping_images):
            outLPyramid_range = []
            outLPyramid_image = []
            for i in range(self.max_levels):
                #h_0 = 448
                #h_1 = 224
                #h_2 = 112
                #h_3 = 56
                #outLPyramids pyramid_level_ct, 1, I, intensity_level_ct, h_level, w_level, c
                #G_is         pyramid_level, 1, I, h_level, w_level, c
                outLPyramid_slice_with_range_weight = self.diffable_slice_separable_expansion(G_is[i], outLPyramids[i] * self.range_weights[...,None,None,None])
                if(self.llf_remap_function == 'gaussian_1d' or self.llf_remap_function == 'exp_1d'):
                    outLPyramid_slice_with_image_weight = self.apply_image_weights_to_pyramid(outLPyramid_slice_with_range_weight, self.image_weights[i])
                    outLPyramid_image.append(outLPyramid_slice_with_image_weight)
                outLPyramid_range.append(outLPyramid_slice_with_range_weight)
            #pack individual pyramids
            intensity_level_images = []
            for i in range(self.img_ct):
                img_i_levels_j = []
                for j in range(self.max_discrete_levels):
                    levels_j = []
                    for k in range(self.max_levels):
                        levels_j.append(outLPyramids[k][:,i,j,...])
                    img_i_levels_j.append(reconstruct_Laplacian(levels_j, self.max_levels))
                intensity_level_images.append(img_i_levels_j)



            intensity_images_range = []
            intensity_images_image = []
            for i in range(self.img_ct):
                img_levels_range = []
                img_levels_image = []
                for j in range(self.max_levels):
                    img_levels_range.append(outLPyramid_range[j][:,i])
                    if(self.llf_remap_function == 'gaussian_1d' or self.llf_remap_function == 'exp_1d'):
                        img_levels_image.append(outLPyramid_image[j][:,i])
                intensity_images_range.append(reconstruct_Laplacian(img_levels_range, self.max_levels))
                if(self.llf_remap_function == 'gaussian_1d' or self.llf_remap_function == 'exp_1d'):
                    intensity_images_image.append(reconstruct_Laplacian(img_levels_image, self.max_levels))
            # for i in range(self.max_levels):
            #     outLPyramid_slice = self.diffable_slice_separable(G_is[i], outLPyramids[i], self.image_weights[i]) #1, h, w, c
            #     outLPyramid.append(outLPyramid_slice)
            return intensity_images_range, intensity_images_image, intensity_level_images
            # return reconstruct_Laplacian(outLPyramid, self.max_levels), None
        else:
            for i in range(self.max_levels):
                #h_0 = 448
                #h_1 = 224
                #h_2 = 112
                #h_3 = 56
                #outLPyramids L, 1, I, k, h_level, w_level, c
                #G_is         L, 1, I, h_level, w_level, c
                outLPyramid_slice = self.diffable_slice_separable(G_is[i], outLPyramids[i], self.image_weights[i]) #1, h, w, c
                outLPyramid.append(outLPyramid_slice)
            return reconstruct_Laplacian(outLPyramid, self.max_levels)
        
    def llf_remap(self, i, level, sigma, beta, alpha):
        diff = i - level
        # if(threshold is None):
        return sigma * level + beta * diff + self.remap(diff, alpha)
        # else:
        #     # compress = sigma * level + beta * diff
        #     compress = sigma * level + tf.sign(diff) * (beta * (tf.abs(diff)-threshold)+threshold)
        #     # compress = sigma * level + tf.sign(diff) * threshold * tf.pow(tf.abs(diff)/threshold, 1/alpha)
        #     details = sigma * level + tf.sign(diff) * threshold * tf.pow(tf.abs(diff)/threshold, alpha)
        #     # details = sigma * level + remap(diff, alpha)
        #     return tf.where(tf.abs(diff) < threshold, details, compress)

    def remapping_1d(self, im_is, im_idx, k):
        
        # threshold = None if self.thresholds is None else self.thresholds[im_idx]
        level = k / (self.max_discrete_levels - 1)
        level = level * (self.max_intensity - self.min_intensity) + self.min_intensity
        if(self.llf_remap_function == 'exp_1d' or self.llf_remap_function == 'manual_1d'):
            i, alpha, beta, sigma = im_is[im_idx], self.alphas[im_idx][k], self.betas[im_idx][k], self.sigmas[im_idx][k]
            return self.llf_remap(i, level, sigma, beta, alpha)
        elif(self.llf_remap_function == 'exp_2d'):
            pass
        elif(self.llf_remap_function == 'gaussian_1d'):
            # w_i =     self.w_i[0, 0, k, :, im_idx][:,None] #weights_ct, 1
            # sigma_i = self.sigma_i[0, 0, k, :, im_idx][:,None] #weights_ct, 1
            w_i =     self.w_i[im_idx,k:k+1][:,None] #weights_ct, 1
            sigma_i = self.sigma_i[im_idx,k:k+1][:,None] #weights_ct, 1
            i = im_is[im_idx]
            return level + radial_basis(i, level, w_i, sigma_i, self.llf_remap_function)
        elif(self.llf_remap_function == 'piecewise_linear'):
            w_i =     self.w_i[0, 0, k, :, im_idx][:,None] #weights_ct, 1
            sigma_i = self.sigma_i[0, 0, k, :, im_idx][:,None] #weights_ct, 1
            i = im_is[im_idx]
            return level + radial_basis(i, level, w_i, sigma_i, self.llf_remap_function)
        elif(self.llf_remap_function == 'gaussian_piecewise'):
            w_i =     self.w_i[0, 0, k, :, im_idx][:,None] #weights_ct, 1
            sigma_i = self.sigma_i[0, 0, k, :, im_idx][:,None] #weights_ct, 1
            i = im_is[im_idx]
            return level + radial_basis(i, level, w_i, sigma_i, self.llf_remap_function)
        elif(self.llf_remap_function == 'manual'):
            debug_basis = 'gaussian_1d'
            if(debug_basis == 'gaussian_1d'):
                w_i =     tf.convert_to_tensor([-1])[:,None] #weights_ct, 1
                sigma_i = tf.convert_to_tensor([5])[:,None] #weights_ct, 1
            else:
                w_i =     tf.convert_to_tensor([1./2.])[:,None] #weights_ct, 1
                sigma_i = tf.convert_to_tensor([0.2])[:,None] #weights_ct, 1

            w_i = tf.convert_to_tensor([self.w_i[im_idx,k]])
            sigma_i = tf.convert_to_tensor([self.sigma_i[im_idx,k]])
            i = im_is[im_idx]
            return level + radial_basis(i, level, w_i, sigma_i, debug_basis)


    def remapping_sampler(self, im_idx, intensity_level, start=0., stop=1., num=1000):
        i = tf.linspace(start, stop, num)
        return self.remapping_1d([i, i, i, i], im_idx, intensity_level)
    
    def visualize(self, im_is):
        x = np.linspace(0, 1, 1000)
        visualizations = []
        reconstructed_images_range, reconstructed_images_image, intensity_level_images = self.gllf_diffable_1d(im_is, debug_reconstruct_all_remapping_images=True)
        for j in range(self.img_ct):
            for i in range(self.max_discrete_levels):
                y = self.remapping_sampler(j, i, start=0., stop=1., num=1000)
                # ,xlim=[-1.2,1.2],ylim=[-1.2,1.2]
                visualizations += [edict(image=viz.plot(x, y) / 255., label='$image=%i, \gamma=%i$'% (j,i), key='remapping_%i_%i' % (i, j))]
                visualizations += [edict(image=intensity_level_images[j][i], label='$image=%i, \gamma=%i$'% (j,i), key='intensity_level_%i_%i' % (i, j))]
            visualizations += [edict(image=reconstructed_images_range[j], label='$image=%i, reconstructed w range$'% (j), key='reconstructed_w_range%i' % (j))]
            if(self.llf_remap_function == 'gaussian_1d' or self.llf_remap_function == 'exp_1d'):
                visualizations += [edict(image=reconstructed_images_image[j], label='$image=%i, reconstructed w range+image$'% (j), key='reconstructed w range+image %i' % (j))]
        return visualizations


    
    def images_to_lookup_1d_noresize(self, im_is, IMSZ=448):
        #in this function
        #K is intensity sample count max_discrete_levels
        #L is level count max_levels
        # compute remapped images and its pyramids
        if(self.thresholds is None):
            thresholds = [None] * len(im_is)
        lpyramid = [[[] for _ in range(len(im_is))] for _ in range(self.max_levels)]
        for k_i in range(self.max_discrete_levels):
            for im_idx in range(len(im_is)):
                # im_i, alpha, beta, sigma, threshold = im_is[im_idx], self.alphas[im_idx], self.betas[im_idx], self.sigmas[im_idx], thresholds[im_idx]
                # r_i_j = self.remapping_1d(im_i, k_i, sigma, beta, alpha, threshold)
                r_i_j = self.remapping_1d(im_is, im_idx, k_i)
                f_i_j_g = GaussianPyramid(r_i_j, self.max_levels)
                f_i_j_l = LaplacianPyramid(f_i_j_g)# K, 1, h, w, c
                for f_i_j_l_i, f_i_j_l_v in enumerate(f_i_j_l):
                    lpyramid[f_i_j_l_i][im_idx].append(f_i_j_l_v) #{L}, I, K, 1, h, w, c
        
        lpyramid = [tf.stack(lpyramid[i],axis=0) for i in range(self.max_levels)]#{L}, I, K, 1, h, w, c
        lpyramid = [tf.transpose(i,(2,0,1,3,4,5)) for i in lpyramid]#{L}, 1, I, K, h, w, c
        return lpyramid

    
    def inner_slice_1d(self, l_i_i, l):
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
    
    def set_inner_slice_1d(self,l_i_i, l):
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
        