from easydict import EasyDict as edict
from gllf.gllf_layer import *
# UNet predicts w_i, sigma_i, alpha, etc.
# Run GLLF

class Net(gllf_layer_radial):
    def __init__(self, downsample_ct, input_images, unet_output_size=3, channels_count_factor=1, **kwargs):
        super().__init__(downsample_ct=downsample_ct, unet_output_size=unet_output_size, channels_count_factor=channels_count_factor,**kwargs)
        self.input_images = input_images
        self.img_ct = len(self.input_images)
        if('noisy_ambient' in self.input_images and (not ('noisy_flash' in self.input_images)) and (not ('deep_denoised' in self.input_images))):
            self.sources_mode = 1
        elif('noisy_ambient' in self.input_images and 'noisy_flash' in self.input_images and not('deep_denoised' in self.input_images)):
            self.sources_mode = 2
        elif('noisy_ambient' in self.input_images and 'noisy_flash' in self.input_images and 'deep_denoised' in self.input_images):
            self.sources_mode = 3
        else:
            print('Could not interprete the input images')
            exit(0)

    def visualize(self, inpt):
        model_visualization = self.forward(inpt, visualize=True)
        images, lbls = {}, {}
        for model_viz in model_visualization:
            images.update({model_viz.key:model_viz.image})
            lbls.update({model_viz.key:model_viz.label})
        return images, lbls

    @tf.function
    def forward(self, inp, visualize=False):
        # alpha = inp.alpha
        # color_matrix = inp.color_matrix
        # adapt_matrix = inp.adapt_matrix
        # ambient = inp.ambient
        # inp = inp.net_ft_input
        
        # if(self.sources_mode == 3):
        _, h, w, _ = inp.net_ft_input.shape
        min_brightness = 5.5
        max_brightness = 25
        brightness_scale = (tf.clip_by_value(1/tf.squeeze(inp.alpha), min_brightness+0.1, max_brightness) - min_brightness) / (max_brightness - min_brightness)
        out, skips = self.resize_encode(inp.net_ft_input)
        direct_denoised = self.resize_decode(out, skips, h, w)
        # direct_denoised, image_weights = self.resize_joint_decode(out, skips, h, w)
        image_weights = self.decode_per_layer(out, skips, brightness_scale, self.max_levels, "decode_per_layer_")
        if(self.llf_remap_function_type == 'no_nn'):
            self.image_weights = tf.stack([image_weights.d4, image_weights.d3, image_weights.d2, image_weights.d1], axis=0)
        else:
            self.image_weights = [image_weights.d4, image_weights.d3, image_weights.d2, image_weights.d1]
        # else:
        #     self.resize_encode(inp.net_ft_input)
        
        #double head neural network
        #one head predicts denoised image
        #another had predicts image

        #Pass the bottleneck to a few convolution layers
        #Upsample the output
        #Pass to GLLF
        
        ambient_scaled = tfu.camera_to_rgb(
            inp.net_ft_input[:, :, :, :3] / inp.alpha, inp.color_matrix, inp.adapt_matrix, do_gamma_correct=False)
        flash_scaled = tfu.camera_to_rgb(
            inp.net_ft_input[:, :, :, 3:6], inp.color_matrix, inp.adapt_matrix, do_gamma_correct=False)

        source_images = []
        if(self.sources_mode >=1):
            source_images.append(ambient_scaled)
        
        if(self.sources_mode >=2):
            source_images.append(flash_scaled)

        if(self.sources_mode >=3):
            direct_denoised = tfu.camera_to_rgb(
                direct_denoised / inp.alpha, inp.color_matrix, inp.adapt_matrix, do_gamma_correct=False)
            source_images.append(direct_denoised)

        if(visualize):
            visualization = self.gllf(source_images, self.bottleneck, brightness_scale, reconstruct_gllf_pyramids=visualize)
            output = self.gllf(source_images, self.bottleneck, brightness_scale, reconstruct_gllf_pyramids=False)
            
            for i in range(len(visualization)):
                visualization[i].image = tfu.gamma_correct(visualization[i].image)
            input_flash = flash_scaled
            input_ambient = ambient_scaled
            ambient = tfu.camera_to_rgb(
                inp.ambient, inp.color_matrix, inp.adapt_matrix, do_gamma_correct=False)
            visualization = [edict(image=tf.ones((100,100,3)), label='Blank', key='blank_0')] + visualization
            visualization = [edict(image=tf.ones((100,100,3)), label='Blank', key='blank_1')] + visualization
            visualization = [edict(image=tfu.gamma_correct(ambient), label='Ambient', key='ambient')] + visualization
            visualization = [edict(image=tfu.gamma_correct(output), label='UNet+GLLF', key='output')] + visualization
            visualization = [edict(image=tfu.gamma_correct(input_ambient), label='Noisy Ambient', key='noisy_ambient')] + visualization
            visualization = [edict(image=tfu.gamma_correct(input_flash), label='Noisy Flash', key='noisy_flash')] + visualization
            
            return visualization
        else:
            output=edict()
            output.output = self.gllf(source_images, self.bottleneck, brightness_scale, reconstruct_gllf_pyramids=visualize)
            output.output = tfu.gamma_correct(output.output)
            return output
