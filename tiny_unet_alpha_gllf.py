from easydict import EasyDict as edict
from gllf.gllf_layer import *
# UNet predicts w_i, sigma_i, alpha, etc.
# Run GLLF

class Net(gllf_layer_radial):
    def __init__(self, downsample_ct, input_images, unet_output_size=3, channels_count_factor=1, **kwargs):
        super().__init__(downsample_ct=downsample_ct, unet_output_size=unet_output_size, channels_count_factor=channels_count_factor,**kwargs)
        self.input_images = input_images
        self.img_ct = len(self.input_images)

    def visualize(self, inpt):
        return self.forward(inpt, visualize=True)

    def forward(self, inp, visualize=False):
        alpha = inp.alpha
        color_matrix = inp.color_matrix
        adapt_matrix = inp.adapt_matrix
        inp = inp.net_ft_input
        self.resize_encode(inp)

        ambient_scaled = tfu.camera_to_rgb(
            inp[:, :, :, :3] / alpha, color_matrix, adapt_matrix, do_gamma_correct=True)
        flash_scaled = tfu.camera_to_rgb(
            inp[:, :, :, 3:6], color_matrix, adapt_matrix, do_gamma_correct=True)
        
        source_images = []
        if('noisy_ambient' in self.input_images):
            source_images.append(ambient_scaled)
        
        if('noisy_flash' in self.input_images):
            source_images.append(flash_scaled)

        if(visualize):
            visualization = self.gllf(source_images, self.bottleneck, reconstruct_gllf_pyramids=visualize)
            # for i in range(len(visualization)):
            #     visualization[i].image = tfu.gamma_correct(visualization[i].image)
            return visualization
        else:
            output=edict()
            output.output = self.gllf(source_images, self.bottleneck, reconstruct_gllf_pyramids=visualize)
            # output.output = tfu.gamma_correct(output.output)
            return output
