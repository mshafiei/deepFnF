from tiny_unet_adjustable import Net as tiny_unet
from easydict import EasyDict as edict
import tensorflow as tf
import utils.tf_utils as tfu
import numpy as np

class Net(tiny_unet):
    def __init__(self, downsample_ct, unet_output_size=3, channels_count_factor=1,**kwargs):
        super().__init__(downsample_ct=downsample_ct, unet_output_size=unet_output_size, channels_count_factor=channels_count_factor,**kwargs)

    def visualize(self, inp):
        model_visualization = self.forward(inp, visualize=True)
        images, lbls = {}, {}
        for model_viz in model_visualization:
            images.update({model_viz.key:model_viz.image})
            lbls.update({model_viz.key:model_viz.label})
        return images, lbls
    
    @tf.function
    def forward(self, inp, visualize=False):
        output = super().lowres_unet(inp.net_ft_input)

        if(visualize):
            ambient_scaled = tfu.camera_to_rgb(
                inp.net_ft_input[:, :, :, :3] / inp.alpha, inp.color_matrix, inp.adapt_matrix, do_gamma_correct=True)
            flash_scaled = tfu.camera_to_rgb(
                inp.net_ft_input[:, :, :, 3:6], inp.color_matrix, inp.adapt_matrix, do_gamma_correct=True)
            output = tfu.camera_to_rgb(
                output / inp.alpha, inp.color_matrix, inp.adapt_matrix, do_gamma_correct=True)
            ambient = tfu.camera_to_rgb(inp.ambient,
                inp.color_matrix, inp.adapt_matrix)

            ambient_scaled = edict(image=ambient_scaled, label='Noisy Ambient', key='noisy_ambient')
            flash_scaled = edict(image=flash_scaled, label='Noisy Flash', key='noisy_flash')
            output = edict(image=output, label='DeepFnF', key='output')
            ambient = edict(image=tfu.gamma_correct(ambient), label='Ambient', key='ambient')
            return [flash_scaled, ambient_scaled, ambient, output]
        else:
            return edict(output=output)
