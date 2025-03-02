from tiny_unet_adjustable import Net as tiny_unet
from easydict import EasyDict as edict
import tensorflow as tf
import utils.tf_utils as tfu
import numpy as np

class Net(tiny_unet):
    def __init__(self, downsample_ct, unet_output_size=3, channels_count_factor=1,**kwargs):
        super().__init__(downsample_ct=downsample_ct, unet_output_size=unet_output_size, channels_count_factor=channels_count_factor,**kwargs)
    
    def encode(self, out, pfx=''):
        skips = edict()
        
        out = self.conv(pfx + 'inp', out, self.channel_count(64))
        for i in range(self.downsample_ct):
            out = self.conv(pfx + 'inp_%i'%i, out, self.channel_count(64*2**(i+1)))
            

        if(self.downsample_ct == 0):
            out, skips.d1 = self.down_block(out, self.channel_count(64  ), pfx + 'down1')
        if(self.downsample_ct <= 1):
            out, skips.d2 = self.down_block(out, self.channel_count(128 ), pfx + 'down2')
        if(self.downsample_ct <= 2):
            out, skips.d3 = self.down_block(out, self.channel_count(256 ), pfx + 'down3')
        if(self.downsample_ct <= 3):
            out, skips.d4 = self.down_block(out, self.channel_count(512 ), pfx + 'down4')
        out, skips.d5 = self.down_block(out, self.channel_count(1024), pfx + 'down5')

        out = self.conv(pfx + 'bottleneck_1', out, self.channel_count(1024))
        out = self.conv(pfx + 'bottleneck_2', out, self.channel_count(1024),
                        activation_name=pfx + 'bottleneck')
        self.bottleneck = out
        self.skips = skips
        return out, skips
    
    def decode(self, out, skips, pfx=''):
        out = self.up_block(out, self.channel_count(512), skips.d5, pfx + 'up1')
        if(self.downsample_ct <= 3):
            out = self.up_block(out, self.channel_count(256), skips.d4, pfx + 'up2')
        if(self.downsample_ct <= 2):
            out = self.up_block(out, self.channel_count(128), skips.d3, pfx + 'up3')
        if(self.downsample_ct <= 1):
            out = self.up_block(out, self.channel_count(64 ), skips.d2, pfx + 'up4')
        if(self.downsample_ct <= 0):
            out = self.up_block(out, self.channel_count(64 ), skips.d1, pfx + 'up5')

        i=0
        for i in range(int(np.log2(self.channel_count(64 )) - np.ceil(np.log2(3)))):
            out = self.conv(pfx + 'end_%i'%(i), out, self.channel_count_end(64)//(2**i), relu=False)

        out = self.conv(pfx + 'end_%i'%(i+1), out, self.output_dim_size,relu=False)
        out = self.conv(pfx + 'end_%i'%(i+2), out, self.output_dim_size,relu=False)
        out = self.conv(pfx + 'end_%i'%(i+3), out, self.output_dim_size,relu=False, activation_name=pfx + 'end')

        # out = self.conv(pfx + 'end_1', out, self.channel_count_end(64))
        # out = self.conv(pfx + 'end_2', out, self.channel_count_end(64), activation_name=pfx + 'end')

        return out
    
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
