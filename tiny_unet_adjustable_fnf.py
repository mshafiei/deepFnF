from tiny_unet_adjustable import Net as tiny_unet
from easydict import EasyDict as edict
class Net(tiny_unet):
    def __init__(self, downsample_ct, unet_output_size=3, channels_count_factor=1):
        super().__init__(downsample_ct=downsample_ct, unet_output_size=unet_output_size, channels_count_factor=channels_count_factor)

    def forward(self, inp):
        return edict(output=super().lowres_unet(inp.net_ft_input))
