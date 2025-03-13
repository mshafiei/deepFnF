from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu
from tiny_unet import Net as tiny_unet
from easydict import EasyDict as edict
class Net(tiny_unet):
    def __init__(self, downsample_ct=0, unet_output_size=3, channels_count_factor=1, use_up_block=True, per_layer_decoder_nchannels=3, has_image_weights=False,llf_remap_function_type="None"):
        super().__init__(unet_output_size=unet_output_size, channels_count_factor=channels_count_factor)
        self.downsample_ct = downsample_ct
        self.channel_count = lambda x: max(1, int(x * self.channels_count_factor)//(2**self.downsample_ct))
        self.channel_count_end = lambda x: max(self.output_dim_size, int(x * self.channels_count_factor)//(2**self.downsample_ct))
        self.use_up_block = use_up_block
        self.per_layer_decoder_nchannels = per_layer_decoder_nchannels
        self.has_image_weights = has_image_weights
        self.llf_remap_function_type = llf_remap_function_type


    def encode(self, out, pfx=''):
        skips = edict()
        
        out = self.conv(pfx + 'inp', out, self.channel_count(64))
        for i in range(self.downsample_ct):
            out = self.conv(pfx + 'inp_%i'%i, out, self.channel_count(64*2**(i+1)))
            

        # if(self.downsample_ct == 0):
        out, skips.d1 = self.down_block(out, self.channel_count(64  ), pfx + 'down1')
        # if(self.downsample_ct <= 1):
        out, skips.d2 = self.down_block(out, self.channel_count(128 ), pfx + 'down2')
        # if(self.downsample_ct <= 2):
        out, skips.d3 = self.down_block(out, self.channel_count(256 ), pfx + 'down3')
        # if(self.downsample_ct <= 3):
        out, skips.d4 = self.down_block(out, self.channel_count(512 ), pfx + 'down4')
        out, skips.d5 = self.down_block(out, self.channel_count(1024), pfx + 'down5')

        out = self.conv(pfx + 'bottleneck_1', out, self.channel_count(1024))
        out = self.conv(pfx + 'bottleneck_2', out, self.channel_count(1024),
                        activation_name=pfx + 'bottleneck')
        self.bottleneck = out
        self.skips = skips
        return out, skips
    
    def up_and_out(self, out, nch, skip, is_out_layer, img_ct, pfx=''):
        nchannels = img_ct * self.per_layer_decoder_nchannels
        out_layer = out
        out = self.up_block(out, nch, skip, pfx)
        if(is_out_layer):
            for i in range(int(np.log2(nch) - np.ceil(np.log2(nchannels)))):
                out_layer = self.conv(pfx + '_end_%i'%(i+1), out_layer, self.channel_count_end(512)//(2**i),relu=False)
            out_layer = self.conv(pfx + '_end_last', out_layer, img_ct * self.per_layer_decoder_nchannels,relu=False)
            out_layer = tf.reshape(out_layer, (*out_layer.shape[:-1], img_ct, self.per_layer_decoder_nchannels))
            return out, out_layer
        else:
            return out, None

    def decode_per_layer(self, out, skips, max_levels=5, pfx=''):
        decode_layers = edict()

    
        # for i in range(5):
        #     out, out_layer = self.up_and_out(out, self.channel_count(512//i), skips[], self.start_level == 0 and self.max_levels > 1, pfx + 'up1')
        if(self.has_image_weights):
            out, skip1 = self.down_block(out, self.channel_count(512), pfx + 'down1')
            out, skip2 = self.down_block(out, self.channel_count(1024), pfx + 'down2')
            if(self.llf_remap_function_type == 'fixed_top_layer'):
                zero = tf.zeros((*out.shape[:-1], 1, self.per_layer_decoder_nchannels))
                one = tf.ones((*out.shape[:-1], self.img_ct - 1, self.per_layer_decoder_nchannels))
                decode_layers.d1 = tf.concat((one, zero), axis=-2)
            else:
                out, decode_layers.d1 = self.up_and_out(out, self.channel_count(512), skip2, True, 1, pfx + 'up1')
            out, decode_layers.d2 = self.up_and_out(out, self.channel_count(256), skip1, True, 1,pfx + 'up2')
            out, decode_layers.d3 = self.up_and_out(out, self.channel_count(128), skips.d5, True, 1, pfx + 'up3')
            _, decode_layers.d4 = self.up_and_out(out, self.channel_count(64), skips.d4, True, 1,pfx + 'up4')
            decode_layers.d2 = tf.concat((tf.ones_like(decode_layers.d2), decode_layers.d2),axis=-2)
            decode_layers.d3 = tf.concat((tf.ones_like(decode_layers.d3), decode_layers.d3),axis=-2)
            decode_layers.d4 = tf.concat((tf.ones_like(decode_layers.d4), decode_layers.d4),axis=-2)
        else:
            decode_layers.d1 = tf.concat((tf.ones((1,2,2,  1,1)),tf.ones((1,2,2,  1,1))), axis=-2)#self.up_and_out(out, self.channel_count(512), skips.d5, True, pfx + 'up1')
            decode_layers.d2 = tf.concat((tf.ones((1,4,4,  1,1)),tf.ones((1,4,4,  1,1))), axis=-2)#self.up_and_out(out, self.channel_count(256), skips.d4, True, pfx + 'up2')
            decode_layers.d3 = tf.concat((tf.ones((1,7,7,  1,1)),tf.ones((1,7,7,  1,1))), axis=-2)#self.up_and_out(out, self.channel_count(128), skips.d3, True, pfx + 'up3')
            decode_layers.d4 = tf.concat((tf.ones((1,14,14,1,1)),tf.ones((1,14,14,1,1))), axis=-2)#self.up_and_out(out, self.channel_count(64), skips.d2, True, pfx + 'up4')
        
        return decode_layers

    def joint_decode(self, out, skips, pfx=''):
        decode_layers = edict()
        if(self.use_up_block):
            out = self.up_block(out, self.channel_count(512), skips.d5, pfx + 'up1')
            # if(self.downsample_ct <= 3):
            out = self.up_block(out, self.channel_count(256), skips.d4, pfx + 'up2')
            # decode_layers.d1 = out[...,-8:]
            # decode_layers.d1 = self.conv(pfx + 'd1_end_last', decode_layers.d1, self.img_ct * self.per_layer_decoder_nchannels, relu=False)
            # decode_layers.d1 = tf.reshape(decode_layers.d1, (*decode_layers.d1.shape[:-1], self.img_ct, self.per_layer_decoder_nchannels))
            # if(self.downsample_ct <= 2):
            out = self.up_block(out, self.channel_count(128), skips.d3, pfx + 'up3')
            # decode_layers.d2 = out[...,-8:]
            # decode_layers.d2 = self.conv(pfx + 'd2_end_last', decode_layers.d2, self.img_ct * self.per_layer_decoder_nchannels, relu=False)
            # decode_layers.d2 = tf.reshape(decode_layers.d2, (*decode_layers.d2.shape[:-1], self.img_ct, self.per_layer_decoder_nchannels))
            # if(self.downsample_ct <= 1):
            out = self.up_block(out, self.channel_count(64 ), skips.d2, pfx + 'up4')
            # decode_layers.d3 =  out[...,-8:]
            # decode_layers.d3 = self.conv(pfx + 'd3_end_last', decode_layers.d3, self.img_ct * self.per_layer_decoder_nchannels, relu=False)
            # decode_layers.d3 = tf.reshape(decode_layers.d3, (*decode_layers.d3.shape[:-1], self.img_ct, self.per_layer_decoder_nchannels))
            # if(self.downsample_ct <= 0):
            out = self.up_block(out, self.channel_count(64 ), skips.d1, pfx + 'up5')
            # decode_layers.d4 = out[...,-8:]
            # decode_layers.d4 = self.conv(pfx + 'd4_end_last', decode_layers.d4, self.img_ct * self.per_layer_decoder_nchannels, relu=False)
            # decode_layers.d4 = tf.reshape(decode_layers.d4, (*decode_layers.d4.shape[:-1], self.img_ct, self.per_layer_decoder_nchannels))

        out = out[...,:-8]
        for i in range(int(np.log2(self.channel_count(64 )) - np.ceil(np.log2(3)))):
            out = self.conv(pfx + 'end_%i'%(i), out, self.channel_count_end(64)//(2**i), relu=False)

        out = self.conv(pfx + 'end_%i'%(i+1), out, self.output_dim_size,relu=False)
        out = self.conv(pfx + 'end_%i'%(i+2), out, self.output_dim_size,relu=False)
        out = self.conv(pfx + 'end_%i'%(i+3), out, self.output_dim_size,relu=False, activation_name=pfx + 'end')

        # out = self.conv(pfx + 'end_1', out, self.channel_count_end(64))
        # out = self.conv(pfx + 'end_2', out, self.channel_count_end(64), activation_name=pfx + 'end')

        return out, decode_layers
    
    def decode(self, out, skips, pfx=''):
        if(self.use_up_block):
            out = self.up_block(out, self.channel_count(512), skips.d5, pfx + 'up1')
            # if(self.downsample_ct <= 3):
            out = self.up_block(out, self.channel_count(256), skips.d4, pfx + 'up2')
            # if(self.downsample_ct <= 2):
            out = self.up_block(out, self.channel_count(128), skips.d3, pfx + 'up3')
            # if(self.downsample_ct <= 1):
            out = self.up_block(out, self.channel_count(64 ), skips.d2, pfx + 'up4')
            # if(self.downsample_ct <= 0):
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

    def downsample(self, inp):
        _, h, w, _ = inp.shape
        #downsample
        return tf.image.resize(inp,(h//(2**self.downsample_ct), w//(2**self.downsample_ct)))
    
    def upsample(self, inp, h, w):
        #upsample
        return tf.image.resize(inp,(h, w))
        
    def resize_encode(self, inp):
        if(self.downsample_ct != 0):
            inp = self.downsample(inp)

        out, skips = self.encode(inp)
        return out, skips

    def resize_decode(self, inp, skips, h, w):
        out = self.decode(inp, skips)
        if(self.downsample_ct != 0):
            #upsample
            return self.upsample(out, h, w)
        else:
            return out
    
    def resize_joint_decode(self, inp, skips, h, w):
        out, decode_layers = self.joint_decode(inp, skips)
        if(self.downsample_ct != 0):
            #upsample
            return self.upsample(out, h, w), decode_layers
        else:
            return out, decode_layers

    def lowres_unet(self, inp):
        _, h, w, _ = inp.shape
        out, skips = self.resize_encode(inp)
        return self.resize_decode(out, skips, h, w)
