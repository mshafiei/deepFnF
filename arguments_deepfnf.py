import numpy as np
import cvgutils.Viz as Viz
import argparse

def parse_arguments_deepfnf():
    def str_to_bool(value):
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
    parser.add_argument('--training_type', type=str,choices=['jitter','no_jitter'], default='jitter', help='Dataset type')
    parser.add_argument('--TESTPATH', type=str, default='/home/mohammad/Projects/optimizer/DifferentiableSolver/data/testset_nojitter', help='testset path')
    parser.add_argument('--TLIST', type=str, default='data/train.txt', help='Training dataset filename')
    parser.add_argument('--VPATH', type=str, default='data/val.txt', help='Validation dataset')
    parser.add_argument('--weight_dir', type=str, default='wts', help='Weight dir')
    parser.add_argument('--weight_file', type=str, default='model.npz', help='Weight dir')
    parser.add_argument('--visualize_freq', type=int, default=10000000, help='How many iterations before visualization')
    parser.add_argument('--val_freq', type=int, default=10000000, help='How many iterations before visualization')
    parser.add_argument('--save_freq', type=int,default=100000,help='How often save parameters')
    parser.add_argument('--use_gpu', type=str_to_bool,default=True,nargs='?', const=True,help='Use gpu')
    parser.add_argument('--mode', default='train', type=str,choices=['train','test'],help='Should we train or test the model?')
    parser.add_argument('--dump_scalars_freq', type=int, default=50, help='How many iterations before visualization')
    parser.add_argument('--displacement', type=float,default=0,help='Jitter')
    parser.add_argument('--min_scale', type=float,default=1.0,help='Jitter')
    parser.add_argument('--max_scale', type=float,default=1.0,help='Jitter')
    parser.add_argument('--max_rotate', type=float,default=0.,help='Jitter')
    parser.add_argument('--lmbda', type=float,default=1,help='Maximum lambda for initialization')
    parser.add_argument('--l2', type=float,default=0.,help='lpips loss')
    parser.add_argument('--grad', type=float,default=0.,help='lpips loss')
    parser.add_argument('--lpips', type=float,default=0.,help='lpips loss')
    parser.add_argument('--wlpips', type=float,default=0.,help='wlpips loss')
    parser.add_argument('--ksz', type=int, default=15, help='size of kernel')
    parser.add_argument('--alpha_width', type=int, default=28, help='size of kernel')
    parser.add_argument('--alpha_height', type=int, default=28, help='size of kernel')
    parser.add_argument('--llf_alpha', type=float, default=1.0, help='size of kernel')
    parser.add_argument('--llf_beta', type=float, default=1.0, help='size of kernel')
    parser.add_argument('--llf_sigma', type=float, default=1.0, help='size of kernel')
    parser.add_argument('--llf_levels', type=int, default=2, help='size of kernel')
    parser.add_argument('--n_pyramid_levels', type=int, default=5, help='size of kernel')
    parser.add_argument('--channels_count_factor', type=float,default=1.,help='Scale the channel count for DeepFnF network')
    parser.add_argument('--num_basis', type=int,default=90,help='number of basis')
    parser.add_argument('--fft_lmbda_pp', type=float,default=0,help='number of basis')
    parser.add_argument('--bilateral_pp', type=float,default=0,help='number of basis')
    parser.add_argument('--bilateral_spatial', type=float,default=8,help='number of basis')
    parser.add_argument('--bilateral_luma', type=float,default=16,help='number of basis')
    parser.add_argument('--bilateral_smooth', type=float,default=8,help='number of basis')
    parser.add_argument('--bilateral_neighbors', type=float,default=6,help='number of basis')
    parser.add_argument('--deepfnf_train_path', type=str,default='/home/mohammad/Projects/deepfnftf2/logs-grid/deepfnf-tf2-orig/train/',help='number of basis')
    parser.add_argument('--insets_json', type=str,default='',help='number of basis')
    parser.add_argument('--print_val_freq', type=int,default=100,help='number of basis')
    parser.add_argument('--bs_lam', type=float,default=10,help='number of basis')
    parser.add_argument('--latexName', default="N/A", type=str,help='Latex name of the experiment')
    parser.add_argument('--model', type=str,default='deepfnf',choices=['deepfnf_refine_unet', 'deepfnf_llf_alpha_map_unet_v2', 'deepfnf_llf_alpha_map_image_v2', 'deepfnf_llf_scalar_alpha','deepfnf_llf_scalar_alpha_encoder','deepfnf_llf_alpha_map_unet', 'deepfnf_llf_alpha_map_image', 'flash','noisy','unet_llf','deepfnf_combine_laplacian_pixelwise', 'deepfnf_llf', 'deepfnf_llf_diffable','deepfnf','deepfnf_fft','unet', 'deepfnf_grad','deepfnf_combine_fft','deepfnf_combine_laplacian','net_flash_image','deepfnf-slim'],help='Neural network model')
    parser.add_argument('--scalemap', type=str_to_bool,default=True,nargs='?', const=True,help='Use scalemap?')
    parser.add_argument('--separate_images', type=str_to_bool,default=False,nargs='?', const=True,help='export images separately or all together')
    parser.add_argument('--overfit', type=str_to_bool,default=False,nargs='?', const=True,help='export images separately or all together')
    parser.add_argument('--dataset_model', type=str, default='default', choices=['default','filelock','prefetch','prefetch_nthread'], help='Validation dataset')
    parser.add_argument('--sigmoid_offset', type=float,default=0.1,help='Jitter')
    parser.add_argument('--sigmoid_intensity', type=float,default=10,help='Jitter')
    parser.add_argument('--test_set_count', type=int,default=128,help='test_set_count')
    parser.add_argument('--subset_idx', type=int,default=-1,help='test_set_count')
    
    

    parser = Viz.logger.parse_arguments(parser)
    return parser

def parse_arguments_deepfnf_fft():
    def str_to_bool(value):
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--TESTPATH', type=str, default='/home/mohammad/Projects/optimizer/DifferentiableSolver/data/testset_nojitter', help='testset path')
    parser.add_argument('--TLIST', type=str, default='data/train.txt', help='Training dataset filename')
    parser.add_argument('--VPATH', type=str, default='data/val.txt', help='Validation dataset')
    parser.add_argument('--model', type=str, default='deepfnf',choices=['deepfnf_llf', 'unet','deepfnf','deepfnf+fft','deepfnf+fft_highdim','deepfnf+fft_grad_image','deepfnf+fft_helmholz','deepfnf+fft_helmholz_highdim','deepfnf-slim'], help='Validation dataset')
    parser.add_argument('--outchannels', type=int, default=3, help='Number of the output channels of the UNet')
    parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
    parser.add_argument('--weight_dir', type=str, default='wts', help='Weight dir')
    parser.add_argument('--weight_file', type=str, default='wts/', help='Weight dir')
    parser.add_argument('--visualize_freq', type=int, default=10000000, help='How many iterations before visualization')
    parser.add_argument('--val_freq', type=int, default=10000000, help='How many iterations before visualization')
    parser.add_argument('--ksz', type=int, default=15, help='size of kernel')
    parser.add_argument('--lmbda_phi', type=float, default=1., help='The min value of lambda phi')
    parser.add_argument('--lmbda_psi', type=float, default=1., help='The min value of lambda psi')
    parser.add_argument('--fixed_lambda', type=str_to_bool,default=False,nargs='?', const=True,help='Do not change the delta value')
    parser.add_argument('--max_lambda', type=float,default=0.001,help='Maximum lambda for initialization')
    parser.add_argument('--lmbda', type=float,default=1,help='Maximum lambda for initialization')
    parser.add_argument('--save_freq', type=int,default=100000,help='How often save parameters')
    parser.add_argument('--mode', default='train', type=str,choices=['train','test'],help='Should we train or test the model?')

    parser = Viz.logger.parse_arguments(parser)
    return parser

