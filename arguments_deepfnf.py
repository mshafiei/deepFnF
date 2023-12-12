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
    parser.add_argument('--VPATH', type=str, default='data/valset', help='Validation dataset')
    parser.add_argument('--weight_dir', type=str, default='wts', help='Weight dir')
    parser.add_argument('--weight_file', type=str, default='model.npz', help='Weight dir')
    parser.add_argument('--visualize_freq', type=int, default=10000000, help='How many iterations before visualization')
    parser.add_argument('--val_freq', type=int, default=10000000, help='How many iterations before visualization')
    parser.add_argument('--save_freq', type=int,default=100000,help='How often save parameters')
    parser.add_argument('--mode', default='train', type=str,choices=['train','test'],help='Should we train or test the model?')
    parser.add_argument('--dump_scalars_freq', type=int, default=50, help='How many iterations before visualization')
    parser.add_argument('--displacement', type=float,default=0,help='Jitter')
    parser.add_argument('--min_scale', type=float,default=1.0,help='Jitter')
    parser.add_argument('--max_scale', type=float,default=1.0,help='Jitter')
    parser.add_argument('--max_rotate', type=float,default=0.,help='Jitter')
    parser.add_argument('--lpips', type=int,default=0.,help='lpips loss')
    parser.add_argument('--channels_count_factor', type=float,default=1.,help='Scale the channel count for DeepFnF network')
    parser.add_argument('--num_basis', type=int,default=90,help='number of basis')
    parser.add_argument('--model', type=str,default='deepfnf',choices=['deepfnf','unet', 'deepfnf_grad'],help='Neural network model')
    parser.add_argument('--scalemap', type=str_to_bool,default=True,nargs='?', const=True,help='Use scalemap?')

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
    parser.add_argument('--VPATH', type=str, default='data/valset', help='Validation dataset')
    parser.add_argument('--model', type=str, default='deepfnf',choices=['unet','deepfnf','deepfnf+fft','deepfnf+fft_highdim','deepfnf+fft_grad_image','deepfnf+fft_helmholz','deepfnf+fft_helmholz_highdim'], help='Validation dataset')
    parser.add_argument('--outchannels', type=int, default=3, help='Number of the output channels of the UNet')
    parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
    parser.add_argument('--weight_dir', type=str, default='wts', help='Weight dir')
    parser.add_argument('--weight_file', type=str, default='wts/', help='Weight dir')
    parser.add_argument('--visualize_freq', type=int, default=10000000, help='How many iterations before visualization')
    parser.add_argument('--val_freq', type=int, default=10000000, help='How many iterations before visualization')
    parser.add_argument('--lmbda_phi', type=float, default=1., help='The min value of lambda phi')
    parser.add_argument('--lmbda_psi', type=float, default=1., help='The min value of lambda psi')
    parser.add_argument('--fixed_lambda', type=str_to_bool,default=False,nargs='?', const=True,help='Do not change the delta value')
    parser.add_argument('--max_lambda', type=float,default=0.001,help='Maximum lambda for initialization')
    parser.add_argument('--save_freq', type=int,default=100000,help='How often save parameters')
    parser.add_argument('--mode', default='train', type=str,choices=['train','test'],help='Should we train or test the model?')

    parser = Viz.logger.parse_arguments(parser)
    return parser

