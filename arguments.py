import cvgutils.Viz as Viz
import argparse

def parse_arguments_deepfnf():

    parser = argparse.ArgumentParser()
    parser.add_argument('--TESTPATH', type=str, default='/home/mohammad/Projects/optimizer/DifferentiableSolver/data/testset_nojitter', help='testset path')
    parser.add_argument('--TLIST', type=str, default='data/train.txt', help='Training dataset filename')
    parser.add_argument('--VPATH', type=str, default='data/valset', help='Validation dataset')
    parser.add_argument('--model', type=str, default='deepfnf',choices=['unet','deepfnf','deepfnf+fft','deepfnf+fft_highdim','deepfnf+fft_grad_image','deepfnf+fft_helmholz','deepfnf+fft_helmholz_highdim'], help='Validation dataset')
    parser.add_argument('--outchannels', type=int, default=3, help='Number of the output channels of the UNet')
    parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
    parser.add_argument('--weight_dir', type=str, default='wts', help='Weight dir')
    parser.add_argument('--weight_file', type=str, default='wts/', help='Weight dir')
    parser.add_argument('--visualize_freq', type=int, default=10001, help='How many iterations before visualization')
    parser.add_argument('--val_freq', type=int, default=10000, help='How many iterations before visualization')
    parser.add_argument('--lmbda_phi', type=float, default=1., help='The min value of lambda phi')
    parser.add_argument('--lmbda_psi', type=float, default=1., help='The min value of lambda psi')
    parser.add_argument('--fixed_lambda', action='store_true',help='Do not change the delta value')
    parser.add_argument('--max_lambda', type=float,default=0.001,help='Maximum lambda for initialization')
    parser.add_argument('--save_freq', type=int,default=100000,help='How often save parameters')
    parser.add_argument('--mode', default='train', type=str,choices=['train','test'],help='Should we train or test the model?')

    parser = Viz.logger.parse_arguments(parser)
    return parser

