#/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Command Line Interface (CLI) for the synthetic experiments. """

# package imports
from s3ts.exp.settings import ExperimentSettings
from s3ts.exp.loop import experiment_loop

# standard library
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='''Perform the experiments showcased in the article.''')

    # compulsory parameters
    parser.add_argument('--dset', type=str, required=True,
        help='Name of the dataset from which create the DTWs')
    parser.add_argument('--dsrc', type=str, required=True, choices=['ts', 'df', 'gf'],
        help=r'Data source for the model {ts: Time Series, df: Dissimilarity Frames, gf: Gramian Frames}')
    parser.add_argument('--arch', type=str, required=True, choices=['nn', 'rnn', 'cnn', 'res'],
        help='Name of the architecture from which create the model')
    parser.add_argument('--pret', type=bool, action=argparse.BooleanOptionalAction, 
                        default=False, help='Use pretrained encoder or not (df/gf mode only)')
    parser.add_argument('--pret_mode', type=bool, action=argparse.BooleanOptionalAction,  
                        default=False, help='Switch between train and pretrain mode')
    
    # model params
    parser.add_argument('--wdw_len', type=int, default=10, help='Window length')
    parser.add_argument('--wdw_str', type=int, default=1, help='Window stride')
    parser.add_argument('--str_str', type=bool, action=argparse.BooleanOptionalAction,  
                        default=False, help='Whether to stride the stream during pretrain')
    parser.add_argument('--enc_feats', type=int, default=None, help='Encoder complexity hyperparameter.')
    parser.add_argument('--dec_feats', type=int, default=64, help='Decoder complexity hyperparameter.')
    
    # dataset sampling
    parser.add_argument('--rho_dfs', type=float, default=0.1, help='Forgetting parameter (DF only)')
    parser.add_argument('--nsamp_tra', type=int, default=1000, help='Event multiplier for the training STS')
    parser.add_argument('--nsamp_tst', type=int, default=1000, help='Event multiplier for the training STS')
    parser.add_argument('--nsamp_pre', type=int, default=1000, help='Event multiplier for the training STS')
    parser.add_argument('--val_size', type=float, default=0.25, help='Cross-validation repetition number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size used during training')
    parser.add_argument('--max_epochs', type=int, default=60, help='Maximum number of epochs for the training')
    parser.add_argument('--lr', type=float, default=1E-4, help='Value of the learning rate')
    
    # directories
    parser.add_argument('--dir_results', type=str, default="results/",
                        help='Results file for the training')
    parser.add_argument('--dir_encoders', type=str, default="encoders", 
                        help='Directory for the training files')
    parser.add_argument('--dir_datasets', type=str, default="datasets/",
                        help='Results file for the training')
    parser.add_argument('--dir_training', type=str, default="training/experiments", 
                        help='Directory for the training files')
    args = parser.parse_args()
   
    # ~~~~~~~~~~~~ Put the arguments in variables ~~~~~~~~~~~~
    
    param = vars(args)
    experiment_loop(ExperimentSettings(**param))
