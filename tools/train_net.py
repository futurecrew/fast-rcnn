#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import io
from util import prevent_sleep
from logger import stdout_redirector, redirect_stdout, stdout_redirected
import os

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--cpu', dest='cpu',
                        help='CPU',
                        default=-1, type=int)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--train_target', dest='train_target',
                        help='train target',
                        default='frcnn', type=str)
    parser.add_argument('--proposal', dest='proposal',
                        help='proposal to use for train',
                        default='ss', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Called with args:')
    print(args)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    prevent_sleep()
    
    # set up caffe
    if args.cpu == 0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        if args.gpu_id is not None:
            caffe.set_device(args.gpu_id)
    
    train_target = args.train_target
    proposal = args.proposal

    if 'VGG16' in args.pretrained_model:
        model_name = 'VGG16'
    elif 'VGG_CNN_M_1024' in args.pretrained_model:
        model_name = 'VGG_CNN_M_1024'
    else:
        raise Exception("This model is not supported. %s" % args.pretrained_model)
    
    cfg.MODEL_NAME = model_name

    imdb = get_imdb(args.imdb_name, train_target)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    
    # DJDJ
    #imdb.rpn_roidb()
    
    roidb = get_training_roidb(imdb, args.train_target)

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters,
              train_target=train_target)
