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
    parser.add_argument('--restore', dest='restore',
                        help='solverstate file path',
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
    parser.add_argument('--model_to_use', dest='model_to_use',
                        help='train model',
                        default='frcnn', type=str)
    parser.add_argument('--proposal', dest='proposal',
                        help='proposal to use for train',
                        default='ss', type=str)
    parser.add_argument('--proposal_file', dest='proposal_file',
                        help='proposal file to use for test',
                        default='', type=str)

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
    
    model_to_use = args.model_to_use
    proposal = args.proposal
    proposal_file = args.proposal_file

    if args.pretrained_model is not None:
        model_name = args.pretrained_model.upper()
    elif args.restore is not None:
        model_name = args.restore.upper()
        
    if 'VGG16' in model_name:
        model_name = 'VGG16'
    elif 'VGG_CNN_M_1024' in model_name:
        model_name = 'VGG_CNN_M_1024'
    elif 'GOOGLENET2' in model_name:
        model_name = 'GOOGLENET2'
    elif 'GOOGLENET3' in model_name:
        model_name = 'GOOGLENET3'
    elif 'GOOGLENET4' in model_name:
        model_name = 'GOOGLENET4'
    elif 'GOOGLENET5' in model_name:
        model_name = 'GOOGLENET5'
    elif 'GOOGLENET' in model_name:
        model_name = 'GOOGLENET'
    else:
        raise Exception("This model is not supported. %s" % model_name)
    
    cfg.MODEL_NAME = model_name

    imdb = get_imdb(args.imdb_name, model_to_use, proposal, proposal_file)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    
    # DJDJ
    #imdb.rpn_train_roidb()
    
    roidb = get_training_roidb(imdb, args.model_to_use)

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, imdb.bbox_means, imdb.bbox_stds, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              restore=args.restore,
              max_iters=args.max_iters,
              model_to_use=model_to_use,
              proposal=proposal)
