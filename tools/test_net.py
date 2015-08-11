#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
from util import prevent_sleep

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--model_to_use', dest='model_to_use',
                        help='train model',
                        default='frcnn', type=str)
    parser.add_argument('--proposal', dest='proposal',
                        help='proposal type to use for test',
                        default='ss', type=str)
    parser.add_argument('--proposal_file', dest='proposal_file',
                        help='proposal file to use for test',
                        default='', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output directory',
                        default='', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    prevent_sleep()
    
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    model_to_use = args.model_to_use
    proposal = args.proposal
    proposal_file = args.proposal_file
    output_dir = args.output_dir

    if 'VGG16' in args.caffemodel.upper():
        model_name = 'VGG16'
    elif 'VGG_CNN_M_1024' in args.caffemodel.upper():
        model_name = 'VGG_CNN_M_1024'
    else:
        raise Exception("This model is not supported. %s" % args.caffemodel)
    
    cfg.MODEL_NAME = model_name

    imdb = get_imdb(args.imdb_name, model_to_use=model_to_use, proposal=proposal, proposal_file=proposal_file)
    imdb.competition_mode(args.comp_mode)

    # DJDJ
    #imdb.evaluate_detections(None, output_dir)
    
    test_net(net, imdb, proposal, proposal_file, output_dir)
