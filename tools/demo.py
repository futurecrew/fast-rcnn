#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, im_detect_mixed
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg16_comp': ('VGG16/compressed',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024_comp': ('VGG_CNN_M_1024/compressed',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel'),
        'caffenet_comp': ('CaffeNet/compressed',
                     'caffenet_fast_rcnn_iter_40000_svd_fc6_1024_fc7_256.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name, classes, mixed=False):
    """Detect object classes in an image using pre-computed object proposals."""

    if mixed == False:
        # Load pre-computed Selected Search object proposals
        box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                                image_name + '_boxes.mat')
        obj_proposals = sio.loadmat(box_file)['boxes']
    
    
    
    # DJDJ
    #obj_proposals = obj_proposals[:300, ]

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    for i in range(3):
        timer = Timer()
        timer.tic()
        if mixed:
            scores, boxes = im_detect_mixed(net, im)
        else:
            scores, boxes = im_detect(net, im, obj_proposals)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    timer = Timer()
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        
        timer.tic()
        keep = nms(dets, NMS_THRESH)
        timer.toc()
        
        dets = dets[keep, :]
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
                                                                    CONF_THRESH)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    print ('nms took {:.3f}s').format(timer.total_time)        

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
                              NETS[args.demo_net][1])

    prototxt_mixed = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'rpn', 'test_mixed.prototxt')
    caffemodel_mixed_frcnn = os.path.join(cfg.ROOT_DIR, 'output', 'fast_rcnn',
                              'voc_2007_trainval_with_rpn', 
                              'vgg_cnn_m_1024_fast_rcnn_step4_with_rpn_iter_80000.caffemodel')
    caffemodel_mixed_rpn = os.path.join(cfg.ROOT_DIR, 'output', 'faster_rcnn',
                              'voc_2007_trainval', 
                              'vgg_cnn_m_1024_rpn_step3_iter_80000.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    
    net_mixed = caffe.Net(prototxt_mixed, caffe.TEST)
    net_mixed.copy_from(caffemodel_mixed_frcnn)
    net_mixed.copy_from(caffemodel_mixed_rpn)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Demo for data/demo/000004.jpg'
    demo(net, '000004', ('car',), mixed=False)
    plt.show()

    demo(net_mixed, '000004', ('car',), mixed=True)
    plt.show()

    """
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Demo for data/demo/001551.jpg'
    demo(net, '001551', ('sofa', 'tvmonitor'), mixed=False)
    plt.show()

    demo(net_mixed, '001551', ('sofa', 'tvmonitor'), mixed=True)
    plt.show()
    """
    
    
