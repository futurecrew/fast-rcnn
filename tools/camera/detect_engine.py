import random
import string

import cherrypy
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

CLASSES_CAR = ('__background__',
           'bicycle', 'bus', 'car', 'motorbike', 'person')

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

class Detector(object):
    def detect(self, image_name, mode, mixed=True):
        
        # DJDJ
        # Load the demo image
        #im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name)
        #im = cv2.imread(im_file)
        
        im = cv2.imread(image_name)
    
        # Detect all object classes and regress object bounds
        for i in range(1):
            timer = Timer()
            timer.tic()
            if mixed:
                scores, boxes = im_detect_mixed(self.net, im)
            else:
                scores, boxes = im_detect(self.net, im, obj_proposals)
            timer.toc()
            print ('Detection took {:.3f}s for '
                   '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
        # Visualize detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        timer = Timer()
        result = {}
        
        if mode == '3':     # Car mode
            classes = CLASSES_CAR
        else:
            classes = CLASSES
            
        for cls in CLASSES:
            if mode == '3' and (cls in CLASSES_CAR) == False:     # Car mode
                continue
            
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            
            timer.tic()
            keep = nms(dets, NMS_THRESH)
            timer.toc()
            
            dets = dets[keep, :]
            result[cls_ind] = dets
            #print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls, CONF_THRESH)
            #vis_detections(im, cls, dets, thresh=CONF_THRESH)
        #print ('nms took {:.3f}s').format(timer.total_time)
        
        return result        
                
    def initialize(self, app):
        prototxt = 'models/VGG16/rpn/test_mixed.prototxt'
        #caffemodel_mixed_frcnn = 'output/fast_rcnn_lazy/voc_2007_2012_trainval_with_rpn/vgg16_fast_rcnn_step4_with_rpn_iter_140000.caffemodel'    
        #caffemodel_mixed_rpn = 'output/faster_rcnn_lazy/voc_2007_2012_trainval/vgg16_rpn_step3_iter_80000.caffemodel'    
        
        caffemodel_mixed_frcnn = 'output/fast_rcnn_lazy/voc_2007_2012_trainval_with_rpn/vgg16_fast_rcnn_step4_with_rpn_iter_110000.caffemodel'    
        caffemodel_mixed_rpn = 'output/faster_rcnn_lazy/voc_2007_2012_trainval/vgg16_rpn_step3_iter_80000.caffemodel'    
    
        if not os.path.isfile(caffemodel_mixed_frcnn):
            raise IOError(('{:s} not found.').format(caffemodel_mixed_frcnn))
        if not os.path.isfile(caffemodel_mixed_rpn):
            raise IOError(('{:s} not found.').format(caffemodel_mixed_rpn))
    
        caffe.set_mode_gpu()
        caffe.set_device(0)
        #net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        
        self.net = caffe.Net(prototxt, caffe.TEST)
        self.net.copy_from(caffemodel_mixed_frcnn)
        self.net.copy_from(caffemodel_mixed_rpn)
    
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/000004.jpg'
        
        #demo(net, '000004', ('car',), mixed=False)
        #plt.show()
    
        #self.detect('000004.jpg')
