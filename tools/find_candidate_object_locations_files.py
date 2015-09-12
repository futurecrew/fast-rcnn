import _init_paths
import os
import cv2
import cPickle
import time
import sys
import argparse
import pprint
from skimage import io
import numpy as np
from caffe.proto import caffe_pb2
from util_detect import iou
import matplotlib.pyplot as plt
import caffe
from fast_rcnn.test import _get_image_blob
from utils.model import last_conv_size
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from fast_rcnn.test import _bbox_pred, _clip_boxes
from utils.cython_nms import nms
from utils.box_prediction import get_predicted_boxes
from caffe import nms_cpp
from caffe import nms_cuda
from util import prevent_sleep

class Detector(object):
    def check_match(self, file_name, im_blob, ground_rects, pred_rects, match_threshold, scores, rigid_rects):
        found_rects = []
        
        no_to_find = len(ground_rects)
        no_found = 0
        
        #print("number of ground_rects {}".format(no_to_find))
                
        for ground_rect in ground_rects:
            max_overlap = 0
            max_overlap_rect = None 
            for pred_rect in pred_rects:
                overlap = iou(pred_rect[0], pred_rect[1], pred_rect[2], pred_rect[3], 
                            ground_rect[0], ground_rect[1], ground_rect[2], ground_rect[3])
                """
                if overlap >= match_threshold:
                    no_found += 1
                    
                    found_rects.append(pred_rect)
                    break
                """
                if overlap >= max_overlap:
                    max_overlap = overlap
                    max_overlap_rect = pred_rect
                    
            if max_overlap >= match_threshold:
                no_found += 1
                
                found_rects.append(max_overlap_rect)

        print '%s out of %s found using %s candidates. %s.jpg' %(no_found, no_to_find, len(pred_rects), file_name)

        if False:
            im = im_blob[0, :, :, :].transpose((1, 2, 0)).copy()
            im += cfg.PIXEL_MEANS
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
            
            
            plt.imshow(im)
            for ground_rect in ground_rects: 
                plt.gca().add_patch(
                    plt.Rectangle((ground_rect[0], ground_rect[1]), ground_rect[2] - ground_rect[0],
                                  ground_rect[3] - ground_rect[1], fill=False,
                                  edgecolor='b', linewidth=3)
                    )

            for found_rect in found_rects:
                plt.gca().add_patch(
                    plt.Rectangle((found_rect[0], found_rect[1]), found_rect[2] - found_rect[0],
                                  found_rect[3] - found_rect[1], fill=False,
                                  edgecolor='r', linewidth=3)
                    )
                
            plt.show(block=False)
            raw_input("")
            plt.close()
            
            
            
            
            for pred_rect, score, rigid_rect in zip(pred_rects, scores, rigid_rects):
                plt.imshow(im)
                for ground_rect in ground_rects: 
                    plt.gca().add_patch(
                        plt.Rectangle((ground_rect[0], ground_rect[1]), ground_rect[2] - ground_rect[0],
                                      ground_rect[3] - ground_rect[1], fill=False,
                                      edgecolor='b', linewidth=3)
                        )
                plt.gca().add_patch(
                    plt.Rectangle((rigid_rect[0], rigid_rect[1]), rigid_rect[2] - rigid_rect[0],
                                  rigid_rect[3] - rigid_rect[1], fill=False,
                                  edgecolor='g', linewidth=3)
                    )
                plt.gca().add_patch(
                    plt.Rectangle((pred_rect[0], pred_rect[1]), pred_rect[2] - pred_rect[0],
                                  pred_rect[3] - pred_rect[1], fill=False,
                                  edgecolor='r', linewidth=3)
                    )
                plt.text(pred_rect[0] + 10, pred_rect[1] + 25, score)
                
                plt.show(block=False)
                raw_input("")
                plt.close()
            
            
        return no_to_find, no_found

    
    def gogo(self, MAX_CAND_BEFORE_NMS, MAX_CAND_AFTER_NMS, voc_base_folder, prototxt, 
             caffemodel, gt, data_list, test_data, step):
        NMS_THRESH = 0.7
        match_threshold = 0.5
        
        image_folder = voc_base_folder + '/JPEGImages'

        with open(gt, 'rb') as fid:
            gtdb = cPickle.load(fid)
                        
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        
        no = 0
        total_no_to_find = 0
        total_no_found = 0
        box_list_to_save = []
        input_data = open(data_list).readlines()
        
        for index, file_name in enumerate(input_data):
            
            file_name = file_name.replace('\n', '').replace('\r', '')
            
            if len(file_name) == 0:
                continue

            no += 1
            
            # DJDJ
            #if file_name != '000009':
            #    continue
            
            im = cv2.imread(image_folder + '/' + file_name + '.jpg')
            
            org_img_height = im.shape[0]
            org_img_width = im.shape[1]
            
            blobs = {'data' : None}
            blobs['data'], im_scale_factors = _get_image_blob(im)
            
            net.blobs['data'].reshape(*(blobs['data'].shape))
            
            #print 'time1 : %.3f' % time.time()
            blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False))
            
            #print 'time2 : %.3f' % time.time()

            cls_pred = blobs_out['cls_pred']
            if 'bbox_pred_rpn' in blobs_out:
                box_deltas = blobs_out['bbox_pred_rpn']
            else:
                box_deltas = blobs_out['bbox_pred']
            
            
            rigid_rects, pred_boxes, sorted_scores = get_predicted_boxes(cls_pred, box_deltas,
                                           blobs['data'].shape[2], blobs['data'].shape[3],
                                           NMS_THRESH, MAX_CAND_BEFORE_NMS,
                                           MAX_CAND_AFTER_NMS)
        
            # Rescale boxes back according to the original image size
            pred_boxes = pred_boxes / im_scale_factors[0]        
            rigid_rects = rigid_rects / im_scale_factors[0]        
            gt_boxes = gtdb[no-1]['boxes']
            
            #gt_boxes = gtdb[no-1]['boxes'] * im_scale_factors

            
            #for pred_box in pred_boxes:
            #    print 'pred_box : %s' % (pred_box, )
                
            #for gt_box in gt_boxes:
            #    print 'gt_box : %s' % (gt_box, )
                
            no_to_find, no_found = self.check_match(file_name, blobs['data'], gt_boxes, 
                                                    pred_boxes, match_threshold, 
                                                    sorted_scores, rigid_rects)

            no_candidates = len(pred_boxes)
            
            total_no_to_find += no_to_find
            total_no_found += no_found
             
            print '[%s] accuracy : %.3f' % (no, float(total_no_found) / float(total_no_to_find))  
            
            """
            for i in range(len(pred_boxes)):
                if pred_boxes[i, 0] > pred_boxes[i, 2]:
                    print '[ error end ]'
                    print 'file_name : %s' % file_name
                    print 'img_width_for_train : %s' % img_height_for_train
                    print 'img_width_for_train : %s' % img_width_for_train
                    print 'pred_boxes[%s] : %s' % (i, pred_boxes[i])
            """
            
            assert (pred_boxes[:, 0] >= 0).all()
            assert (pred_boxes[:, 1] >= 0).all()
            assert (pred_boxes[:, 2] < org_img_width).all()
            assert (pred_boxes[:, 3] < org_img_height).all()
            assert (pred_boxes[:, 2] >= pred_boxes[:, 0]).all()
            assert (pred_boxes[:, 3] >= pred_boxes[:, 1]).all()
                        
            box_list_to_save.append(pred_boxes.astype(np.int16))
        
        # Save RPN proposal boxes    
        proposal_file = os.path.join('data', 'rpn_data',
                '{:s}_{:s}_rpn_top_{:d}_candidate.pkl'.
                format(test_data, step, MAX_CAND_AFTER_NMS))

        with open(proposal_file, 'wb') as fid:
            cPickle.dump(box_list_to_save, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote rpn roidb to {}'.format(proposal_file)            

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate RPN detections')
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--prototxt', dest='prototxt',
                        help='prototxt to use',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--data_type', dest='data_type',
                        help='data_type(trainval or test)',
                        default=None, type=str)
    parser.add_argument('--step', dest='step',
                        help='step(1 or 3)',
                        default=None, type=str)
    parser.add_argument('--max_output', dest='max_output',
                        help='Maximum number of candidates',
                        default=2300, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    voc_base_folder = 'E:/data/VOCdevkit2/VOC2007/'
    
    #caffemodel = 'E:/project/fast-rcnn/output/faster_rcnn_cls_only/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_100.caffemodel'
    #caffemodel = 'E:/project/fast-rcnn/output/faster_rcnn_bbox_only/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_100.caffemodel'
    #prototxt = 'E:/project/fast-rcnn/models/VGG_CNN_M_1024/rpn/test.prototxt'

    iters = 80000
    
    MAX_CAND_BEFORE_NMS = 10000
    
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)
    
    prevent_sleep()

    test_data = 'voc_2007_%s' % args.data_type
    step = 'step_%s' % args.step
    caffemodel = args.pretrained_model
    MAX_CAND_AFTER_NMS = args.max_output
    prototxt = args.prototxt

    data_list = voc_base_folder + '/ImageSets/Main/%s.txt' % args.data_type
    gt = 'E:/project/fast-rcnn/data/cache/voc_2007_%s_gt_roidb.pkl' % args.data_type
    
    #cfg_file = 'E:/project/fast-rcnn/experiments/cfgs/faster_rcnn.yml'

    caffe.set_mode_gpu()
    
    detector = Detector()
    detector.gogo(MAX_CAND_BEFORE_NMS, MAX_CAND_AFTER_NMS, voc_base_folder, prototxt, 
                  caffemodel, gt, data_list, test_data, step)