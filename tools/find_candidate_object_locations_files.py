import _init_paths
import os
import cv2
import cPickle
import time
from skimage import io
import numpy as np
from caffe.proto import caffe_pb2
from util_detect import iou
import matplotlib.pyplot as plt
import caffe
from labels import read_label_file
from fast_rcnn.test import _get_image_blob
from utils.model import last_conv_size
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from fast_rcnn.test import _bbox_pred, _clip_boxes
from utils.cython_nms import nms
from caffe import nms_cpp
#from utils.nms import nms
#from utils.nms2 import nms
#from utils.nms3 import nms

class Detector(object):
    
    def get_img_rect(self, img_height, img_width, conv_height, conv_width, axis1, axis2, axis3):

        anchors = np.array([[128*2, 128*1], [128*1, 128*1], [128*1, 128*2], 
                           [256*2, 256*1], [256*1, 256*1], [256*1, 256*2], 
                           [512*2, 512*1], [512*1, 512*1], [512*1, 512*2]])
                
        img_center_x = img_width * axis3 / conv_width
        img_center_y = img_height * axis2 / conv_height
        anchor_size = anchors[axis1]
        img_x1 = img_center_x - anchor_size[:, 0] / 2 
        img_x2 = img_center_x + anchor_size[:, 0] / 2 
        img_y1 = img_center_y - anchor_size[:, 1] / 2 
        img_y2 = img_center_y + anchor_size[:, 1] / 2 
        
        img_rect = np.zeros((len(axis1), 4), np.float32)
        img_rect[:, 0] = img_x1
        img_rect[:, 1] = img_y1
        img_rect[:, 2] = img_x2
        img_rect[:, 3] = img_y2
        
        return img_rect

    def check_match(self, file_name, im_blob, ground_rects, pred_rects, match_threshold, scores, proposal_rects):
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
            
            
            
            """
            for pred_rect, score, proposal_rect in zip(pred_rects, scores, proposal_rects):
                plt.imshow(im)
                for ground_rect in ground_rects: 
                    plt.gca().add_patch(
                        plt.Rectangle((ground_rect[0], ground_rect[1]), ground_rect[2] - ground_rect[0],
                                      ground_rect[3] - ground_rect[1], fill=False,
                                      edgecolor='b', linewidth=3)
                        )
                plt.gca().add_patch(
                    plt.Rectangle((proposal_rect[0], proposal_rect[1]), proposal_rect[2] - proposal_rect[0],
                                  proposal_rect[3] - proposal_rect[1], fill=False,
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
            """
            
        return no_to_find, no_found
    
    def gogo(self, TOP_N, MAX_CANDIDATES, voc_base_folder, prototxt, caffemodel, gt, data_list):
        NMS_THRESH = 0.7
        match_threshold = 0.5
        
        image_folder = voc_base_folder + '/JPEGImages'

        with open(gt, 'rb') as fid:
            gtdb = cPickle.load(fid)
                        
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        
        no = 0
        total_no_to_find = 0
        total_no_found = 0
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
            
            blobs = {'data' : None}
            blobs['data'], im_scale_factors = _get_image_blob(im)
            
            net.blobs['data'].reshape(*(blobs['data'].shape))
            
            #print 'time1 : %.3f' % time.time()
            blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False))
            
            #print 'time2 : %.3f' % time.time()

            cls_pred = blobs_out['cls_pred']
            box_deltas = blobs_out['bbox_pred']
            
            pos_pred = cls_pred[:, :, 1, :, :]
            sorted_pred = np.argsort(pos_pred, axis=None)[::-1]
            sorted_scores = np.sort(pos_pred, axis=None)[::-1]
            height = pos_pred.shape[2]
            width = pos_pred.shape[3]
            proposal_rects = []

            top_index = sorted_pred[:TOP_N]
            sorted_scores = sorted_scores[:TOP_N]
            axis1 = top_index / height / width
            axis2 = top_index / width % height
            axis3 = top_index % width
            
            img_height = blobs['data'].shape[2]
            img_width = blobs['data'].shape[3]


            # DJDJ
            """
            axis1 = np.array([2, 2, 2, 2])
            axis2 = np.array([22, 22, 22, 22])
            axis3 = np.array([26, 28, 30, 32])
            top_index = axis1 * pos_pred.shape[2] * pos_pred.shape[3] + axis2 * pos_pred.shape[3] + axis3
            sorted_scores = sorted_scores[top_index]
            """
            
            
            #print 'time3 : %.3f' % time.time()
        
            proposal_rects = self.get_img_rect(img_height, img_width, pos_pred.shape[2], pos_pred.shape[3], axis1, axis2, axis3)
            
            #print 'time4 : %.3f' % time.time()
            
            """
            print ''
            print 'proposal_rects[0] : %s' % proposal_rects[0]            
            """

            box_deltas_rects = np.zeros((len(axis1), 4), np.float32)
            box_deltas_rects[:, 0] = box_deltas[0, axis1*4, axis2, axis3]
            box_deltas_rects[:, 1] = box_deltas[0, axis1*4+1, axis2, axis3]
            box_deltas_rects[:, 2] = box_deltas[0, axis1*4+2, axis2, axis3]
            box_deltas_rects[:, 3] = box_deltas[0, axis1*4+3, axis2, axis3]

            pred_boxes = _bbox_pred(proposal_rects, box_deltas_rects)
            pred_boxes = _clip_boxes(pred_boxes, (img_height, img_width))            
            
            box_info = np.hstack((pred_boxes,
                                  sorted_scores[:, np.newaxis])).astype(np.float32)            

            time1 = time.time()
            keep1 = nms(box_info, NMS_THRESH, MAX_CANDIDATES)
            print 'keep1 : %s' % len(keep1)
            print 'nms %s took %.3f sec' % (len(box_info), time.time() - time1)            

            x1s = np.ascontiguousarray(box_info[:, 0])
            y1s = np.ascontiguousarray(box_info[:, 1])
            x2s = np.ascontiguousarray(box_info[:, 2])
            y2s = np.ascontiguousarray(box_info[:, 3])
            scores = np.ascontiguousarray(box_info[:, 4])
            time1 = time.time()
            keep = nms_cpp(x1s, y1s, x2s, y2s, scores, NMS_THRESH)
            print 'keep : %s' % len(keep)
            print 'nms %s took %.3f sec' % (len(box_info), time.time() - time1)
            
            pred_boxes = pred_boxes[keep, :]
            pred_boxes = pred_boxes[:MAX_CANDIDATES]
        
            gt_boxes = gtdb[no-1]['boxes'] * im_scale_factors
            
            #for pred_box in pred_boxes:
            #    print 'pred_box : %s' % (pred_box, )
                
            #for gt_box in gt_boxes:
            #    print 'gt_box : %s' % (gt_box, )
                
            no_to_find, no_found = self.check_match(file_name, blobs['data'], gt_boxes, pred_boxes, match_threshold, sorted_scores, proposal_rects)

            #print 'time7 : %.3f' % time.time()
            
            no_candidates = len(pred_boxes)
            
            total_no_to_find += no_to_find
            total_no_found += no_found
             
            print '[%s] accuracy : %.3f' % (no, float(total_no_found) / float(total_no_to_find))  
                

if __name__ == '__main__':
    voc_base_folder = 'E:/data/VOCdevkit2/VOC2007/'
    prototxt = 'E:/project/fast-rcnn/models/VGG_CNN_M_1024/rpn/test.prototxt'

    #caffemodel = 'E:/project/fast-rcnn/output/faster_rcnn_cls_only/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_100.caffemodel'
    #caffemodel = 'E:/project/fast-rcnn/output/faster_rcnn_bbox_only/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_100.caffemodel'

    iters = 80000
    
    TOP_N = 30000
    MAX_CANDIDATES = 2300
    
    caffemodel = 'E:/project/fast-rcnn/output/faster_rcnn/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_%s.caffemodel' % iters

    #data_list = voc_base_folder + '/ImageSets/Main/trainval.txt'
    #gt = 'E:/project/fast-rcnn/data/cache/voc_2007_trainval_gt_roidb.pkl'
    
    data_list = voc_base_folder + '/ImageSets/Main/test.txt'
    gt = 'E:/project/fast-rcnn/data/cache/voc_2007_test_gt_roidb.pkl'
    
    cfg_file = 'E:/project/fast-rcnn/experiments/cfgs/faster_rcnn.yml'
    cfg_from_file(cfg_file)

    caffe.set_mode_gpu()
    
    detector = Detector()
    detector.gogo(TOP_N, MAX_CANDIDATES, voc_base_folder, prototxt, caffemodel, gt, data_list)