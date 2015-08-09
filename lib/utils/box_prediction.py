import os
import cv2
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from fast_rcnn.test import _bbox_pred, _clip_boxes
from utils.cython_nms import nms
from caffe import nms_cpp
from caffe import nms_cuda

def get_img_rect(img_height, img_width, conv_height, conv_width, axis1, axis2, axis3):

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


# Delete out of range rows
def _remove_out_of_ranges(boxes, scores, img_height, img_width):
    out_of_range = np.where(boxes[:, 0] >= img_width - 1)[0]            
    if len(out_of_range) > 0:
        boxes = np.delete(boxes, out_of_range, 0)
        scores = np.delete(scores, out_of_range, 0)

    out_of_range = np.where(boxes[:, 1] >= img_height - 1)[0]            
    if len(out_of_range) > 0:
        boxes = np.delete(boxes, out_of_range, 0)
        scores = np.delete(scores, out_of_range, 0)

    out_of_range = np.where(boxes[:, 2] < 0)[0]            
    if len(out_of_range) > 0:
        boxes = np.delete(boxes, out_of_range, 0)
        scores = np.delete(scores, out_of_range, 0)

    out_of_range = np.where(boxes[:, 3] < 0)[0]            
    if len(out_of_range) > 0:
        boxes = np.delete(boxes, out_of_range, 0)
        scores = np.delete(scores, out_of_range, 0)
        
    return boxes, scores


def get_predicted_boxes(cls_pred, box_deltas, 
                      img_height_for_train, img_width_for_train,
                      nms_thres, max_cand_before_nms,
                      max_cand_after_nms):
    pos_pred = cls_pred[:, :, 1, :, :]
    sorted_pred = np.argsort(pos_pred, axis=None)[::-1]
    sorted_scores = np.sort(pos_pred, axis=None)[::-1]
    height = pos_pred.shape[2]
    width = pos_pred.shape[3]
    rigid_rects = []

    top_index = sorted_pred[:max_cand_before_nms]
    sorted_scores = sorted_scores[:max_cand_before_nms]
    axis1 = top_index / height / width
    axis2 = top_index / width % height
    axis3 = top_index % width
    
    # DJDJ
    """
    axis1 = np.array([2, 2, 2, 2])
    axis2 = np.array([22, 22, 22, 22])
    axis3 = np.array([26, 28, 30, 32])
    top_index = axis1 * pos_pred.shape[2] * pos_pred.shape[3] + axis2 * pos_pred.shape[3] + axis3
    sorted_scores = sorted_scores[top_index]
    """
    
    
    #print 'time3 : %.3f' % time.time()

    rigid_rects = get_img_rect(img_height_for_train, img_width_for_train, 
                                  pos_pred.shape[2], pos_pred.shape[3], axis1, axis2, axis3)
    
    #print 'time4 : %.3f' % time.time()
    
    """
    print ''
    print 'rigid_rects[0] : %s' % rigid_rects[0]            
    """

    box_deltas_rects = np.zeros((len(axis1), 4), np.float32)
    box_deltas_rects[:, 0] = box_deltas[0, axis1*4, axis2, axis3]
    box_deltas_rects[:, 1] = box_deltas[0, axis1*4+1, axis2, axis3]
    box_deltas_rects[:, 2] = box_deltas[0, axis1*4+2, axis2, axis3]
    box_deltas_rects[:, 3] = box_deltas[0, axis1*4+3, axis2, axis3]

    pred_boxes = _bbox_pred(rigid_rects, box_deltas_rects)

    #print 'img_height : %s' % img_height
    #print 'img_width : %s' % img_width
    
    # Delete out of range rows
    pred_boxes, sorted_scores = _remove_out_of_ranges(pred_boxes, sorted_scores, 
                                      img_height_for_train, img_width_for_train)
    
    
    """
    for i in range(len(pred_boxes)):
        if pred_boxes[i, 0] > pred_boxes[i, 2]:
            print '[ error 2 ]'
            print 'pred_boxes[%s] : %s' % (i, pred_boxes[i])
    """

    pred_boxes = _clip_boxes(pred_boxes, (img_height_for_train, img_width_for_train))            
    
    """
    for i in range(len(pred_boxes)):
        if pred_boxes[i, 0] > pred_boxes[i, 2]:
            print '[ error 3 ]'
            print 'file_name : %s' % file_name
            print 'pred_boxes[%s] : %s' % (i, pred_boxes[i])
    """
    
    box_info = np.hstack((pred_boxes,
                          sorted_scores[:, np.newaxis])).astype(np.float32)            

    """
    time1 = time.time()
    keep = nms(box_info, nms_thres, max_cand_after_nms)
    print ''
    print 'nms %s took %.3f sec. keep : %s' % (len(box_info), time.time() - time1, len(keep))
    """            

    x1s = np.ascontiguousarray(box_info[:, 0])
    y1s = np.ascontiguousarray(box_info[:, 1])
    x2s = np.ascontiguousarray(box_info[:, 2])
    y2s = np.ascontiguousarray(box_info[:, 3])
    scores = np.ascontiguousarray(box_info[:, 4])
    time1 = time.time()
    keep = nms_cpp(x1s, y1s, x2s, y2s, scores, nms_thres, max_cand_after_nms)
    print 'nms_cpp %s took %.3f sec. keep : %s' % (len(box_info), time.time() - time1, len(keep))

    """
    time1 = time.time()
    keep = nms_cuda(x1s, y1s, x2s, y2s, scores, nms_thres, max_cand_after_nms)
    print 'nms_cuda %s took %.3f sec. keep : %s' % (len(box_info), time.time() - time1, len(keep))
    """
    
    pred_boxes = pred_boxes[keep, :]
    pred_boxes = pred_boxes[:max_cand_after_nms]
    
    return rigid_rects, pred_boxes, sorted_scores
        