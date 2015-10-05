#!/usr/bin/env python

import _init_paths
import os
import cv2
from multiprocessing import Process, Queue, Lock
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


def gogo(MAX_CAND_AFTER_NMS, gpu_id_list, MULTI_CPU_NO,
         data_folder, data_ext, prototxt, 
         caffemodel, gt, data_list, test_data, model_name, step):
    with open(gt, 'rb') as fid:
        gtdb = cPickle.load(fid)
    
    total_no_to_find = 0
    total_no_found = 0

    start_time = time.time()
    
    # DJDJ
    #gtdb = gtdb[0:300]
    
    if data_list != None and os.path.exists(data_list):
        input_data = open(data_list).readlines()
    else:
        input_data = []
        for gt in gtdb:
            input_data.append(gt['label_file'].split('.xml')[0])
    
    queue_gpu_to_main = Queue()
    total_data_no = len(input_data)
    chunk_size = total_data_no / len(gpu_id_list)
    if total_data_no % len(gpu_id_list) > 0:
        chunk_size += 1
    
    proposal_folder  = 'output/rpn_data/' + test_data 
    if not os.path.exists(proposal_folder):
        os.makedirs(proposal_folder)
        
    for i in range(len(gpu_id_list)):
        start_idx = chunk_size * i
        end_idx = min(chunk_size * (i + 1), total_data_no)
        gpu_id = gpu_id_list[i]
        
        print 'creating a gpu process[%s] (%s ~ %s)' % (gpu_id, start_idx, end_idx-1)
        p = Process(target=gogo_one_gpu, args=(MAX_CAND_AFTER_NMS, MULTI_CPU_NO,
                                               gpu_id, queue_gpu_to_main,
                                               data_folder, data_ext, prototxt, 
                                               proposal_folder,
                                               input_data[start_idx : end_idx], 
                                               gtdb[start_idx : end_idx],
                                               caffemodel))
        p.start()

    end_gpu_no = 0
    total_box_list_to_save = []
    while True:
        queue_data = queue_gpu_to_main.get()
        end_gpu_no += 1
        
        print 'get end_gpu %s' % queue_data['gpuid']
        if end_gpu_no == len(gpu_id_list):
            break
        else:
            continue

    total_no_to_find = 0
    total_no_found = 0
            
    for gpu_id in gpu_id_list:
        file = proposal_folder + '/gpu_result_' + str(gpu_id) + '.pickle'
        with open(file, 'rb') as fid:
            dump_data = cPickle.load(fid)
            print 'read pickled data : %s' % file
            total_no_to_find += dump_data['total_no_to_find']
            total_no_found += dump_data['total_no_found']
            total_box_list_to_save.extend(dump_data['box_list_to_save'])


    print 'total elapsed time = %.1f min' % (float(time.time() - start_time) / 60)
    print 'total accuracy [%s/%s] : %.3f' % (total_no_found, total_no_to_find,
                                             float(total_no_found) / float(total_no_to_find))  
    
    # Save RPN proposal boxes    
    proposal_file = proposal_folder + '/' + model_name + '_' + step + '_rpn_top_' + str(MAX_CAND_AFTER_NMS) + '_candidate.pkl'

    with open(proposal_file, 'wb') as fid:
        cPickle.dump(total_box_list_to_save, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote rpn roidb to {}'.format(proposal_file)            
    
        
def gogo_one_gpu(MAX_CAND_AFTER_NMS, MULTI_CPU_NO,
                 gpuid, queue_gpu_to_main,
                 data_folder, data_ext, prototxt, 
                 proposal_folder,
                 input_data, gtdb,
                 caffemodel):

    caffe.set_mode_gpu()
    caffe.set_device(gpuid)
        
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    
    process_list = []
    total_box_list_to_save = []
    queue_child_to_gpu = Queue()
    queue_gpu_to_child_list = []
    total_data_no = len(input_data)
    chunk_size = total_data_no / MULTI_CPU_NO
    if total_data_no % MULTI_CPU_NO > 0:
        chunk_size += 1
    
    for i in range(MULTI_CPU_NO):
        queue_gpu_to_child = Queue()
        queue_gpu_to_child_list.append(queue_gpu_to_child)
        start_idx =  chunk_size * i
        end_idx = min(chunk_size * (i + 1), total_data_no)
        
        print 'creating a child process[%s] (%s ~ %s)' % (i, start_idx, end_idx-1)
        p = Process(target=predict, args=(gpuid, i, input_data[start_idx : end_idx], 
                                   gtdb[start_idx : end_idx], data_folder, 
                                   data_ext, queue_child_to_gpu, 
                                   queue_gpu_to_child, 
                                   MAX_CAND_AFTER_NMS))
        p.start()
        process_list.append(p)
    
    end_child_no = 0
    ret_list = []
    while True:
        queue_data = queue_child_to_gpu.get()
        if 'box_list_to_save' in queue_data:
            end_child_no += 1
            ret_list.append(queue_data)
            
            print 'get end_child %s' % queue_data['pid']
            if end_child_no == MULTI_CPU_NO:
                break
            else:
                continue

        pid = queue_data['pid']
        blob_data = queue_data['blob_data']
                         
        net.blobs['data'].reshape(*(blob_data.shape))
        
        blobs_out = net.forward(data=blob_data.astype(np.float32, copy=False))
        
        queue_gpu_to_child = queue_gpu_to_child_list[pid]
        queue_gpu_to_child.put(blobs_out)


    total_no_to_find = 0
    total_no_found = 0
            
    for i in range(len(process_list)):
        for ret_value in ret_list:
            if ret_value['pid'] == i:
                total_no_to_find += ret_value['total_no_to_find']
                total_no_found += ret_value['total_no_found']
                total_box_list_to_save.extend(ret_value['box_list_to_save'])
                break

    queue_data = {}
    queue_data['gpuid'] = gpuid

    dump_data = {}
    dump_data['gpuid'] = gpuid
    dump_data['total_no_to_find'] = total_no_to_find
    dump_data['total_no_found'] = total_no_found
    dump_data['box_list_to_save'] = total_box_list_to_save
            
    proposal_file = proposal_folder + '/gpu_result_' + str(gpuid) + '.pickle'
    with open(proposal_file, 'wb') as fid:
        cPickle.dump(dump_data, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote rpn proposal gpu %s to %s' % (gpuid, proposal_file)            
    queue_gpu_to_main.put(queue_data)

def predict(gpuid, pid, input_data, gtdb, data_folder, data_ext, 
            queue_child_to_gpu, queue_gpu_to_child, 
            MAX_CAND_AFTER_NMS):
    MAX_CAND_BEFORE_NMS = 10000
    NMS_THRESH = 0.7
    match_threshold = 0.5
    no = 0
    total_no_to_find = 0
    total_no_found = 0
    box_list_to_save = []
    
    print 'starting child process %s' % pid
    
    for file_name in input_data:
        if ' ' in file_name:
            file_name = file_name.split(' ')[0]
        file_name = file_name.replace('\n', '').replace('\r', '')
        
        if len(file_name) == 0:
            continue

        no += 1
        
        gt_boxes = gtdb[no-1]['boxes']
            
        # if there is no ground truth then ignore this image
        if len(gt_boxes) == 0:
            print 'no labels'
            continue
        
        # DJDJ
        #if file_name != '000009':
        #    continue
        
        #print 'data file : %s' % (data_folder + '/' + file_name)
        
        im = cv2.imread(data_folder + '/' + file_name + '.' + data_ext)
        
        org_img_height = im.shape[0]
        org_img_width = im.shape[1]
        
        blobs = {'data' : None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
        
        queue_data = {}
        queue_data['pid'] = pid
        queue_data['blob_data'] = blobs['data']

        queue_child_to_gpu.put(queue_data)
        blobs_out = queue_gpu_to_child.get()

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
        
        #for pred_box in pred_boxes:
        #    print 'pred_box : %s' % (pred_box, )
            
        #for gt_box in gt_boxes:
        #    print 'gt_box : %s' % (gt_box, )
            
        file_full_path = data_folder + '/' + file_name + '.' + data_ext
        
        no_to_find, no_found = check_match(file_name, file_full_path,
                                                blobs['data'], gt_boxes, 
                                                pred_boxes, match_threshold, 
                                                sorted_scores, rigid_rects)

        no_candidates = len(pred_boxes)
        
        total_no_to_find += no_to_find
        total_no_found += no_found
         
        print 'gpu pid [%s][%s][%s/%s]' % (gpuid, pid, no, len(input_data))  
        
        """ 
        print 'gpu pid [%s][%s][%s/%s] accuracy : %.3f' % (gpuid, pid, no, 
                                                      len(input_data), 
                                                      float(total_no_found) / float(total_no_to_find))  
        """
        
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

    queue_data = {}
    queue_data['pid'] = pid
    queue_data['total_no_to_find'] = total_no_to_find
    queue_data['total_no_found'] = total_no_found
    queue_data['box_list_to_save'] = box_list_to_save
    queue_child_to_gpu.put(queue_data)    

    print 'finishing child process %s' % pid

def check_match(file_name, file_full_path, im_blob, ground_rects, pred_rects, match_threshold, scores, rigid_rects):
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

    #print '%s out of %s found using %s candidates. %s.jpg' %(no_found, no_to_find, len(pred_rects), file_name)

    if False:
    #if True and file_name == 'ILSVRC2012_val_00000096':
        im = cv2.imread(file_full_path)
        im = im[:, :, (2, 1, 0)]
        
        plt.imshow(im)
        ground_rects = np.asarray(ground_rects, np.float)
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
        
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate RPN detections')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default='0', type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--prototxt', dest='prototxt',
                        help='prototxt to use',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--data_type', dest='data_type',
                        help='data_type(trainval or test)',
                        default=None, type=str)
    parser.add_argument('--data_list', dest='data_list',
                        help='data list file',
                        default=None, type=str)
    parser.add_argument('--data_folder', dest='data_folder',
                        help='data folder',
                        default=None, type=str)
    parser.add_argument('--data_ext', dest='data_ext',
                        help='data extension(jpg, JPEG)',
                        default=None, type=str)
    parser.add_argument('--gt', dest='gt',
                        help='ground truth rdb pickle file',
                        default=None, type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default='vgg_cnn_m_1024', type=str)
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
    MULTI_CPU_NO = 3
    
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)
    
    if args.gpu_id is not None:
        gpu_id_list = []
        for gpu_id in args.gpu_id.split(','):
            gpu_id_list.append(int(gpu_id.strip()))
    else:
        gpu_id_list = None
    
    prevent_sleep()

    step = 'step_%s' % args.step
    caffemodel = args.pretrained_model
    MAX_CAND_AFTER_NMS = args.max_output
    prototxt = args.prototxt
    model_name = args.model_name
    data_list = args.data_list
    data_folder = args.data_folder
    data_ext = args.data_ext
    

    if 'voc_2007' in args.imdb_name:
        if args.data_type == 'trainval':
            gt = 'data/cache/voc_2007_trainval_gt_roidb.pkl'
            data_list = '/home/nvidia/www/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
            test_data = 'voc_2007_trainval'
        elif args.data_type == 'test':
            gt = 'data/cache/voc_2007_test_gt_roidb.pkl'
            data_list = '/home/nvidia/www/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
            test_data = 'voc_2007_test'
        data_folder = '/home/nvidia/www/data/VOCdevkit/VOC2007/JPEGImages/'
        data_ext = 'jpg'
    elif 'imagenet' in args.imdb_name:
        if args.data_type == 'train' or args.data_type == 'trainval' :
            gt = 'data/cache/imagenet_train_gt_roidb.pkl'
            data_folder = '/home/nvidia/www/data/ilsvrc14/ILSVRC2014_DET_train/ILSVRC2014_DET_train_all_data'
            test_data = 'imagenet_train'
        elif args.data_type == 'test':
            gt = 'data/cache/imagenet_val_gt_roidb.pkl'
            data_folder = '/home/nvidia/www/data/ilsvrc14/ILSVRC2013_DET_val'
            test_data = 'imagenet_val'            
        data_ext = 'JPEG'
            
    print 'using gt : %s' % gt
    

    gogo(MAX_CAND_AFTER_NMS, gpu_id_list, MULTI_CPU_NO, 
          data_folder, data_ext, prototxt, 
          caffemodel, gt, data_list, test_data, model_name, step)