import _init_paths
import os
import cv2
import cPickle
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

    def check_match(self, im_blob, ground_rects, pred_rects, match_threshold):
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

        #if no_to_find != no_found:
        if True:
            print '%s out of %s found using %s candidates' %(no_found, no_to_find, len(pred_rects))

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
                
            """
            for pred_box in pred_boxes:
                plt.gca().add_patch(
                    plt.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0],
                                  pred_box[3] - pred_box[1], fill=False,
                                  edgecolor='g', linewidth=3)
                    )
            """
                
            plt.show()

        return no_to_find, no_found
    
    def gogo(self, voc_base_folder, prototxt, caffemodel, gt, data_list):
        TOP_N = 300
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
            
            
            #if file_name != '000009' and file_name != '000017':
            #    continue
            
            
            im = cv2.imread(image_folder + '/' + file_name + '.jpg')
            
            blobs = {'data' : None}
            blobs['data'], im_scale_factors = _get_image_blob(im)
            
            net.blobs['data'].reshape(*(blobs['data'].shape))
            blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False))

            cls_pred = blobs_out['cls_pred']
            box_deltas = blobs_out['bbox_pred']
            
            pos_pred = cls_pred[:, :, 1, :, :]
            sorted_pred = np.argsort(pos_pred, axis=None)[::-1]
            height = pos_pred.shape[2]
            width = pos_pred.shape[3]
            proposal_rects = []

            top_index = sorted_pred[:TOP_N]
            axis1 = top_index / height / width
            axis2 = top_index / width % height
            axis3 = top_index % width
            
            img_height = blobs['data'].shape[2]
            img_width = blobs['data'].shape[3]
            conv_height, scale_height = last_conv_size(img_height)
            conv_width, scale_width = last_conv_size(img_width)



            """
            # DJDJ
            top_index = np.array([6510])
            axis1 = np.array([3])
            axis2 = np.array([10])
            axis3 = np.array([20])
            pred = box_deltas[0, axis1[0]*4:axis1[0]*4+4, axis2[0], axis3[0]]
            """
            
            
            proposal_rects = self.get_img_rect(img_height, img_width, conv_height, conv_width, axis1, axis2, axis3)
            
            """
            print ''
            print 'proposal_rects[0] : %s' % proposal_rects[0]            
            print 'pred[%s] : %s' % (0, pred)
            """

            box_deltas_rects = np.zeros((len(axis1), 4), np.float32)
            box_deltas_rects[:, 0] = box_deltas[0, axis1*4, axis2, axis3]
            box_deltas_rects[:, 1] = box_deltas[0, axis1*4+1, axis2, axis3]
            box_deltas_rects[:, 2] = box_deltas[0, axis1*4+2, axis2, axis3]
            box_deltas_rects[:, 3] = box_deltas[0, axis1*4+3, axis2, axis3]

            pred_boxes = _bbox_pred(proposal_rects, box_deltas_rects)
            pred_boxes = _clip_boxes(pred_boxes, (img_height, img_width))            
            
            gt_boxes = gtdb[no-1]['boxes'] * im_scale_factors
            
            #for pred_box in pred_boxes:
            #    print 'pred_box : %s' % (pred_box, )
                
            for gt_box in gt_boxes:
                print 'gt_box : %s' % (gt_box, )
                
            no_to_find, no_found = self.check_match(blobs['data'], gt_boxes, pred_boxes, match_threshold)
            
            no_candidates = len(pred_boxes)
            
            total_no_to_find += no_to_find
            total_no_found += no_found
             
            #print '[%s] %s out of %s found using %s candidates' %(no, no_found, no_to_find, no_candidates)
            
            #if no == 100:
            #    break
                
        print 'accuracy : %.3f' % (float(total_no_found) / float(total_no_to_find))  
                

if __name__ == '__main__':
    voc_base_folder = 'E:/data/VOCdevkit2/VOC2007/'
    prototxt = 'E:/project/fast-rcnn/models/VGG_CNN_M_1024/rpn/test.prototxt'

    #caffemodel = 'E:/project/fast-rcnn/output/faster_rcnn_cls_only/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_100.caffemodel'
    #caffemodel = 'E:/project/fast-rcnn/output/faster_rcnn_bbox_only/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_100.caffemodel'

    caffemodel = 'E:/project/fast-rcnn/output/faster_rcnn/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_1000.caffemodel'
    data_list = voc_base_folder + '/ImageSets/Main/trainval.txt'
    gt = 'E:/project/fast-rcnn/data/cache/voc_2007_trainval_gt_roidb.pkl'
    
    cfg_file = 'E:/project/fast-rcnn/experiments/cfgs/faster_rcnn.yml'
    cfg_from_file(cfg_file)

    detector = Detector()
    detector.gogo(voc_base_folder, prototxt, caffemodel, gt, data_list)