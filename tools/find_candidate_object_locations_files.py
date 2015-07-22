import _init_paths
import os
import cv2
from skimage import io
import numpy as np
from caffe.proto import caffe_pb2
from util_detect import iou
import matplotlib.pyplot as plt
import caffe
from labels import read_label_file
from fast_rcnn.test import _get_image_blob
from utils.model import last_conv_size
from fast_rcnn.config import cfg

class Detector(object):
    
    def parse_label(self, label):
        #0:(104,78,375,183)-0:(133,88,197,123)-14:(195,180,213,229)-14:(26,189,44,238)
        
        class_list = []
        rect_list = []
        rect_resized_list = []
        label, label_resized = label.split('|')

        for one_label in label.split('-'):
            data_class, data_rect = one_label.split(':')
            point = data_rect.strip('(').strip(')').split(',')
            point = map(int, point)
            
            class_list.append(data_class)
            class_list = map(int, class_list)
            rect_list.append(Rect(Point(point[0], point[1]), Point(point[2], point[3])))

        for one_label in label_resized.split('-'):
            data_class, data_rect = one_label.split(':')
            point = data_rect.strip('(').strip(')').split(',')
            point = map(int, point)
            rect_resized_list.append(Rect(Point(point[0], point[1]), Point(point[2], point[3])))

        return class_list, rect_list, rect_resized_list
    
    def check_match(self, img, ground_rects, match_threshold, min_size):
        rects = []
        found_rects = []
        
        dlib.find_candidate_object_locations(img, rects, min_size=min_size)
        
        no_to_find = len(ground_rects)
        no_found = 0
        
        #print("number of rectangles found {}".format(len(rects)))
        #print("number of ground_rects {}".format(no_to_find))
                
        for ground_rect in ground_rects: 
            for rect in rects:
                if iou(rect.left(), rect.top(), rect.right(), rect.bottom(), 
                            ground_rect.left, ground_rect.top, ground_rect.right, ground_rect.bottom) >= match_threshold:
                    no_found += 1
                    
                    rec = dlib.rectangle(left = rect.left(), top = rect.top(), right = rect.right(), bottom = rect.bottom())
                    found_rects.append(rec)
                    break

        if no_to_find != no_found:
            print '%s out of %s found using %s candidates' %(no_found, no_to_find, len(rects))

            win = dlib.image_window()
            win.clear_overlay()
            win.set_image(img)
    
            for ground_rect in ground_rects: 
                rec = dlib.rectangle(left = ground_rect.left, top = ground_rect.top, right = ground_rect.right, bottom = ground_rect.bottom)
                win.add_overlay(rec, color=dlib.rgb_pixel(0, 0, 255))
    
            for found_rect in found_rects: 
                win.add_overlay(found_rect, color=dlib.rgb_pixel(255, 0, 0))
                
            dlib.hit_enter_to_continue()
                            
        return no_to_find, no_found, len(rects)

    def get_img_rect(self, img_height, img_width, conv_height, conv_width, axis1, axis2, axis3):

        anchors = [[128*2, 128*1], [128*1, 128*1], [128*1, 128*2], 
                   [256*2, 256*1], [256*1, 256*1], [256*1, 256*2], 
                   [512*2, 512*1], [512*1, 512*1], [512*1, 512*2]]
                
        img_center_x = img_width * axis3 / conv_width
        img_center_y = img_height * axis2 / conv_height
        anchor_size = anchors[axis1]
        img_x1 = img_center_x - anchor_size[0] / 2 
        img_x2 = img_center_x + anchor_size[0] / 2 
        img_y1 = img_center_y - anchor_size[1] / 2 
        img_y2 = img_center_y + anchor_size[1] / 2 
        
        return (img_x1, img_y1, img_x2, img_y2)

    def display(self, im_blob, proposal_rects):
        import matplotlib.pyplot as plt
        
        im = im_blob[0, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        plt.imshow(im)
        for proposal_rect in proposal_rects:
            plt.gca().add_patch(
                plt.Rectangle((proposal_rect[0], proposal_rect[1]), proposal_rect[2] - proposal_rect[0],
                              proposal_rect[3] - proposal_rect[1], fill=False,
                              edgecolor='r', linewidth=3)
                )
        plt.show()


    def gogo(self, voc_base_folder, prototxt, caffemodel):
        match_threshold = 0.5
        min_size = 1
        
        label_folder = voc_base_folder + '/Annotations'
        image_folder = voc_base_folder + '/JPEGImages'
        
        #labels = read_label_file(label_folder)
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        
        no = 0
        total_no_to_find = 0
        total_no_found = 0
        data = []
        for index, file_name in enumerate(os.listdir(image_folder)):

            no += 1
            
            if file_name != '000012.jpg':
                continue
            
            im = cv2.imread(image_folder + '/' + file_name)
            
            blobs = {'data' : None}
            blobs['data'], im_scale_factors = _get_image_blob(im)
            
            net.blobs['data'].reshape(*(blobs['data'].shape))
            blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False))

            cls_pred = blobs_out['cls_pred']
            
            pos_pred = cls_pred[:, :, 1, :, :]
            sorted_pred = np.argsort(pos_pred, axis=None)[::-1]
            height = pos_pred.shape[2]
            width = pos_pred.shape[3]
            proposal_rects = []
            
            for i in range(100):
                top_index = sorted_pred[i]
                axis1 = top_index / height / width
                axis2 = top_index / width % height
                axis3 = top_index % width
                
                img_height = blobs['data'].shape[2]
                img_width = blobs['data'].shape[3]
                conv_height, scale_height = last_conv_size(img_height)
                conv_width, scale_width = last_conv_size(img_width)
                
                proposal_rect = self.get_img_rect(img_height, img_width, conv_height, conv_width, axis1, axis2, axis3)
                proposal_rects.append(proposal_rect)
                
                print pos_pred[0, axis1, axis2, axis3]
                print 'axis : (0, %s, %s, %s)' % (axis1, axis2, axis3)
                print 'proposal_rect : %s' % (proposal_rect, )
                
            self.display(blobs['data'], proposal_rects)
            
            box_deltas = blobs_out['bbox_pred']
            pred_boxes = _bbox_pred(boxes, box_deltas)
            pred_boxes = _clip_boxes(pred_boxes, im.shape)            
            
            
            class_list, rect_list, rect_resized_list = self.parse_label(label)
            
            no_to_find, no_found, no_candidates = self.check_match(new_img, rect_list, match_threshold, min_size)
            
            total_no_to_find += no_to_find
            total_no_found += no_found
             
            print '[%s] %s out of %s found using %s candidates' %(no, no_found, no_to_find, no_candidates)
            
            if no == 100:
                break
                
        print 'accuracy : %.3f' % (float(total_no_found) / float(total_no_to_find))  
                

if __name__ == '__main__':
    voc_base_folder = 'E:/data/VOCdevkit/VOC2007/'
    prototxt = 'E:/project/fast-rcnn/models/VGG_CNN_M_1024/rpn/test.prototxt'

    #caffemodel = 'E:/project/fast-rcnn/output/faster_rcnn_cls_only/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_100.caffemodel'
    #caffemodel = 'E:/project/fast-rcnn/output/faster_rcnn_bbox_only/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_100.caffemodel'
    caffemodel = 'E:/project/fast-rcnn/output/faster_rcnn/voc_2007_trainval/vgg_cnn_m_1024_rpn_iter_1000.caffemodel'
    
    detector = Detector()
    detector.gogo(voc_base_folder, prototxt, caffemodel)