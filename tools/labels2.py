import xml.etree.ElementTree as ET
import os
from os.path import isfile
import cPickle
import traceback
from util_detect import iou
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt
import numpy as np
import leveldb

def check_match(file_name, ground_rects, ground_classes, 
                class_no_to_find, class_no_to_found,
                pred_rects, match_threshold):
    found_rects = []
    
    no_to_find = len(ground_rects)
    no_found = 0
    
    ground_rects = np.array(ground_rects).astype(np.int)
    pred_rects = np.array(pred_rects).astype(np.int)
    
    #print("number of ground_rects {}".format(no_to_find))
            
    for i, ground_rect in enumerate(ground_rects):
        max_overlap = 0
        max_overlap_rect = None 
        class_no_to_find[ground_classes[i]-1] += 1
        
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
            class_no_to_found[ground_classes[i]-1] += 1
            
            found_rects.append(max_overlap_rect)

    if False:
    #if True:
        im = cv2.imread(file_name)
        im = im[:, :, (2, 1, 0)]
        
        
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
            
        #plt.show(block=False)
        #raw_input("")
        #plt.close()
        plt.show()
        
        
        
        """
        for pred_rect in pred_rects:
            plt.imshow(im)
            for ground_rect in ground_rects: 
                plt.gca().add_patch(
                    plt.Rectangle((ground_rect[0], ground_rect[1]), ground_rect[2] - ground_rect[0],
                                  ground_rect[3] - ground_rect[1], fill=False,
                                  edgecolor='b', linewidth=3)
                    )
            plt.gca().add_patch(
                plt.Rectangle((pred_rect[0], pred_rect[1]), pred_rect[2] - pred_rect[0],
                              pred_rect[3] - pred_rect[1], fill=False,
                              edgecolor='r', linewidth=3)
                )
            
            plt.show(block=False)
            raw_input("")
            plt.close()
        """

    return no_to_find, no_found

if __name__ == '__main__':
    match_threshold = 0.5
    
    #candidate_db = '/home/dj/big/workspace/fast-rcnn/output/rpn_data/voc_2007_trainval/vgg_cnn_m_1024_step_1_rpn_top_2300_candidate_db/'
    #gt_file = '/home/dj/big/workspace/fast-rcnn/data/cache/voc_2007_trainval_gt_roidb.pkl'

    """
    img_folder = '/home/dj/data/ilsvrc14/ILSVRC2014_DET_train/'
    candidate_db = '/home/dj/big/workspace/fast-rcnn/output/rpn_data/imagenet_train/vgg16_step_1_rpn_top_2300_candidate_db/'
    gt_file = '/home/dj/big/workspace/fast-rcnn/data/cache/imagenet_train_gt_roidb.pkl'
    img_extension = 'JPEG'
    """

    """
    img_folder = '/home/dj/data/ilsvrc14/ILSVRC2013_DET_val/'
    candidate_db = '/home/dj/big/workspace/fast-rcnn/output/rpn_data/imagenet_val/vgg16_step_1_rpn_top_2300_candidate_db_backup'
    gt_file = '/home/dj/big/workspace/fast-rcnn/data/cache/imagenet_val_gt_roidb.pkl'
    img_extension = 'JPEG'
    """

    img_folder = '/home/dj/data/ilsvrc14/ILSVRC2013_DET_val/'
    candidate_db = '/home/dj/big/workspace/fast-rcnn/data/selective_search_data/imagenet_val_db/'
    gt_file = '/home/dj/big/workspace/fast-rcnn/data/cache/imagenet_val_gt_roidb.pkl'
    img_extension = 'JPEG'
    
    
    db = leveldb.LevelDB(candidate_db)
    
    total_no_to_find = 0
    total_no_found = 0
    class_no_to_find = [0] * 200
    class_no_found = [0] * 200
    
    with open(gt_file, 'rb') as f:
        gtdb = cPickle.load(f)
    
    gt_rect_list = {}
    gt_class_list = {}
    for one_gt in gtdb:
        label_file = one_gt['label_file']
        key = label_file.split('.')[0]
        gt_rect_list[key] = one_gt['gt_boxes']
        gt_class_list[key] = one_gt['gt_classes']
        
    i = 0
    for key, value in db.RangeIter():
        candidates = cPickle.loads(value)
        if (key in gt_rect_list) == False:
            continue
        
        gt_rect = gt_rect_list[key]
        gt_class = gt_class_list[key]
        img_file = img_folder + '/' + key + '.' + img_extension
        
        no_to_find, no_found = check_match(img_file, gt_rect, gt_class, 
                                           class_no_to_find, class_no_found,
                                           candidates, match_threshold)
        total_no_to_find += no_to_find
        total_no_found += no_found
        
        print '[%s] %s out of %s (%s candidates).  acc=%.3f' % (i, no_found, no_to_find, 
                                                                              len(candidates), 
                                                                                  (float(total_no_found) / float(total_no_to_find)))
        i += 1
            
    print 'accuracy per class'
    for i in xrange(200):
        if class_no_to_find[i] > 0:
            print '%.3f' % (float(class_no_found[i]) / float(class_no_to_find[i]))
        else:
            print 'nan'
        
    print 'actual found : %s, to_find : %s' % (total_no_found, total_no_to_find)
    print 'accuracy : %.3f' % (float(total_no_found) / float(total_no_to_find))

    
    