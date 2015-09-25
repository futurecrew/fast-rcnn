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

def check_match(file_name, ground_rects, pred_rects, match_threshold):
    found_rects = []
    
    no_to_find = len(ground_rects)
    no_found = 0
    
    ground_rects = np.array(ground_rects).astype(np.int)
    pred_rects = np.array(pred_rects).astype(np.int)
    
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
            
        plt.show(block=False)
        raw_input("")
        plt.close()
        
        
        
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

def parse_label_one_folder(folder, gt_list, labels):
        
    # DJDJ
    #if len(labels) > 100:
    #    return
    zero_label = 0
    for file_name in os.listdir(folder):
        if isfile(folder + '/' + file_name) == False:
            zero_label += parse_label_one_folder(folder + '/' + file_name, gt_list, labels)
            
        file_path = folder + '/' + file_name
        
        output_list = parse_label_one_file(file_path, file_name, gt_list)
        
        if output_list == None:
            continue
        
        if len(output_list) == 0:
            zero_label += 1
        
        labels[file_name] = output_list

        if len(labels) % 1000 == 0:
            print '%s label files done' % len(labels)
    return zero_label

def parse_label_one_file(file_path, file_name, gt_list):
    if file_path[-4:] != ".xml" or (len(gt_list) != 0 and (file_name[:-4] in gt_list) == False):
        return None
        
    with open(file_path) as f:
        content = f.read()

    output_list = []
    try:
        root = ET.fromstring(content)
        label_string = ''
        label_resized_string = ''
        
        image_width = root.find('size').find('width').text
        image_height = root.find('size').find('height').text
    
        for object in root.findall('object'):
            label_name = object.find('name').text
            xmin = float(object.find('bndbox').find('xmin').text)
            ymin = float(object.find('bndbox').find('ymin').text)
            xmax = float(object.find('bndbox').find('xmax').text)
            ymax = float(object.find('bndbox').find('ymax').text)
    
            output_list.append((label_name, xmin, ymin, xmax, ymax))
    except:
        print traceback.format_exc()

    return output_list

def parse_label_files(folder, gt_file_list, output_pickle):
    
    if os.path.exists(output_pickle):
        with open(output_pickle, 'rb') as fid:
            labels = cPickle.load(fid)
        print 'Labels loaded from {}'.format(output_pickle)
        return labels, 0
    
    labels = {}
    gt_list = []
    
    if gt_file_list != None:
        with open(gt_file_list, 'rb') as f:
            for line in f.readlines():
                gt_list.append(line.split('\n')[0])
            
    zero_label = parse_label_one_folder(folder, gt_list, labels)
    
    with open(output_pickle, 'wb') as fid:
        cPickle.dump(labels, fid)

    print 'total %s label files done' % len(labels)
            
    return labels, zero_label

if __name__ == '__main__':
    match_threshold = 0.5
    

    #prediction_mat = '/home/nvidia/www/workspace/fast-rcnn/output/ss/ss_output.mat'
    #gt_file_list = '/home/nvidia/www/data/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
    #gt_output_pickle = '/home/nvidia/www/voc2007_train_labels.pkl'

    #prediction_mat = 'E:/project/fast-rcnn/data/selective_search_data/voc_2007_trainval.mat'
    prediction_mat = '/home/nvidia/www/workspace/fast-rcnn/output/ss/ss_output.mat'
    gt_file_list = '/home/nvidia/www/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
    gt_output_pickle = '/home/nvidia/www/voc2007_trainval_labels.pkl'

    gt_folder = '/home/nvidia/www/data/VOCdevkit/VOC2007/Annotations'
    img_folder = '/home/nvidia/www/data/VOCdevkit/VOC2007/JPEGImages'
    img_extension = 'jpg'

    
    """
    #gt_folder = 'E:/tmp/ILSVRC2013_DET_bbox_train'
    #gt_output_pickle = 'E:/ILSVRC2013_DET_labels.pkl'

    prediction_mat = None
    gt_folder = 'E:/data/ilsvrc14/ILSVRC2014_DET_bbox_train/ILSVRC2014_DET_bbox_train_all_data'
    gt_file_list = None
    gt_output_pickle = 'E:/ILSVRC2014_DET_labels.pkl'
    img_folder = 'E:/data/ilsvrc14/ILSVRC2014_DET_train/ILSVRC2014_train_all_data'
    img_extension = 'JPEG'
    """
    
    labels, zero_label = parse_label_files(gt_folder, gt_file_list, gt_output_pickle)
    label_count = {}
    total_label_count = 0
    missing_file_count = 0
    
    print 'file : %s, zero_label : %s' % (len(labels), zero_label)
    
    if prediction_mat != None:
        prediction = loadmat(prediction_mat)
        
        pred_images = prediction['images']
        pred_boxes = prediction['boxes']
        if len(pred_images) == 1:
            pred_images = prediction['images'][0]
        if len(pred_boxes) == 1:
            pred_boxes = prediction['boxes'][0]
        preds = {}
        for image, box in zip(pred_images, pred_boxes):
            if isinstance(image, np.ndarray):
                image = image[0]
            if isinstance(image, np.ndarray):
                image = image[0]
            if len(box) == 1:
                box = box[0]

            image_name = image.encode('ascii','ignore')
            
            preds[image_name] = box[:, (1, 0, 3, 2)]
        
        total_no_to_find = 0
        total_no_found = 0
        total_pred_rects = 0
        i = 0
        for key, value in labels.iteritems():
            file_name = key[:-3] + img_extension
            img_file = img_folder + '/' + file_name
            if isfile(img_file) == False:
                print '%s does not exist' % key
                missing_file_count += 1
            
            i += 1
            ground_rects = []
            for label in value:
                label_name, xmin, ymin, xmax, ymax = label
                if (label_name in label_count) == False:
                    label_count[label_name] = 0 
                label_count[label_name] += 1
                total_label_count += 1
                
                ground_rects.append((xmin, ymin, xmax, ymax))
            
            pred_rects = preds[key[:-4]]
            no_to_find, no_found = check_match(img_file, ground_rects, pred_rects, match_threshold)
            total_no_to_find += no_to_find
            total_no_found += no_found
            total_pred_rects += len(pred_rects)
            
            print '[%s] %s out of %s found using %s candidates. %s. total acc=%.3f' % (i, no_found, no_to_find, 
                                                                                  len(pred_rects), file_name, 
                                                                                  (float(total_no_found) / float(total_no_to_find)))
            
        print 'file : %s, label : %s, category : %s' % (len(labels), total_label_count, len(label_count))
        print 'actual found : %s, to_find : %s' % (total_no_found, total_no_to_find)
        print 'accuracy : %.3f' % (float(total_no_found) / float(total_no_to_find))
        print 'total_pred_rects : %s' % total_pred_rects
        print 'missing_file_count : %s' % missing_file_count
    
    
    