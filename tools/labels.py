import xml.etree.ElementTree as ET
import os
from os.path import isfile
import cPickle
import traceback
from util_detect import iou
from scipy.io import loadmat

def check_match(file_name, ground_rects, pred_rects, match_threshold):
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

    print '%s out of %s found using %s candidates. %s' %(no_found, no_to_find, len(pred_rects), file_name)

    return no_to_find, no_found

def parse_label_one_folder(folder, gt_list, labels):
        
    # DJDJ
    #if len(labels) > 100:
    #    return

    for file_name in os.listdir(folder):
        if isfile(folder + '/' + file_name) == False:
            parse_label_one_folder(folder + '/' + file_name, gt_list, labels)
            
        file_path = folder + '/' + file_name
        
        output_list = parse_label_one_file(file_path, file_name, gt_list)
        
        if output_list == None:
            continue
        
        labels[file_name] = output_list

        if len(labels) % 1000 == 0:
            print '%s label files done' % len(labels)

def parse_label_one_file(file_path, file_name, gt_list):
    if file_path[-4:] != ".xml" or (file_name[:-4] in gt_list) == False:
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
        return labels
    
    labels = {}
    gt_list = []
    
    if gt_file_list != None:
        with open(gt_file_list, 'rb') as f:
            for line in f.readlines():
                gt_list.append(line.split('\n')[0])
            
    parse_label_one_folder(folder, gt_list, labels)
    
    with open(output_pickle, 'wb') as fid:
        cPickle.dump(labels, fid)

    print 'total %s label files done' % len(labels)
            
    return labels

if __name__ == '__main__':
    match_threshold = 0.5
    
    prediction_mat = 'E:/project/fast-rcnn/data/selective_search_data/voc_2007_trainval.mat'
    gt_folder = 'E:/data/VOCdevkit/VOC2007/Annotations'
    gt_file_list = 'E:/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
    gt_output_pickle = 'E:/voc2007_trainval_labels.pkl'
    img_folder = 'E:/data/VOCdevkit/VOC2007/JPEGImages'
    img_extension = 'jpg'
    
    #gt_folder = 'E:/tmp/ILSVRC2013_DET_bbox_train'
    #gt_output_pickle = 'E:/ILSVRC2013_DET_labels.pkl'

    #gt_folder = 'E:/data/ilsvrc14/ILSVRC2014_DET_bbox_train'
    #gt_output_pickle = 'E:/ILSVRC2014_DET_labels.pkl'
    #img_folder = 'E:/data/ilsvrc14/ILSVRC2014_DET_train/ILSVRC2014_train_all_data'
    #img_extension = 'JPEG'

    
    labels = parse_label_files(gt_folder, gt_file_list, gt_output_pickle)
    label_count = {}
    total_label_count = 0
    missing_file_count = 0
    
    prediction = loadmat(prediction_mat)
    pred_images = prediction['images']
    pred_boxes = prediction['boxes']
    if len(pred_images) == 1:
        pred_images = prediction['images'][0]
    if len(pred_boxes) == 1:
        pred_boxes = prediction['boxes'][0]
    preds = {}
    for image, box in zip(pred_images, pred_boxes):
        if len(image) == 1:
            image = image[0]
        if len(box) == 1:
            box = box[0]
            
        image_name = image[0].encode('ascii','ignore')
        preds[image_name] = box
     
    for key, value in labels.iteritems():
        img_file = img_folder + '/' + key[:-3] + img_extension
        if isfile(img_file) == False:
            print '%s does not exist' % key
            missing_file_count += 1
        
        ground_rects = []
        for label in value:
            label_name, xmin, ymin, xmax, ymax = label
            if (label_name in label_count) == False:
                label_count[label_name] = 0 
            label_count[label_name] += 1
            total_label_count += 1
            
            ground_rects.append((xmin, ymin, xmax, ymax))
        
        pred_rects = preds[key[:-4]]
        check_match(img_file, ground_rects, pred_rects, match_threshold)
    
    print 'file : %s, label : %s, category : %s' % (len(labels), total_label_count, len(label_count))
    print 'missing_file_count : %s' % missing_file_count
    
    
    