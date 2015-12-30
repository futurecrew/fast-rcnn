import _init_paths
import numpy as np
import cPickle
from fast_rcnn.test import apply_nms
from datasets.factory import get_imdb
from fast_rcnn.config import cfg


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='imagenet_val', type=str)



def gogo():
    print 'starting ensemble'

    thresh = 0.01
    #data_type = 'val'
    data_type = 'test'
    
    exclude_worst = True

    num_classes = 201
    
    imdb_name = 'imagenet_' + data_type
    base_dir = '/home/dj/big/workspace/fast-rcnn/output/ensemble/'
    data = []
    result = []
    
    # 1. vgg16 frcnn    
    data.append('vgg16_imagenet_fast_rcnn_with_ss_iter_500000')
    result.append('result_comp4-1648.txt')

    # 2. vgg16 step 2_1   
    #data.append('vgg16_imagenet_fast_rcnn_step2_with_rpn_iter_520000')
    #result.append('result_comp4-13436.txt')

    # 3. vgg16 step 2_2    
    #data.append('vgg16_imagenet_fast_rcnn2_step2_with_rpn_iter_520000')
    #result.append('result_comp4-31392.txt')
    
    # (2, 3) vgg16 avg (step 2_1, step 2_2)    
    data.append('vgg16_imagenet_fast_rcnn_avg_2_3')
    result.append('result_comp4-3153.txt')
    
    # (2, 3, 5) vgg16 avg (step 2_1, step 2_2, step 2_4)    
    #data.append('vgg16_imagenet_fast_rcnn_with_rpn_avg_2_3_5')
    #result.append('result_comp4-33831.txt')
    
    # (2, 3, 6) vgg16 avg (step 2_1, step 2_2, step 4)    
    #data.append('vgg16_imagenet_fast_rcnn_avg_2_3_6')
    #result.append('result_comp4-22284.txt')
    
    # (2, 3, 4, 5) vgg16 avg (step 2_1, step 2_2, step 2_3, step 2_4)    
    #data.append('vgg16_imagenet_fast_rcnn_with_rpn_avg_2_3_4_5')
    #result.append('result_comp4-6730.txt')
    
    # (2, 3, 5, 6) vgg16 avg (step 2_1, step 2_2, step 2_4, step 4)    
    #data.append('vgg16_imagenet_fast_rcnn_avg_2_3_5_6')
    #result.append('result_comp4-6456.txt')

    # 6. vgg16 step 4    
    data.append('vgg16_imagenet_fast_rcnn_step4_with_rpn_iter_360000')
    result.append('result_comp4-16205.txt')

    # 7. vgg19 frcnn    
    data.append('vgg19_imagenet_fast_rcnn_with_ss_iter_470000')
    result.append('result_comp4-37160.txt')

    # 8. googlenet frcnn    
    #data.append('googlenet_imagenet_fast_rcnn_with_ss_iter_480000')
    #result.append('result_comp4-42391.txt')

    # 9. vgg16 step 3
    #data.append('vgg16_imagenet_fast_rcnn_step2_with_rpn_step3_iter_520000')
    #result.append('result_comp4-25665.txt')
    
    output_dir = '%s/results' % base_dir 
    
    all_boxes = None

    total_result = np.zeros((num_classes, len(data)))
    data_no = 0
    for one_result, one_data in zip(result, data):
        result_file = base_dir + one_data + '/val/' + one_result
        with open(result_file, 'rt') as f:
            line_no = 0
            for one_line in f.readlines():
                try:
                    one_number = float(one_line.rstrip())
                except:
                    continue
                line_no += 1
                total_result[line_no, data_no] = one_number                 
                if line_no >= num_classes - 1:
                    break
        data_no += 1
        
    min_data_index_per_class = np.argmin(total_result, axis=1)
    
    data_no = 0
    for one_data in data:
        det_file = base_dir + one_data + '/' + data_type + '/detections.pkl'
        if data_type == 'test':
            submission_file = base_dir + one_data + '/' + data_type + '/submission.txt'
        else:
            submission_file = ''
        
        print '[%s] processing %s' % (data_no + 1, one_data)
        
        with open(det_file, 'rb') as f:
            det = cPickle.load(f)

            num_images = len(det[0])

            # all_boxes[cls][image] = N x 5 array of detections in (x1, y1, x2, y2, score)
            if all_boxes == None:
                all_boxes = [[[] for _ in xrange(num_images)]
                             for _ in xrange(num_classes)]
            
            for cls_no in xrange(num_classes):
                if exclude_worst and cls_no > 0 and min_data_index_per_class[cls_no] == data_no:
                    continue
                
                for img_no in xrange(num_images):
                    det_value = det[cls_no][img_no]

                    if len(det_value) > 0:
                        inds = np.where((det_value[:, 4] >= thresh))[0]
                        det_value = det_value[inds]
                    
                    if len(all_boxes[cls_no][img_no]) == 0:
                        all_boxes[cls_no][img_no] = det_value
                    else:
                        all_boxes[cls_no][img_no] = np.vstack((all_boxes[cls_no][img_no], det_value))

        data_no += 1

    print ''
    print 'Applying NMS to all detections'
    nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)
    
    all_boxes = None

    print 'Evaluating detections'
    imdb = get_imdb(imdb_name)    
    imdb.evaluate_detections(nms_dets, output_dir, submission_file)
    

if __name__ == '__main__':
    gogo()