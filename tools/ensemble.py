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
    imdb_name = 'imagenet_val'
    base_dir = '/home/dj/big/workspace/fast-rcnn/output/ensemble/'
    data = []
    
    # vgg16 frcnn    
    data.append('vgg16_imagenet_fast_rcnn_with_ss_iter_500000')

    # vgg16 step 2    
    data.append('vgg16_imagenet_fast_rcnn_step2_with_rpn_iter_520000')

    # vgg16 step 2 again    
    #data.append('vgg16_imagenet_fast_rcnn2_with_rpn_iter_xx0000')

    # vgg16 step 4    
    data.append('vgg16_imagenet_fast_rcnn_step4_with_rpn_iter_360000')

    # vgg19 frcnn    
    data.append('vgg19_imagenet_fast_rcnn_with_ss_iter_470000')

    # googlenet frcnn    
    #data.append('googlenet_imagenet_fast_rcnn_ss_iter_480000')

    # vgg16 step 3
    data.append('vgg16_imagenet_fast_rcnn_step2_with_rpn_step3_iter_520000')
    
    output_dir = '%s/results' % base_dir 
    
    i = 0
    all_boxes = None
    
    for one_data in data:
        det_file = base_dir + one_data + '/val/detections.pkl'
        
        i += 1
        print '[%s] processing %s' % (i, one_data)
        
        with open(det_file, 'rb') as f:
            det = cPickle.load(f)

            num_classes = len(det)
            num_images = len(det[0])

            # all_boxes[cls][image] = N x 5 array of detections in (x1, y1, x2, y2, score)
            if all_boxes == None:
                all_boxes = [[[] for _ in xrange(num_images)]
                             for _ in xrange(num_classes)]
            
            for cls_no in xrange(num_classes):
                for img_no in xrange(num_images):
                    det_value = det[cls_no][img_no]

                    if len(det_value) > 0:
                        inds = np.where((det_value[:, 4] > thresh))[0]
                        det_value = det_value[inds]
                    
                    if len(all_boxes[cls_no][img_no]) == 0:
                        all_boxes[cls_no][img_no] = det_value
                    else:
                        all_boxes[cls_no][img_no] = np.vstack((all_boxes[cls_no][img_no], det_value))

    print ''
    print 'Applying NMS to all detections'
    nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)
    
    all_boxes = None

    print 'Evaluating detections'
    imdb = get_imdb(imdb_name)    
    imdb.evaluate_detections(nms_dets, output_dir)
    

if __name__ == '__main__':
    gogo()