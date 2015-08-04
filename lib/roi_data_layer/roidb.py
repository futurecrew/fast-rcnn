# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from fast_rcnn.config import cfg
import utils.cython_bbox

import cv2
from utils.cython_bbox import bbox_overlaps
from utils.blob import im_scale_after_resize
from utils.model import last_conv_size

def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

def prepare_roidb_rpn(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb

    anchors = [[128*2, 128*1], [128*1, 128*1], [128*1, 128*2], 
               [256*2, 256*1], [256*1, 256*1], [256*1, 256*2], 
               [512*2, 512*1], [512*1, 512*1], [512*1, 512*2]]
    
    
    for i in xrange(len(imdb.image_index)):
        image_path = imdb.image_path_at(i)
        roidb[i]['image'] = image_path
        gt_boxes = roidb[i]['boxes']
        gt_classes = roidb[i]['gt_classes']
    
        im = cv2.imread(image_path)
        resize_scale = im_scale_after_resize(im, cfg.TRAIN.SCALES[0], cfg.TRAIN.MAX_SIZE)
        
        # Generate anchors based on the resized image
        im_height = int(im.shape[0] * resize_scale)
        im_width = int(im.shape[1] * resize_scale)
        
        conv_height, scale_height = last_conv_size(im_height, cfg.MODEL_NAME)
        conv_width, scale_width = last_conv_size(im_width, cfg.MODEL_NAME)
        
        resized_gt_boxes = gt_boxes * resize_scale
        
        labels = np.zeros((9 * conv_height * conv_width), dtype=np.int16)
        labels.fill(-1)
        
        # indexes for ground true rectangles
        gt_indexes = np.zeros((9 * conv_height * conv_width), dtype=np.int16)
        gt_indexes.fill(-1)

        rois = np.zeros((9 * conv_height * conv_width, 4), dtype=np.float16)
        gt_rois = np.zeros((9 * conv_height * conv_width, 4), dtype=np.float16)
        boxes = np.zeros((9, 4), dtype=np.int32)
        
        gt_no = len(gt_boxes)
        max_of_maxes = np.zeros((gt_no))
        max_anchors = np.zeros((gt_no))
        max_ys = np.zeros((gt_no))
        max_xs = np.zeros((gt_no))
        max_labels = np.zeros((gt_no))
        max_classes = np.zeros((gt_no))
        max_boxes = np.zeros((gt_no, 4))

        
        if i % 100 == 0:
            print 'processing image %s' % i
        
        for center_y in xrange(0, conv_height):
            for center_x in xrange(0, conv_width):
                #print 'processing [%s, %s]' % (center_y, center_x)

                anchor_i = -1
                boxes.fill(-1)
                for anchor_w, anchor_h in anchors:
                    x1 = center_x * scale_width - anchor_w / 2
                    y1 = center_y * scale_height - anchor_h / 2
                    x2 = x1 + anchor_w
                    y2 = y1 + anchor_h
                    
                    anchor_i += 1
                    
                    if x1 < 0 or y1 < 0 or x2 > im_width or y2 > im_height:
                        continue
                    
                    boxes[anchor_i, :] = x1, y1, x2, y2
                    
                    #print '(%s, %s, %s, %s) appended' % (x1, y1, x2, y2)
                
                gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                            resized_gt_boxes.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                
                # For positive train data
                I = np.where(maxes > cfg.TRAIN.FG_THRESH)[0]
                
                if len(I) > 0:
                    # set label to 1 when a box of overlapping area if bigger than FG_THRESH 
                    base_index = I * conv_height * conv_width + center_y * conv_width + center_x
                    labels[base_index] = gt_classes[argmaxes[I]]
                    gt_indexes[base_index] = argmaxes[I]
                    rois[base_index] = boxes[I]

                # For negative train data
                I = np.where(maxes < cfg.TRAIN.BG_THRESH_HI)[0]
                
                if len(I) > 0:
                    # set label to 0 when a box of overlapping area if bigger than FG_THRESH 
                    base_index = I * conv_height * conv_width + center_y * conv_width + center_x
                    labels[base_index] = 0
                    gt_indexes[base_index] = -1


                # Check max overlapping anchor
                argmaxes = gt_overlaps.argmax(axis=0)
                maxes = gt_overlaps.max(axis=0)
                
                for m in range(len(gt_boxes)):
                    if maxes[m] > max_of_maxes[m]:
                        max_of_maxes[m] = maxes[m]
                        max_ys[m] = center_y
                        max_xs[m] = center_x
                        max_classes[m] = gt_classes[m]
                        max_anchors[m] = argmaxes[m]
                        max_boxes[m] = boxes[max_anchors[m]].copy()

                
        # set label to 1 of the anchor which has the biggest overlapping area among all the anchors 
        for m in range(len(gt_boxes)):
            base_index = max_anchors[m] * conv_height * conv_width + max_ys[m] * conv_width + max_xs[m]
            labels[base_index] = max_classes[m]
            gt_indexes[base_index] = m
            rois[base_index] = max_boxes[m]
        
        bbox_targets = _compute_targets_rpn(rois, labels, gt_indexes, resized_gt_boxes)
        
        # convert all the positive data labels to 1 and all the negative 
        pos_index = np.where(labels > 0)[0]
        labels[pos_index] = 1
        
        """
        pos_index = np.where(gt_indexes >= 0)[0]        
        for iii in pos_index:
            print 'label : %s' % (labels[iii]) 
            print 'rois : %s' % (rois[iii])
            print 'bbox_targets : %s' % (bbox_targets[iii])
        """
                    
        #print 'total %s boxes are generated.' % len(box_list)        
        # need gt_overlaps as a dense array for argmax
        #gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        #max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        #max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = labels
        roidb[i]['bbox_targets'] = bbox_targets
        roidb[i]['train_target'] = 'rpn'        
        #roidb[i]['rois'] = rois
        roidb[i]['resized_gt_boxes'] = resized_gt_boxes
        roidb[i]['conv_width'] = conv_width
        roidb[i]['conv_height'] = conv_height
        roidb[i]['conv_scale_width'] = scale_width
        roidb[i]['conv_scale_height'] = scale_height
        
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        #zero_inds = np.where(max_overlaps == 0)[0]
        #assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        #nonzero_inds = np.where(max_overlaps > 0)[0]
        #assert all(max_classes[nonzero_inds] != 0)


def add_bbox_regression_targets(roidb, train_target):
    """Add information needed to train bounding-box regressors."""
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'

    num_images = len(roidb)
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    if train_target == 'rpn':
        # Compute values needed for means and stds
        # var(x) = E(x^2) - E(x)^2
        class_counts = 0
        one_sums = np.zeros((1, 4))
        one_squared_sums = np.zeros((1, 4))
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            cls_inds = np.where(targets[:, 0] > 0)[0]
            if cls_inds.size > 0:
                class_counts += cls_inds.size
                one_sums[0, :] += targets[cls_inds, 1:].sum(axis=0)
                one_squared_sums[0, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0)
    
        one_means = one_sums / class_counts
        one_stds = np.sqrt(one_squared_sums / class_counts - one_means ** 2)
    
        # Normalize targets
        if cfg.TRAIN.NORMALIZE_BBOX:
            for im_i in xrange(num_images):
                targets = roidb[im_i]['bbox_targets']
                cls_inds = np.where(targets[:, 0] > 0)[0]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] -= one_means[0, :]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] /= one_stds[0, :]
            
        means = np.zeros((1, 36))
        stds = np.zeros((1, 36))
        
        for i in range(9):
            means[:, i*4:i*4+4] = one_means
            stds[:, i*4:i*4+4] = one_stds
        
    else:
        # Infer number of classes from the number of columns in gt_overlaps
        for im_i in xrange(num_images):
            rois = roidb[im_i]['boxes']
            max_overlaps = roidb[im_i]['max_overlaps']
            max_classes = roidb[im_i]['max_classes']
            roidb[im_i]['bbox_targets'] = \
                    _compute_targets(rois, max_overlaps, max_classes)
    
        # Compute values needed for means and stds
        # var(x) = E(x^2) - E(x)^2
        class_counts = np.zeros((num_classes, 1)) + cfg.EPS
        sums = np.zeros((num_classes, 4))
        squared_sums = np.zeros((num_classes, 4))
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(1, num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                if cls_inds.size > 0:
                    class_counts[cls] += cls_inds.size
                    sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                    squared_sums[cls, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0)
    
        means = sums / class_counts
        stds = np.sqrt(squared_sums / class_counts - means ** 2)
    
        # Normalize targets
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(1, num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()

def _compute_targets(rois, overlaps, labels):
    """Compute bounding-box regression targets for an image."""
    # Ensure ROIs are floats
    rois = rois.astype(np.float, copy=False)

    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = utils.cython_bbox.bbox_overlaps(rois[ex_inds, :],
                                                     rois[gt_inds, :])

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + cfg.EPS
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + cfg.EPS
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + cfg.EPS
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + cfg.EPS
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1] = targets_dx
    targets[ex_inds, 2] = targets_dy
    targets[ex_inds, 3] = targets_dw
    targets[ex_inds, 4] = targets_dh
    return targets


def _compute_targets_rpn(rois, labels, gt_indexes, gt_boxes):
    """Compute bounding-box regression targets for an image."""
    
    """
    print 'len(np.where(labels > 0) : %s' % len(np.where(labels > 0)[0])
    print 'len(np.where(gt_indexes >= 0) : %s' % len(np.where(gt_indexes >= 0)[0])
    """
    
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(gt_indexes >= 0)[0]

    gt_lables = labels[ex_inds]
    gt_rois = gt_boxes[gt_indexes[ex_inds], :]
    ex_rois = rois[ex_inds, :]
        
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + cfg.EPS
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + cfg.EPS
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    
    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + cfg.EPS
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + cfg.EPS
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    """
    print 'rois[6510] : %s' % rois[6510]
    print 'gt_boxes[0] : %s' % gt_boxes[0]
    
    roi_width_temp = rois[6510][2] - rois[6510][0]
    roi_height_temp = rois[6510][3] - rois[6510][1]
    roi_ctr_x_temp = rois[6510][0] + roi_width_temp / 2
    
    gt_width_temp = gt_boxes[0][2] - gt_boxes[0][0]
    gt_height_temp = gt_boxes[0][3] - gt_boxes[0][1]
    gt_ctr_x_temp = gt_boxes[0][0] + gt_width_temp / 2
    
    print 'gt ctr_x - ex ctr_x : %s' % (gt_ctr_x_temp - roi_ctr_x_temp)
    print 'gt width / ex width : %s' % (gt_width_temp / roi_width_temp)
    """

    if cfg.TRAIN.COMPUTE_LOGISTIC_BBOX_TARGET:
        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = np.log(gt_widths / ex_widths)
        targets_dh = np.log(gt_heights / ex_heights)
    else:
        targets_dx = gt_ctr_x - ex_ctr_x
        targets_dy = gt_ctr_y - ex_ctr_y
        targets_dw = gt_widths / ex_widths
        targets_dh = gt_heights / ex_heights
        
    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = gt_lables
    targets[ex_inds, 1] = targets_dx
    targets[ex_inds, 2] = targets_dy
    targets[ex_inds, 3] = targets_dw
    targets[ex_inds, 4] = targets_dh
    
    return targets
