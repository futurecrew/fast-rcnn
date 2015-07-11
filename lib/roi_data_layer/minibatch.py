# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

from utils.model import last_conv_size

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    
    if roidb[0]['proposal'] == 'rpn':
        conv_h, scale_h = last_conv_size(im_blob.shape[2])
        conv_w, scale_w = last_conv_size(im_blob.shape[3])
        
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0, 9, conv_h, conv_w), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 36, conv_h, conv_w), dtype=np.float32)
        bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_loss \
                = _sample_rois_rpn(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes, conv_h, conv_w)
    
            """
            # DJDJ
            labels = np.ones((1, 9, conv_h, conv_w), dtype=np.float32)
            bbox_targets = np.ones((1, 36, conv_h, conv_w), dtype=np.float32)
            bbox_loss = np.ones((1, 36, conv_h, conv_w), dtype=np.float32)
            """

            # Add to RoIs blob
            if im_rois != None:
                batch_ind = im_i * np.ones((im_rois.shape[0], 1))
                rois_blob_this_image = np.hstack((batch_ind, im_rois))
                rois_blob = np.vstack((rois_blob, rois_blob_this_image))
    
            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.vstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_loss_blob = np.vstack((bbox_loss_blob, bbox_loss))
            
        # For debug visualizations
        #_vis_minibatch_rpn(im_blob, rois_blob, labels_blob, roidb, bbox_targets_blob, bbox_loss_blob)
    
        blobs = {'data': im_blob,
                 'labels': labels_blob}            
    else:
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        # all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_loss \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes)
    
            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))
    
            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_loss_blob = np.vstack((bbox_loss_blob, bbox_loss))
            # all_overlaps = np.hstack((all_overlaps, overlaps))
        
        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)
    
        blobs = {'data': im_blob,
                 'rois': rois_blob,
                 'labels': labels_blob}

    if cfg.TRAIN.BBOX_REG:
        blobs['bbox_targets'] = bbox_targets_blob
        blobs['bbox_loss_weights'] = bbox_loss_blob

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
                             replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
                             replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_loss_weights = \
            _get_bbox_regression_labels(roidb['bbox_targets'][keep_inds, :],
                                        num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_loss_weights

def _sample_rois_rpn(roidb, fg_rois_per_image, rois_per_image, num_classes, 
                 union_conv_height, union_conv_width):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    new_labels = np.zeros(labels.shape, dtype=np.int16)
    bbox_target = roidb['bbox_targets']
    new_bbox_target = np.zeros(bbox_target.shape, dtype=np.float16)

    conv_width = roidb['conv_width']
    conv_height = roidb['conv_height']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(labels > 0)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
                             replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(labels == 0)[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
                             replace=False)

    new_labels[fg_inds] = 1
    new_labels[bg_inds] = 0

    if 'rois' in roidb:
        rois = roidb['rois'][fg_inds]
    else:
        rois = None
        
    """
    print 'labels.shape %s' % labels.shape
    print 'bbox_target.shape %s' % (bbox_target.shape, )
    
    for i in fg_inds:
        print 'label : %s ' % labels[i]
        print 'bbox_target : %s ' % bbox_target[i]
    """
        
    new_bbox_target[fg_inds] = bbox_target[fg_inds]

    new_bbox_target, bbox_loss_weights = \
            _get_bbox_regression_labels_rpn(new_bbox_target,
                                        num_classes)


    """    
    print 'label no 1 : %s' % len(np.where(new_labels == 1)[0])
    print 'new_bbox_target no 1 : %s' % len(np.where(new_bbox_target != 0)[0])
    print 'bbox_loss_weights no 1 : %s' % len(np.where(bbox_loss_weights > 0)[0])
    """

    """
    # DJDJ
    labels = np.ones((1, 9, conv_h, conv_w), dtype=np.float32)
    bbox_targets = np.ones((1, 36, conv_h, conv_w), dtype=np.float32)
    bbox_loss = np.ones((1, 36, conv_h, conv_w), dtype=np.float32)
    """

    new_labels = new_labels.reshape((1, 9, conv_height, conv_width))

    new_bbox_target = new_bbox_target.reshape((1, 9, conv_height, conv_width, 4))
    new_bbox_target = new_bbox_target.transpose(0, 1, 4, 2, 3)
    new_bbox_target = new_bbox_target.reshape((1, 36, conv_height, conv_width))
    
    bbox_loss_weights = bbox_loss_weights.reshape((1, 9, conv_height, conv_width, 4))
    bbox_loss_weights = bbox_loss_weights.transpose(0, 1, 4, 2, 3)
    bbox_loss_weights = bbox_loss_weights.reshape((1, 36, conv_height, conv_width))
    
    output_labels = np.zeros((1, 9, union_conv_height, union_conv_width))
    output_bbox_targets = np.zeros((1, 36, union_conv_height, union_conv_width))
    output_bbox_loss_weights = np.zeros((1, 36, union_conv_height, union_conv_width))
    
    output_labels[:, :, 0:conv_height, 0:conv_width] = new_labels   
    output_bbox_targets[:, :, 0:conv_height, 0:conv_width] = new_bbox_target   
    output_bbox_loss_weights[:, :, 0:conv_height, 0:conv_width] = bbox_loss_weights   
    
    """
    # Generate positive rois based on index for debugging
    anchors = [[128*2, 128*1], [128*1, 128*1], [128*1, 128*2], 
               [256*2, 256*1], [256*1, 256*1], [256*1, 256*2], 
               [512*2, 512*1], [512*1, 512*1], [512*1, 512*2]]

    conv_scale_width = roidb['conv_scale_width']
    conv_scale_height = roidb['conv_scale_height']

    rois = np.zeros((len(fg_inds), 4), dtype=np.int16)
    for i, fg_ind in enumerate(fg_inds):
        center_x = fg_ind % conv_width
        center_y = (fg_ind - center_x) / conv_width % conv_height
        anchor = fg_ind / conv_height / conv_width
        
        anchor_w = anchors[anchor][0]
        anchor_h = anchors[anchor][1]
        
        x1 = center_x * conv_scale_width - anchor_w / 2
        y1 = center_y * conv_scale_height - anchor_h / 2
        x2 = x1 + anchor_w
        y2 = y1 + anchor_h
        
        rois[i, :] = x1, y1, x2, y2
    """
    
    

    """
    pos_labels = np.where(new_labels == 1)
    
    i = 0
    for d0, d1, d2, d3 in zip(pos_labels[0], pos_labels[1], pos_labels[2], pos_labels[3]):
        print '[%s] label : %s, bbox_target : %s, bbox_loss_weights : %s' % (i, new_labels[d0, d1, d2, d3], 
                                                     new_bbox_target[d0, d1*4 : d1*4+4, d2, d3],
                                                     bbox_loss_weights[d0, d1*4 : d1*4+4, d2, d3])
        i += 1
    """
    
    """
    print 'label no 2 : %s' % len(np.where(output_labels == 1)[0])
    print 'new_bbox_target no 2 : %s' % len(np.where(output_bbox_targets != 0)[0])
    print 'bbox_loss_weights no 2 : %s' % len(np.where(output_bbox_loss_weights > 0)[0])
    """
        
    return output_labels, None, rois, output_bbox_targets, output_bbox_loss_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_loss_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]

    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]

    return bbox_targets, bbox_loss_weights

def _get_bbox_regression_labels_rpn(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_loss_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    
    bbox_targets = np.zeros((clss.size, 4), dtype=np.float32)
    bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]

    #print ''
    #print 'len(inds) : %s' % len(inds)

    for ind in inds:
        bbox_targets[ind, :] = bbox_target_data[ind, 1:]
        bbox_loss_weights[ind, :] = [1., 1., 1., 1.]
        
        #print 'bbox_targets[ind, :] : %s - %s ' % (bbox_target_data[ind, 0], bbox_targets[ind, :])

    return bbox_targets, bbox_loss_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()

def _vis_minibatch_rpn(im_blob, rois_blob, labels_blob, roidb, bbox_targets_blob, bbox_loss_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        resized_gt_boxes = roidb[int(im_ind)]['resized_gt_boxes']
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        #cls = labels_blob[i]
        plt.imshow(im)
        
        for resized_gt_box in resized_gt_boxes:
            resized_gt_box = resized_gt_box.astype(np.int)
            plt.gca().add_patch(
                plt.Rectangle((resized_gt_box[0], resized_gt_box[1]), resized_gt_box[2] - resized_gt_box[0],
                              resized_gt_box[3] - resized_gt_box[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
        
        print 'roid : %s' % roi
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
