"""The ROI proposal layer.

RoIProposalLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
import numpy as np
from utils.box_prediction import get_predicted_boxes

class RoIProposalLayer(caffe.Layer):
    """ROI proposal layer used for testing."""

    #def __init__(self, haha):

    def setup(self, bottom, top):

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(300, 5)
                

    def forward(self, bottom, top):
        self.NMS_THRESH = 0.7
        self.MAX_CAND_BEFORE_NMS = 3000
        self.MAX_CAND_AFTER_NMS = 300        

        input_img_data = bottom[0].data
        input_img_no = len(input_img_data) 
        cls_pred = bottom[1].data
        box_deltas = bottom[2].data
        
        blob = np.zeros((0, 5))
        for i in range(input_img_no):
            rigid_rects, pred_boxes, sorted_scores = get_predicted_boxes(cls_pred, box_deltas,
                                           input_img_data.shape[2], input_img_data.shape[3],
                                           self.NMS_THRESH, self.MAX_CAND_BEFORE_NMS,
                                           self.MAX_CAND_AFTER_NMS)
            pred_no = len(pred_boxes)
            batch_idx = np.zeros((pred_no, 1))
            batch_idx.fill(i)
            
            one_blob = np.hstack((batch_idx, pred_boxes))
            blob = np.vstack((blob, one_blob))
        
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

