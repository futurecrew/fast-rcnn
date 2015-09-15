# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, bbox_means, bbox_stds, roidb, output_dir,
                 pretrained_model=None, model_to_use='frcnn', proposal='ss'):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        if model_to_use == 'frcnn':
           self.output_dir = self.output_dir + '_with_' + proposal
           
        self.model_to_use = model_to_use
        self.proposal = proposal

        self.bbox_means = bbox_means
        self.bbox_stds = bbox_stds
                
        """
        if cfg.TRAIN.LAZY_PREPARING_ROIDB == False:        
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb, model_to_use)
            print 'done'
        """
        
        self.solver = caffe.SGDSolver(solver_prototxt)
        
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb, bbox_means, bbox_stds)
        
    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net
        
        if cfg.TRAIN.BBOX_REG:
            if 'bbox_pred_rpn' in net.params:
                bbox_pred = 'bbox_pred_rpn'
            else:
                bbox_pred = 'bbox_pred'
                
            # save original values
            orig_0 = net.params[bbox_pred][0].data.copy()
            orig_1 = net.params[bbox_pred][1].data.copy()

            if cfg.TRAIN.NORMALIZE_BBOX:
                # scale and shift with bbox reg unnormalization; then save snapshot
                if self.model_to_use == 'rpn':
                    means = np.zeros((1, 36))
                    stds = np.zeros((1, 36))
                    
                    for i in range(9):
                        means[:, i*4:i*4+4] = self.bbox_means
                        stds[:, i*4:i*4+4] = self.bbox_stds
                    
                    means = means.ravel()
                    stds = stds.ravel()
                    
                    extended_stds = stds[:, np.newaxis, np.newaxis, np.newaxis]
                else:
                    means = self.bbox_means.ravel()
                    stds = self.bbox_stds.ravel()

                    extended_stds = stds[:, np.newaxis]
                    
                net.params[bbox_pred][0].data[...] = \
                        (net.params[bbox_pred][0].data *
                         extended_stds)
                net.params[bbox_pred][1].data[...] = \
                        (net.params[bbox_pred][1].data *
                         stds + means)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = self.solver_param.snapshot_prefix + infix
        if self.model_to_use == 'frcnn':
            filename += '_with_{:s}'.format(self.proposal)
        filename += '_iter_{:d}'.format(self.solver.iter) + '.caffemodel'
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG:
            # restore net to original state
            net.params[bbox_pred][0].data[...] = orig_0
            net.params[bbox_pred][1].data[...] = orig_1

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

def get_training_roidb(imdb, model_to_use):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb, model_to_use)
    print 'done'

    return imdb.roidb

def train_net(solver_prototxt, bbox_means, bbox_stds, roidb, output_dir,
              pretrained_model=None, max_iters=40000,
              model_to_use='frcnn', proposal='ss'):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, bbox_means, bbox_stds, roidb, output_dir,
                       pretrained_model=pretrained_model,
                       model_to_use=model_to_use,
                       proposal=proposal)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'
