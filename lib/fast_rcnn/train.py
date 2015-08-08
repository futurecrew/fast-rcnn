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

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None, model_to_use='frcnn', proposal='ss'):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        if model_to_use == 'frcnn':
           self.output_dir = self.output_dir + '_with_' + proposal
           
        self.model_to_use = model_to_use
        self.proposal = proposal

        print 'Computing bounding-box regression targets...'
        self.bbox_means, self.bbox_stds = \
                rdl_roidb.add_bbox_regression_targets(roidb, model_to_use)
        print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)
        
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
                    extended_stds = self.bbox_stds[:, np.newaxis, np.newaxis, np.newaxis]
                else:
                    extended_stds = self.bbox_stds[:, np.newaxis]
                net.params[bbox_pred][0].data[...] = \
                        (net.params[bbox_pred][0].data *
                         extended_stds)
                net.params[bbox_pred][1].data[...] = \
                        (net.params[bbox_pred][1].data *
                         self.bbox_stds + self.bbox_means)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_with_{:s}'.format(self.proposal) + 
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
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
            
            """
            axis1 = 3
            axis2 = 10
            axis3 = 20
            for i in range(len(self.solver.net.blobs['bbox_pred'].data)):
                pred = self.solver.net.blobs['bbox_pred'].data[i, axis1*4:axis1*4+4, axis2, axis3]
                print 'pred[%s] : %s' % (i, pred)
            """
            
            
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
    if model_to_use == 'rpn':
        rdl_roidb.prepare_roidb_rpn(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000,
              model_to_use='frcnn', proposal='ss'):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model,
                       model_to_use=model_to_use,
                       proposal=proposal)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'
