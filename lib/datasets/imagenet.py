# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.pascal_voc
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import sys
from fast_rcnn.config import cfg

from utils.model import last_conv_size


class imagenet(datasets.pascal_voc):
    def __init__(self, image_set, model_to_use='frcnn', proposal='ss', proposal_file='', devkit_path=None):
        datasets.imdb.__init__(self, 'imagenet_' + image_set)
        self._image_set = image_set
        self._default_path = '/home/dj/data/ilsvrc14'
        #self._default_path = 'E:/data/ilsvrc14'

        self._devkit_path = self._default_path + '/ILSVRC2015_devkit'
        self._data_path = self._default_path
        
        print 'image_set : %s' % image_set
        
        if image_set == 'train':
            self._label_path = self._data_path + '/ILSVRC2014_DET_bbox_train/ILSVRC2014_DET_bbox_train_all_data'
            self._image_path = self._data_path + '/ILSVRC2014_DET_train/ILSVRC2014_DET_train_all_data'
            #self._label_path = self._data_path + '/ILSVRC2014_DET_bbox_train/ILSVRC2014_DET_bbox_train_10000_data'
            #self._image_path = self._data_path + '/ILSVRC2014_DET_train/ILSVRC2014_DET_train_10000_data'
            #self._data_id_file = self._devkit_path + '/data/det_lists/train.txt'
        elif image_set == 'val':
            self._label_path = self._data_path + '/ILSVRC2013_DET_bbox_val'
            self._image_path = self._data_path + '/ILSVRC2013_DET_val'
            self._data_id_file = self._devkit_path + '/data/det_lists/val.txt'
        elif image_set == 'val_10':
            self._label_path = self._data_path + '/ILSVRC2013_DET_bbox_val'
            self._image_path = self._data_path + '/ILSVRC2013_DET_val'
            self._data_id_file = self._devkit_path + '/data/det_lists/val_10.txt'
        elif image_set == 'val_2000':
            self._label_path = self._data_path + '/ILSVRC2013_DET_bbox_val'
            self._image_path = self._data_path + '/ILSVRC2013_DET_val'
            self._data_id_file = self._devkit_path + '/data/det_lists/val_2000.txt'
        
        class_name_list_file = 'data/imagenet_det.txt'
        self._classes_names, self._classes= self._load_class_info(self._data_path + '/ILSVRC2015_devkit/data/meta_det.mat',
                                               class_name_list_file)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()
        self._gt_roidb = None
        self.gt_roidb()
        self.proposal_file = proposal_file
        
        # Default to roidb handler
        if model_to_use == 'rpn':           # Step 1, 3
            self._roidb_handler = self.rpn_train_roidb
        elif model_to_use == 'frcnn':
            if proposal == 'rpn':           # Step 2, 4
                self._roidb_handler = self.rpn_proposal_roidb
            elif proposal == 'ss':
                self._roidb_handler = self.selective_search_roidb

        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        print 'self._image_path : %s' % self._image_path
        
        assert os.path.exists(self._devkit_path), \
                'Imagenet devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def _load_class_info(self, meta_det_file, class_name_list_file):
        det_meta = sio.loadmat(meta_det_file)
        class_names = []
        wnids = []
        
        class_names.append('__background__')    # always index 0
        wnids.append('__background__')
        
        with open(class_name_list_file) as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip())
                
        for data in det_meta['synsets'][0]:
            wnid = data[1][0].encode('ascii', 'ignore')
            class_name = data[2][0].encode('ascii', 'ignore')
            if class_name in class_names:
                wnids.append(wnid)

        return class_names, wnids
        
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        image_path = os.path.join(self._image_path,
                                  self._image_index[i] + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
    
    def _load_image_set_index(self):
        if self._image_set == 'train':
            image_index = []
            for label_file in os.listdir(self._label_path):
                image_index.append(label_file[:-4])
        else:
            image_set_file = os.path.join(self._data_path, 'ILSVRC2015_devkit/data/det_lists',
                                          self._image_set + '.txt')
            assert os.path.exists(image_set_file), \
                    'Path does not exist: {}'.format(image_set_file)
            with open(image_set_file) as f:
                image_index = [x.split(' ')[0] for x in f.readlines()]
        
        print '_load_image_set_index() done'
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        if self._gt_roidb != None:
            return self._gt_roidb
        
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
                self._image_index = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            print 'len(self._image_index) : %s' % len(self._image_index)
            
            self._gt_roidb = roidb
            return roidb

        gt_roidb = []
        zero_label_list = []
        for i, label_file in enumerate(self._image_index):
            gt = self._load_imagenet_annotation(label_file + '.xml')
            if gt == None:
                zero_label_list.append(i)
                continue
            
            gt_roidb.append(gt)
            if i % 1000 == 0:
                print '%s labels read' % i
                
        # remove zero label data from train data
        for i in zero_label_list[::-1]: 
            del self._image_index[i]
                                           
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(self._image_index, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        print 'len(self._image_index) : %s' % len(self._image_index)

        self._gt_roidb = gt_roidb
        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        return self.gt_roidb()

        """        
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb
        """

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'imagenet_' + self._year))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def rpn_train_roidb(self):
        """
        Return the database of rpn regions of interest.
        Ground-truth ROIs are also included.
        """
        return self.gt_roidb()

    def rpn_proposal_roidb(self):
        """
        Return the database of rpn regions of interest.
        Ground-truth ROIs are also included.
        """
        
        gt_roidb = self.gt_roidb()
        return gt_roidb
    
        #rpn_roidb = self._load_rpn_roidb(gt_roidb)
        #roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)        
        #return roidb
    
    def _load_rpn_roidb(self, gt_roidb):
        filename = os.path.abspath(self.proposal_file)
        assert os.path.exists(filename), \
               'RPN data not found at: {}'.format(filename)
        with open(filename, 'rb') as fid:
            file_list = cPickle.load(fid)
            box_list = cPickle.load(fid)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_imagenet_annotation(self, label_file):
        """
        Load image and bounding boxes info from XML file in the imagenet
        format.
        """
        filename = os.path.join(self._label_path, label_file)
        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        if len(objs) == 0:
            return None
        
        # Load object bounding boxes into a data frame.
        ix = 0
        for obj in objs:
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            cls = self._class_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            
            if x2 <= x1 or y2 <= y1:
                boxes = np.delete(boxes, len(boxes)-1, 0)
                print '%s is deleted here. (%d, %d, %d, %d)' % (label_file, x1, y1, x2, y2)
                continue
            
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            ix += 1

        if ix == 0:
            return None
        
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'gt_boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'label_file' : label_file,
                'flipped' : False}


    def _write_imagenet_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # ilsvrc14/ILSVRC2015_devkit/results/resultscomp4-4520_det_val.txt
        path = self._devkit_path + '/results/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        # read test data name, id mapping file
        data_name_id_dict = {}
        with open(self._data_id_file, 'r') as f:
            for line in f.readlines():
                name, id = line.split()
                data_name_id_dict[name] = id                
        
        print 'Writing imagenet results file'
        filename = path + comp_id + '_' + 'det_' + self._image_set + '.txt'
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    
                    index_id = data_name_id_dict[index]
                    
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index_id, cls_ind, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))
                        
        # Make none to save memory
        all_boxes[:] = []
        print 'boxes are cleared in the memory.'
        
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'Imagenetdevkit-matlab-wrapper')

        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'imagenet_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path + '/evaluation', comp_id,
                       self._image_set, output_dir, int(rm_results))

        """
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} '.format(datasets.OCTAVE)
        cmd += '--eval '
        cmd += 'imagenet_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d})' \
               .format(self._devkit_path + '/evaluation', comp_id,
                       self._image_set, output_dir, int(rm_results))
        """
        
        print('Running:\n{}'.format(cmd))
        
        status = subprocess.call(cmd, shell=True)
        """        
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        while True:
            buff = process.stdout.readline()
            if buff == '' and process.poll() != None: 
                break
            sys.stdout.write(buff)
        process.wait()
        """


    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_imagenet_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.imagenet('train')
    res = d.roidb
    from IPython import embed; embed()
