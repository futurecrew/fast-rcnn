import os
import subprocess
import scipy.io as sio
import shutil

def read_child_info(meta_det_file_all, det_file):
    wnid_list = []
    with open(det_file, 'r') as f:
        i = 0
        for one_line in f.readlines():
            wnid = one_line.split(' ')[0]
            wnid_list.append(wnid)
    
    children_meta = sio.loadmat(meta_det_file_all)
    wnid_dict = {}
    for data in children_meta['synsets'][0]:
        det_id = data[0][0][0]
        wnid = data[1][0].encode('ascii', 'ignore')
        wnid_dict[det_id] = wnid
        
    det_parent_to_children_map = {}
    det_children_list = []
    det_child_to_parent_map = {}
    det_parent_list = []
    for data in children_meta['synsets'][0]:
        det_id = data[0][0]
        wnid = data[1][0].encode('ascii', 'ignore')
        class_name = data[2][0].encode('ascii', 'ignore')
        wnid_dict[wnid] = class_name
        
        if wnid in wnid_list:
            det_parent_list.append(wnid)
        else:
            continue
        
        #if wnid == 'n02346627':
         #   print 'hehe'
            
        
        child_list = []
        if len(data[4]) > 0:
            children = data[4][0]
    
            if len(children) > 0:    
                for child in children:
                    child_wnid = wnid_dict[child]
                    
                    #if child_wnid == 'n02346627':
                    #   print 'hehe2'
                        
                    child_list.append(child_wnid)
                    det_children_list.append(child_wnid)
                    det_child_to_parent_map[child_wnid] = wnid
                    
        det_parent_to_children_map[wnid] = child_list
    
    return det_parent_to_children_map, det_children_list, det_child_to_parent_map, wnid_list

def create_train_data(det_file, base_train_img_dir, target_train_img_dir, det_parent_to_children_map):
    with open(det_file, 'r') as f:
        i = 0
        for line in f.readlines():
            wnid = line.split(' ')[0]
            src_tar = base_train_img_dir + '/' + wnid + '.tar'
            dest_dir = target_train_img_dir + '/' + wnid
    
            if os.path.exists(dest_dir) == False:
                os.mkdir (dest_dir)
            
            if os.path.exists(src_tar):        
                cmd = 'tar xf ' + src_tar + ' -C ' + dest_dir
                subprocess.call(cmd, shell=True)
                print 'extracted parent %s' % src_tar
            else:
                print 'no parent %s' % src_tar
                
            children = det_parent_to_children_map[wnid] 
            for child in children:
                src_tar = base_train_img_dir + '/' + child + '.tar'
                cmd = 'tar xf ' + src_tar + ' -C ' + dest_dir
                subprocess.call(cmd, shell=True)
                print 'extracted child %s' % src_tar
                
            i += 1
            print '%s %s done' % (i, wnid)

def create_valid_data(det_file, cls_file, base_valid_img_dir, target_valid_img_dir, 
                      parent_to_child_map, det_children_list, 
                      det_child_to_parent_map, det_parent_list, 
                      cls_valid_gt):
    last_index = cls_valid_gt.rfind('.')
    output_valid_gt = cls_valid_gt[:last_index] + '_200.txt'

    cls_id_to_wnid_map = {}
    with open(cls_file, 'r') as f:
        i = 0
        for one_line in f.readlines():
            wnid = one_line.split(' ')[0]
            id = one_line.split(' ')[1]
            cls_id_to_wnid_map[id] = wnid

    det_wnid_to_id_map = {}
    with open(det_file, 'r') as f:
        i = 0
        for one_line in f.readlines():
            wnid = one_line.split(' ')[0]
            id = one_line.split(' ')[1]
            det_wnid_to_id_map[wnid] = id

    gt_id_in_200_list = []
    with open(cls_valid_gt, 'r') as f:
        i = 0
        img_no = 1
        for one_gt_id in f.readlines():
            i += 1
            one_gt_id = one_gt_id.rstrip()
            gt_wnid = cls_id_to_wnid_map[one_gt_id]
            
            if gt_wnid in det_children_list:
                gt_parent_wnid = det_child_to_parent_map[gt_wnid]
                print '%s changed to %s' % (gt_wnid, gt_parent_wnid)
            elif gt_wnid in det_parent_list:
                gt_parent_wnid = gt_wnid
            else:
                continue
                
            gt_id_in_200 = det_wnid_to_id_map[gt_parent_wnid]
            gt_id_in_200_list.append(gt_id_in_200)
            src_img_file = '%s/ILSVRC2012_val_%08d.JPEG' % (base_valid_img_dir, i)  
            dest_img_file = '%s/ILSVRC2012_val_%08d.JPEG' % (target_valid_img_dir, img_no)  
            shutil.copyfile(src_img_file, dest_img_file)
            
            if img_no == 74:
                print 'haha' 

            print '%s %s done' % (img_no, gt_wnid)
            
            img_no += 1
            
    with open(output_valid_gt, 'wt') as f:
        for id in gt_id_in_200_list:
            f.write(id + '\n')
            
def meta():        
    meta_det_file = '/home/dj/big/data/ilsvrc14/ILSVRC2015_devkit/data/meta_det.mat'
    
    det_meta = sio.loadmat(meta_det_file)
    
    for data in det_meta['synsets'][0]:
        wnid = data[1][0].encode('ascii', 'ignore')
        class_name = data[2][0].encode('ascii', 'ignore')
        try:
            if len(data) > 3 and len(data[3]) > 0:
                desc = data[3][0].encode('ascii', 'ignore')
            else:
                desc = ''
            if len(data) > 4:
                children = data[4][0].encode('ascii', 'ignore')
            else:
                children = None
        except:
            print data
        
        if children != None and children != class_name:
            print children 
    
    print det_meta
    
        
if __name__ == '__main__':
    
    cls_file = '/home/dj/big/data/ilsvrc14/ILSVRC2015_devkit/data/map_clsloc.txt'
    det_file = '/home/dj/big/data/ilsvrc14/ILSVRC2015_devkit/data/map_det.txt'
    meta_det_file_all = '/home/dj/big/data/ilsvrc14/ILSVRC2014_devkit/data/meta_det.mat'
    cls_valid_gt = '/home/dj/big/data/ilsvrc14/ILSVRC2015_devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt'
    base_train_img_dir = '/home/dj/big/data/ilsvrc14/ILSVRC2012_img_train'
    target_train_img_dir = '/home/dj/big/data/ilsvrc14/ILSVRC2012_img_train_200'
    base_valid_img_dir = '/home/dj/big/data/ilsvrc14/ILSVRC2012_img_val'
    target_valid_img_dir = '/home/dj/big/data/ilsvrc14/ILSVRC2012_img_val_200'
    
    parent_to_child_map, det_children_list, det_child_to_parent_map, det_parent_list = \
        read_child_info(meta_det_file_all, det_file)
    #create_train_data(det_file, base_train_img_dir, target_train_img_dir, det_parent_to_children_map)
    create_valid_data(det_file, cls_file, base_valid_img_dir, target_valid_img_dir, parent_to_child_map, det_children_list, det_child_to_parent_map, det_parent_list, cls_valid_gt)
    
