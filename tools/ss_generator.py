import os
import time
import subprocess
import scipy.io as sio
import numpy as np

def create_target_file_list(folder, output_list):
    with open(output_list, 'w') as f:
        for file_name in os.listdir(folder):
            index = file_name.find('.')
            f.write(file_name[:index] + '\n')
    print '%s is generated.' % output_list

def call_external_ss(img_base_folder, data_list, output_dir, extension, multi_no, ss_exe, ss_path):
    file_id = []
    file_name = []
    with open(data_list, 'r') as f:
        for one_line in f.readlines():
            file_id.append(one_line.rstrip())
            file_name.append(one_line.rstrip() + '.' + extension)
    
    a = {}
    a['data'] = np.array(file_name)
    
    img_list_mat_file = '%s/img_list.mat' % (output_dir)
    sio.savemat(img_list_mat_file, a)

    total_data_no = len(file_name)
    
    # DJDJ
    total_data_no = 10
    multi_no = 3
    
    chunk_size = total_data_no / multi_no
    if total_data_no % multi_no > 0:
        chunk_size += 1

    start_idx_list = []
    end_idx_list = []
        
    for i in range(multi_no):    
        start_idx =  chunk_size * i
        end_idx = min(chunk_size * (i + 1), total_data_no)
        
        start_idx_list.append(start_idx)
        end_idx_list.append(end_idx)
        
        
        cmd = 'cd {} && '.format(ss_path)
        cmd += '{:s} -nodisplay -nodesktop '.format(ss_exe)
        cmd += '-r "dbstop if error; '
        cmd += 'op_selective_search_boxes({:d}, {:d}, \'{:s}\', \'{:s}\', \'{:s}\'); quit;"' \
               .format(start_idx, end_idx, img_list_mat_file, img_base_folder, output_dir)
        print 'Running [%s] : %s' % ((i+1), cmd)
        
        status = subprocess.call(cmd, shell=True)
        
    
    while True:
        all_done = True
        for i, start_idx, end_idx in zip(range(multi_no), start_idx_list, end_idx_list):
            done_noti_file = '%s/matlab_ss_noti_%s_%s.txt' % (output_dir, start_idx, end_idx)
            
            if os.path.isfile(done_noti_file) == False:
                all_done = False
                break
            
        if all_done == True:
            for i, start_idx, end_idx in zip(range(multi_no), start_idx_list, end_idx_list):
                done_noti_file = '%s/matlab_ss_noti_%s_%s.txt' % (output_dir, start_idx, end_idx)
                os.remove(done_noti_file)
            break
        else:
            time.sleep(5)
                
    print 'Finished all the selective search processes.'
    
    
    #voc_org = sio.loadmat('E:/project/fast-rcnn/data/selective_search_data/voc_2007_train.mat')
    
    _images = [0] * total_data_no
    images = []
    images.append(_images)
    boxes = [0] * total_data_no
    for i, start_idx, end_idx in zip(range(multi_no), start_idx_list, end_idx_list):
        ss_output_file = '%s/matlab_ss_output_%s_%s.mat' % (output_dir, start_idx, end_idx)
        
        a = sio.loadmat(ss_output_file)
        #images[0, start_idx:end_idx] = file_id[start_idx:end_idx]
        
        for j in range(len(a['result'])):
            images[0][start_idx + j] = [file_id[start_idx + j]]
            boxes[start_idx + j] = a['result'][j, 0]
        
    a = {}
    a['images'] = images
    a['boxes'] = boxes
    
    final_ss_output_file = '%s/ss_output.mat' % (output_dir)
    sio.savemat(final_ss_output_file, a)
    
    voc_new = sio.loadmat(final_ss_output_file)
    
    print 'Finished the selective search.'
    
    
    
if __name__ == '__main__':
    label_folder = 'E:/data/ilsvrc14/ILSVRC2014_DET_bbox_train/ILSVRC2014_DET_bbox_train_all_data'
    img_folder = 'E:/data/ilsvrc14/ILSVRC2014_DET_train/ILSVRC2014_DET_train_all_data'
    ss_path = 'E:/project/SelectiveSearchCodeIJCV'
    ss_exe = 'matlab.exe'

    output_list = os.getcwd() + '/output/ss/data_list.txt'
    
    #create_target_file_list(label_folder, output_list)
    
    extension = 'JPEG'
    multi_no = 3

    output_dir = os.getcwd() + '/output/ss'

    call_external_ss(img_folder, output_list, output_dir, extension, multi_no, ss_exe, ss_path)
    