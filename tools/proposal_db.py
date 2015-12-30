import _init_paths
import os
import leveldb
import cPickle
import scipy.io as sio
from datasets.factory import get_imdb
import numpy as np

def remove_folder( folder):
    if os.path.exists(folder) == False:
        return
    
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def make_proposal_db(input_proposal, output_db, box_order = None):
    if os.path.isfile(input_proposal) == False:
        print 'File not found %s' % input_proposal
        return
    
    print 'reading input data file : %s' % input_proposal
    if '.pkl' in input_proposal:
        try:
            with open(input_proposal, 'rb') as f:
                file_list = cPickle.load(f)
                box_list = cPickle.load(f)
        except:
            with open(input_proposal, 'rb') as f:
                data = cPickle.load(f)
                file_list = data['images'][0]
                box_list = data['boxes']
        
        if len(file_list) == 1:
            file_list = file_list[0]
            
        if box_order != None:
            new_list = []
            for one_box_list in box_list:
                new_one_box_list = one_box_list[:, box_order]
                new_list.append(new_one_box_list)   
            box_list = new_list
            
        print 'finished reading the pickle file.'
    elif '.mat' in input_proposal:
        matlab_data = sio.loadmat(input_proposal)
        raw_file_data = matlab_data['images'].ravel()
        raw_box_data = matlab_data['boxes'].ravel()
        file_list = []
        for i in xrange(raw_file_data.shape[0]):
            file = raw_file_data[i]
            if isinstance(file, list) == True:
                file = file[0]
            file_list.append(file.encode('ascii', 'ignore'))
        box_list = []
        for i in xrange(raw_box_data.shape[0]):
            if len(raw_box_data[i]) > 0:
                box_list.append(raw_box_data[i][:, box_order] - 1)
            else:
                box_list.append(raw_box_data[i])            
        print 'finished reading the mat file.'
    else:
        print 'unsupported file format.'
        print '.pkl and .mat files are supported.'
        return

    remove_folder(output_db)
    
    db = leveldb.LevelDB(output_db)        
    batch = leveldb.WriteBatch()

    i = 0
    for file, box in zip(file_list, box_list):
        if isinstance(file, list) == True:
            file = file[0]
        if isinstance(file, list) == True:
            file = file[0]
        batch.Put(file, cPickle.dumps(box))
        i += 1
        if i % 5000 == 0:
            print 'inserted %s data into DB' % i
            db.Write(batch, sync = True)
            del batch
            batch = leveldb.WriteBatch()

    if i % 5000 > 0:
        db.Write(batch, sync = True)

    print 'inserted total %s proposal data into DB' % i
    print 'finished writing proposal DB : %s' % output_db

def make_classification_db(imdb, input_file_list, output_db):
    result_list = []
    total_scores = np.zeros((len(input_file_list), 200))
    
    for input_classification in input_file_list:
        if os.path.isfile(input_classification) == False:
            print 'File not found %s' % input_classification
            return
        
        print 'reading input data file : %s' % input_classification
        if '.pkl' in input_classification:
            with open(input_classification, 'rb') as f:
                result = cPickle.load(f)
                result_list.append(result)
                
            print 'finished reading the pickle file.'
        else:
            print 'unsupported file format.'
            print '.pkl and .mat files are supported.'
            return

    remove_folder(output_db)
    
    db = leveldb.LevelDB(output_db)        
    batch = leveldb.WriteBatch()

    i = 0
    num_images = len(imdb.image_index)
    for i in xrange(num_images):
        input_file = imdb.image_path_at(i)
        input_id = input_file.split('/')[-1]

        model_no = 0
        for result in result_list:
            total_scores[model_no, :] = result[i][0]
            model_no += 1
            
        avg_scores = np.average(total_scores, axis=0)
            
        batch.Put(input_id, cPickle.dumps(avg_scores))
        i += 1
        if i % 5000 == 0:
            print 'inserted %s data into DB' % i
            db.Write(batch, sync = True)
            del batch
            batch = leveldb.WriteBatch()

    if i % 5000 > 0:
        db.Write(batch, sync = True)

    print 'inserted total %s classification data into DB' % i
    print 'finished writing classification DB : %s' % output_db
            
def read_data(input_db):
    db = leveldb.LevelDB(input_db)
    
    #aa = db.Get('n04228054_952')
    #boxes = cPickle.loads(aa)
    for key, value in db.RangeIter():
        print 'key : %s' % (key)
        boxes = cPickle.loads(value)
        #print boxes        

if __name__ == '__main__':
    #input_file = 'E:/project/fast-rcnn/output/rpn_data/imagenet_val/vgg_cnn_m_1024_step_1_rpn_top_2300_candidate.pkl'
    #input_file = '/home/dj/big/workspace/fast-rcnn/output/rpn_data/imagenet_train/vgg_cnn_m_1024_step_1_rpn_top_2300_candidate.pkl'
    #input_file = '/home/dj/big/workspace/fast-rcnn/output/rpn_data/imagenet_val/vgg_cnn_m_1024_step_1_rpn_top_2300_candidate.pkl'
    #box_order = None

    #input_file = '/home/nvidia/www/workspace/fast-rcnn/output/ss/trainval/ss_voc_2007_trainval_output.mat'
    #box_order = (1, 0, 3, 2)
    
    input_file = '/home/dj/big/workspace/fast-rcnn/data/selective_search_data/voc_2007_test.mat'
    #input_file = '/home/nvidia/www/workspace/fast-rcnn/data/selective_search_data/imagenet_val.mat'
    box_order = (1, 0, 3, 2)
    output_db = input_file.split('.')[0] + '_db'

    """    
    imdb_name = 'imagenet_train'
    imdb = get_imdb(imdb_name)
    input_file_list = []
    input_file_list.append('/home/dj/big/workspace/fast-rcnn/output/classifier/results/imagenet_train_vgg16_200_iter_10000.pkl')
    input_file_list.append('/home/dj/big/workspace/fast-rcnn/output/classifier/results/imagenet_train_googlenet_200_iter_60000.pkl')
    
    output_db = input_file_list[0].split('.')[0] + '_' + str(len(input_file_list)) + '_models_db'
    """
    
    #make_proposal_db(input_file, output_db, box_order)
    read_data('/home/dj/big/workspace/fast-rcnn/output/rpn_data/voc_2007_2012_trainval/vgg16_step_1_rpn_top_2300_candidate_db')
    
    #make_classification_db(imdb, input_file_list, output_db)

