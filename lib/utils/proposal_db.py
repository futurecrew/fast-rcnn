import os
import leveldb
import cPickle

def remove_folder( folder):
    if os.path.exists(folder) == False:
        return
    
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def gogo(input_proposal, output_db):
    if os.path.isfile(input_proposal) == False:
        print 'File not found %s' % input_proposal
        return
    
    with open(input_proposal, 'r') as f:
        file_list = cPickle.load(f)
        box_list = cPickle.load(f)
        
    print 'finished reading the pickle file.'

    remove_folder(output_db)
    
    db = leveldb.LevelDB(output_db)        
    batch = leveldb.WriteBatch()

    i = 0
    for file, box in zip(file_list, box_list):
        batch.Put(file, cPickle.dumps(box))
        i += 1
        if i % 5000 == 0:
            print 'inserted %s data into db' % i
            db.Write(batch, sync = True)
            del batch
            batch = leveldb.WriteBatch()

    if i % 5000 > 0:
        db.Write(batch, sync = True)

    print 'inserted total %s data into db' % i
    print 'finished writing DB.'
            
def read_data(input_db):
    db = leveldb.LevelDB(input_db)
    
    for key, value in db.RangeIter():
        print 'key : %s' % (key)
        boxes = cPickle.loads(value)
        #print boxes        

if __name__ == '__main__':
    #input_proposal = 'E:/project/fast-rcnn/output/rpn_data/imagenet_val/vgg_cnn_m_1024_step_1_rpn_top_2300_candidate.pkl'
    input_proposal = '/home/dj/big/workspace/fast-rcnn/output/rpn_data/imagenet_train/vgg_cnn_m_1024_step_1_rpn_top_2300_candidate.pkl'
    #input_proposal = '/home/dj/big/workspace/fast-rcnn/output/rpn_data/imagenet_val/vgg_cnn_m_1024_step_1_rpn_top_2300_candidate.pkl'
    output_db = input_proposal.split('.pkl')[0] + '_db'
    
    gogo(input_proposal, output_db)
    read_data(output_db)    