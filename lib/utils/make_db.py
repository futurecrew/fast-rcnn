import cPickle
import leveldb


def make_db(input_proposal, output_db, box_order = None):
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
    
    
#if __name__ == '__main__':
#    make_db()