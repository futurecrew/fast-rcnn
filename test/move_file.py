import shutil
from os import listdir
from os.path import isfile

src_dir = '/home/nvidia/www/data/ilsvrc14/ILSVRC2014_DET_train/ILSVRC2014_DET_train_all_data/'
dest = '/home/nvidia/www/data/ilsvrc14/ILSVRC2014_DET_train/ILSVRC2014_DET_train_all_data/'

for f in listdir(src_dir):
    full_path = src_dir + '/' + f
    if isfile(full_path):
        continue
    elif full_path == dest:
        continue
    
    #shutil.move(full_path + '/.', dest)
    
    for f2 in listdir(full_path):
        full_path2 = full_path + '/' + f2
        shutil.move(full_path2, dest)

    print '%s is moved' % f
            