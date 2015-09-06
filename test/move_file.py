import shutil
from os import listdir
from os.path import isfile

dir = 'E:/data/ilsvrc14/ILSVRC2014_DET_train/ILSVRC2014_train_all_data'
dest = dir

for f in listdir(dir):
    full_path = dir + '/' + f
    if isfile(full_path):
        continue
    
    #shutil.move(full_path + '/.', dest)
    
    for f2 in listdir(full_path):
        full_path2 = full_path + '/' + f2
        shutil.move(full_path2, dest)

    print '%s is moved' % f
            