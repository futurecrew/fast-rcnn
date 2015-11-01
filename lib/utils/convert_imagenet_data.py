import os
import sys
import glob
import struct
import random
import time
import numpy
import cPickle
from PIL import Image, ImageOps
from skimage.util import img_as_ubyte
from skimage import segmentation
from skimage.morphology import watershed
from skimage.morphology import white_tophat, black_tophat
from skimage.morphology import disk
from skimage import measure
from skimage import morphology

import matplotlib.pyplot as plt
import leveldb
import caffe
from caffe.proto import caffe_pb2
from subprocess import call

class Converter:

    def __init__(self, label_id_mapping_file):
        self.initialize_labels(label_id_mapping_file)

        self.train_no = 0
        self.valid_no = 0
        self.train_batch = leveldb.WriteBatch()
        self.valid_batch = leveldb.WriteBatch()
        self.test_batch = leveldb.WriteBatch()

    def initialize_labels(self, label_id_mapping_file):        
        self.labels_name_to_index = {}
        self.labels_index_to_name = {}

        with open(label_id_mapping_file) as f:
            #index = 1
            while True:
                one_line = f.readline().replace('\n', '')
                if one_line == '':
                    break
                label = one_line.split(' ')[0]
                index = int(one_line.split(' ')[1])
                self.labels_name_to_index[label] = index
                self.labels_index_to_name[index] = label
                #index += 1
    
    def bytes2int(self, str):
        return int(str.encode('hex'), 16)
     
    def remove_folder(self, folder):
        if os.path.exists(folder) == False:
            return
        
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    
    def create_image_rotations(self, img, angle):
        # converted to have an alpha layer
        im2 = img.convert('RGBA')
        # rotated image
        #rot = im2.rotate(angle, expand=1)
        rot = im2.rotate(angle)
        # a white image same size as rotated image
        fff = Image.new('RGBA', img.size, (255,)*4)
        # create a composite image using the alpha layer of rot as a mask
        out = Image.composite(rot, fff, rot)
        # (converting back to mode='1' or whatever..)
        return out.convert(img.mode)

    def generate_data_list(self, train_data_folder, valid_data_folder, 
                                  train_list_file, valid_list_file, 
                                  valid_label_file, shuffle_in_folder):
        
        print 'generate_data_list()'
        
        train_data_list = []
        valid_data_list = []
        
        modes = ['train', 'valid']
        data_folders = [train_data_folder, valid_data_folder]
        output_lists = [train_data_list, valid_data_list]

        #modes = ['valid']
        #data_folders = [valid_data_folder]
        #output_lists = [valid_data_list]
        
        with open(valid_label_file, 'r') as f:
            valid_labels = f.read()

        valid_label_list = valid_labels.split('\n')

        try:
            
            for mode, data_folder, output_list in zip(modes, data_folders, output_lists):
                if mode == 'train':
                    directory_names = list(set(glob.glob(os.path.join(data_folder, "*"))).difference(set(glob.glob(os.path.join(data_folder,"*.*")))))
                else:
                    directory_names = []
                    directory_names.append(data_folder)
                    
                train_no = 0
                valid_no = 0
                i = 0
                
                for folder in directory_names:
                    for index, file_name in enumerate(sorted(os.listdir(folder))):
                        if file_name[-5:] != ".JPEG":
                            continue
                        
                        #file_name = "{0}{1}{2}".format(folder, os.sep, file_name)
                      
                        if mode == 'train':  
                            wordnetId = folder.split(os.sep)[-1]
                            file_name = wordnetId + os.sep + file_name
                            label = self.labels_name_to_index[wordnetId]
                            train_no += 1
                        elif mode == 'valid':
                            label = valid_label_list[valid_no]
                            valid_no += 1
                        
                        output_list.append('%s\t%s\n' % (file_name, label))
                            
                        i += 1
                        
                        if i % 10000 == 0:
                            print '%s files are read.' % i
        
            if shuffle_in_folder:
                random.shuffle(train_data_list)
                random.shuffle(valid_data_list)
                    
            output_file = open(train_list_file, 'w')
            for data in train_data_list:
                output_file.write(data)
            output_file.close()
                    
            output_file = open(valid_list_file, 'w')
            for data in valid_data_list:
                output_file.write(data)
            output_file.close()
                    
    
            print 'Train list file is created : %s' % (train_list_file)
            print 'Valid list file is created : %s' % (valid_list_file)

        except:
            print 'here'
    
    
    def insert_db(self, train_or_valid, image, label, features, channel_no, inverse):
        if inverse:
            image_ubyte = 255 - img_as_ubyte(image)
        else:
            image_ubyte = img_as_ubyte(image)

        image_ubyte = numpy.transpose(image_ubyte, (2, 0, 1))
                
        image_string = image_ubyte.tostring()
        
        if features != None:
            delimeter = '!@#$'
            self.datum.data = image_string + delimeter + features
        elif channel_no > 3:
            selem = disk(6)
            w_tophat = white_tophat(image_ubyte, selem)
            b_tophat = black_tophat(image_ubyte, selem)
            self.datum.data = image_string + w_tophat.tostring() + b_tophat.tostring()
        else:
            self.datum.data = image_string
            
        self.datum.label = int(label)                
    
        serialized = self.datum.SerializeToString()
        
        if train_or_valid == 'train':
            self.train_batch.Put("%08d" % self.train_no, serialized)                    
            self.train_no += 1
        else:
            self.valid_batch.Put("%08d" % self.valid_no, serialized)                    
            self.valid_no += 1
    
    def getLargestRegion(self, props, labelmap, imagethres):
            
        regionmaxprop = None
        for regionprop in props:
            # check to see if the region is at least 50% nonzero
            if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
                continue
            if regionmaxprop is None:
                regionmaxprop = regionprop
            if regionmaxprop.filled_area < regionprop.filled_area:
                regionmaxprop = regionprop
        return regionmaxprop

    def get_features(self, image):
        features = []
        
        image = image.copy()
        # Create the thresholded image to eliminate some of the background
        imagethr = numpy.where(image > numpy.mean(image),0.,1.0)
         
        #Dilate the image
        imdilated = morphology.dilation(imagethr, numpy.ones((4,4)))
         
        # Create the label list
        label_list = measure.label(imdilated)
        label_list = imagethr*label_list
        label_list = label_list.astype(int)
        region_list = measure.regionprops(label_list)
        maxregion = self.getLargestRegion(region_list, label_list, imagethr)
        # guard against cases where the segmentation fails by providing zeros
        ratio = 0.0
        minor_axis_length = 0.0
        major_axis_length = 0.0
        area = 0.0
        convex_area = 0.0
        eccentricity = 0.0
        equivalent_diameter = 0.0
        euler_number = 0.0
        extent = 0.0
        filled_area = 0.0
        orientation = 0.0
        perimeter = 0.0
        solidity = 0.0
        centroid = [0.0,0.0]
        if ((not maxregion is None) and (maxregion.major_axis_length != 0.0)):
            ratio = 0.0 if maxregion is None else maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
            minor_axis_length = 0.0 if maxregion is None else maxregion.minor_axis_length
            major_axis_length = 0.0 if maxregion is None else maxregion.major_axis_length
            area = 0.0 if maxregion is None else maxregion.area
            convex_area = 0.0 if maxregion is None else maxregion.convex_area
            eccentricity = 0.0 if maxregion is None else maxregion.eccentricity
            equivalent_diameter = 0.0 if maxregion is None else maxregion.equivalent_diameter
            euler_number = 0.0 if maxregion is None else maxregion.euler_number
            extent = 0.0 if maxregion is None else maxregion.extent
            filled_area = 0.0 if maxregion is None else maxregion.filled_area
            orientation = 0.0 if maxregion is None else maxregion.orientation
            perimeter = 0.0 if maxregion is None else maxregion.perimeter
            solidity = 0.0 if maxregion is None else maxregion.solidity
            centroid = [0.0,0.0] if maxregion is None else maxregion.centroid
        """
        else:
            print maxregion
            plt.figure(1, figsize=(1, 1), dpi=100)
            plt.gray();                
            plt.subplot(1, 1, 1)
            plt.imshow(image)
            plt.show()
        """
            

        features.append(ratio)
        features.append(minor_axis_length)
        features.append(major_axis_length)
        features.append(area)
        features.append(convex_area)
        features.append(eccentricity)
        features.append(equivalent_diameter)
        features.append(euler_number)
        features.append(extent)
        features.append(filled_area)
        features.append(orientation)
        features.append(perimeter)
        features.append(solidity)
        features.append(centroid[0])
        features.append(centroid[1])
        
        for i in range(len(features)):
            features[i] *= 256               
        feature_array = numpy.array(features, dtype=numpy.uint8)
         
        return feature_array.tostring() 

             
    def convert_data_to_db(self, train_data_folder, valid_data_folder, min_pixel, 
                       train_db_name, valid_db_name, 
                       train_list_file, valid_list_file,
                       channel_no, preserve_ar):
        
        self.remove_folder(train_db_name)
        self.remove_folder(valid_db_name)
                
        self.train_db = leveldb.LevelDB(train_db_name)
        self.valid_db = leveldb.LevelDB(valid_db_name)
    
        self.datum = caffe.proto.caffe_pb2.Datum()
        self.datum.channels = channel_no
        self.datum.width = min_pixel
        self.datum.height = min_pixel
    
        print "convert_train_data"
        print "train_db_name : %s" % train_db_name
        print "valid_db_name : %s" % valid_db_name
        print "channel_no : %s" % channel_no
    
        modes = ['train', 'valid']
        
        start_time = time.time()
        
        for mode in modes:
            if mode == 'train':
                image_list_file = open(train_list_file, 'rb')
                data_folder = train_data_folder
            else: 
                image_list_file = open(valid_list_file, 'rb')
                data_folder = valid_data_folder
                
            lines = image_list_file.readlines()
            image_list_file.close()
            
            total_data_no = len(lines)
        
            print ''
            print 'processing %s' % mode
            
            for i, line in enumerate(lines):
                parsed = line.split('\t')
                label = parsed[1]
                file_path = parsed[0]
                file_path = file_path.replace('\r', '')
                file_path = file_path.replace('\n', '')
                
                
                org_image = Image.open(data_folder + '/' + file_path)                
                org_size = org_image.size
                
                if preserve_ar == 'preserve':
                    if org_size[0] > org_size[1]:
                        hpercent = (min_pixel/float(org_image.size[1]))
                        wsize = int((float(org_image.size[0])*float(hpercent)))
                        image_width = wsize
                        image_height = min_pixel
                    else:
                        wpercent = (min_pixel/float(org_image.size[0]))
                        hsize = int((float(org_image.size[1])*float(wpercent)))
                        image_width = min_pixel
                        image_height = hsize
                elif preserve_ar == 'ignore':
                    image_width = min_pixel
                    image_height = min_pixel

                #if file_path.endswith('n04008634_20954.JPEG'):
                #    print 'here'
                #if file_path.endswith('n02105855_2933.JPEG'):
                #    print 'here'

                if org_image.mode != 'RGB':                    
                    #print org_image.mode 
                    org_image = org_image.convert('RGB')
                
                image = org_image.resize((image_width, image_height), Image.ANTIALIAS)

                self.datum.width = image_width
                self.datum.height = image_height
                    
                self.insert_db(mode, image, label, None, channel_no, False)
                
                if mode == 'train' and self.train_no > 0 and self.train_no % 1000 == 0:
                    self.train_db.Write(self.train_batch, sync = True)
                    del self.train_batch
                    self.train_batch = leveldb.WriteBatch()
                    print "%.1f %% done." % (i * 100.0 / total_data_no)
                    print 'Processed %i total train images. %d sec' % (self.train_no, (time.time() - start_time))
                    start_time = time.time()
    
                if mode == 'valid' and self.valid_no > 0 and self.valid_no % 1000 == 0:
                    self.valid_db.Write(self.valid_batch, sync = True)
                    del self.valid_batch
                    self.valid_batch = leveldb.WriteBatch()
                    print 'Processed %i valid images.' % self.valid_no
                    
        # Write last batch of images
        if self.train_no % 1000 != 0:
            self.train_db.Write(self.train_batch, sync = True)
        if self.valid_no % 1000 != 0:
            self.valid_db.Write(self.valid_batch, sync = True)
    
        print 'Processed %d train, %d valid' % (self.train_no, self.valid_no)
    
    def make_mean_var_file(self, train_db_name, mean_output_file, var_output_file, width, height):
        db = leveldb.LevelDB(train_db_name)

        image_no = 0
        try:
            for i in xrange(sys.maxint):
                datum = caffe_pb2.Datum.FromString(db.Get("%08d" % (i)))
                image_no += 1
        except:
            pass
        
        arr = numpy.zeros((width, height), numpy.float32)
        var = numpy.zeros((width, height), numpy.float32)
        
        for i in xrange(image_no):
            datum = caffe_pb2.Datum.FromString(db.Get("%08d" % (i)))
            data = numpy.fromstring(datum.data, numpy.ubyte)
            imarr = numpy.asarray(numpy.reshape(data, (height, width)), dtype=numpy.float32)
            arr = arr + imarr / image_no
        
        for i in xrange(image_no):
            datum = caffe_pb2.Datum.FromString(db.Get("%08d" % (i)))
            data = numpy.fromstring(datum.data, numpy.ubyte)
            imarr = numpy.asarray(numpy.reshape(data, (height, width)), dtype=numpy.float32)
            var = var + numpy.square(imarr - arr) / image_no
            
        var = numpy.sqrt(var)

        # Round values in array and cast as 8-bit integer
        #arr = numpy.array(numpy.round(arr), dtype=numpy.uint8)
        
        with open(mean_output_file,'wb') as fp:
            cPickle.dump(arr, fp)
        
        with open(var_output_file,'wb') as fp:
            cPickle.dump(var, fp)

        print "pickled %s" % mean_output_file
        print "pickled %s" % var_output_file
        
    def save_test_file_names(self, data_set_folder, test_file_name_db_name):
        self.remove_folder(test_file_name_db_name)    
            
        test_file_name_db = leveldb.LevelDB(test_file_name_db_name)
        
        files = []
        
        for fileNameDir in os.walk(data_set_folder):   
            for index, fileName in enumerate(fileNameDir[2]):
                if fileName[-5:] != ".JPEG":
                  continue
                
                files.append(fileName)
            
        test_file_name_db.Put("test_file_names", files.__str__())
    
    def convert_test_data(self, data_set_folder, min_pixel, test_db_name, test_output_pickle_path, 
                          inverse, channel_no = 1):
        self.remove_folder(test_db_name)
            
        test_db = leveldb.LevelDB(test_db_name)
        
        pickleTestX = test_output_pickle_path + "/testX_size_" + str(min_pixel) + ".pickle"
        pickleFileNames = test_output_pickle_path + "/fileNames.pickle"
        
        if not os.path.exists(test_output_pickle_path):
            os.makedirs(test_output_pickle_path)

        numberofImages = 0    
    
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = channel_no
        datum.width = min_pixel
        datum.height = min_pixel
        
        test_batch = leveldb.WriteBatch()
    
        print "Load test dataset from image files"
    
        for fileNameDir in os.walk(data_set_folder):   
            for index, fileName in enumerate(fileNameDir[2]):
                if fileName[-5:] != ".JPEG":
                  continue
                numberofImages += 1
        
        imageSize = min_pixel * min_pixel
        num_rows = numberofImages # one row for each image in the test dataset

        batch_size = 10000    
        data_size = min(batch_size, numberofImages)
        testX = numpy.zeros((data_size, channel_no, imageSize), dtype=numpy.uint8)
        
        files = []
        db_index = 0
        pickle_index = 0
        batch_no = 1
        
        print "Reading images"
        for fileNameDir in os.walk(data_set_folder):   
            for index, fileName in enumerate(fileNameDir[2]):
                if fileName[-5:] != ".JPEG":
                  continue
                
                nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
                org_image = Image.open(nameFileImage)
                files.append(fileName)
                
                image = org_image.resize((min_pixel, min_pixel), Image.ANTIALIAS)
    
                """
                print fileName
                
                plt.figure(1, figsize=(1, 1), dpi=100)
                plt.gray();                
                plt.subplot(1, 1, 1)
                plt.imshow(image)
                plt.show()
                """
    
                if inverse:
                    image_ubyte = 255 - img_as_ubyte(image)
                else:
                    image_ubyte = img_as_ubyte(image)
                
                if channel_no > 1:
                    selem = disk(6)
                    w_tophat = white_tophat(image_ubyte, selem)
                    b_tophat = black_tophat(image_ubyte, selem)
                    datum.data = image_ubyte.tostring() + w_tophat.tostring() + b_tophat.tostring()
                    image_output = numpy.concatenate((image_ubyte, w_tophat, b_tophat), axis=1)
                else:
                    datum.data = image_ubyte.tostring()
                    image_output = image_ubyte
                
                    
                test_batch.Put("%08d" % db_index, datum.SerializeToString())
    
                testX[pickle_index] = numpy.reshape(image_output, (channel_no, imageSize))
    
                db_index += 1
                pickle_index += 1
                
                if db_index % 1000 == 0:
                    test_db.Write(test_batch, sync = True)
                    del test_batch
                    test_batch = leveldb.WriteBatch()
                    print 'Processed %i test images.' % db_index
    
                if pickle_index % batch_size == 0:
                    pickle_file_name = pickleTestX + "_" + str(batch_no)
                    with open(pickle_file_name,'wb') as fp:
                        cPickle.dump(testX, fp)
                        print "pickled %s" % pickle_file_name
                        data_size = min(batch_size, numberofImages - batch_size * batch_no)
                        testX = numpy.zeros((data_size, channel_no, imageSize), dtype=numpy.uint8)
                        batch_no += 1
                        pickle_index = 0
                
                report = [int((j+1)*num_rows/20.) for j in range(20)]
                if db_index in report: print numpy.ceil(db_index *100.0 / num_rows), "% done"
    
    
        # Write last batch of images
        if db_index % 1000 != 0:
            test_db.Write(test_batch, sync = True)
    
        if pickle_index % batch_size > 0:
            pickle_file_name = pickleTestX + "_" + str(batch_no)
            with open(pickle_file_name,'wb') as fp:
                cPickle.dump(testX, fp)
                print "pickled %s" % pickle_file_name
                        
        with open(pickleFileNames,'wb') as fp:
            cPickle.dump(files, fp)
    
        print 'Processed a total of %i images.' % db_index
    

if __name__ == '__main__':
    base_folder = '/home/dj/big/data/ilsvrc14/'
    train_data_folder = base_folder + '/ILSVRC2012_img_train_200'
    valid_data_folder = base_folder + '/ILSVRC2012_img_val_200'
    #train_data_folder = base_folder + '/ILSVRC2012_img_train_small'
    #valid_data_folder = base_folder + '/ILSVRC2012_img_val_small'
    test_data_folder = base_folder + '/ILSVRC2012_img_test'
    valid_label_file = base_folder + '/ILSVRC2015_devkit/data/ILSVRC2015_clsloc_validation_ground_truth_200.txt'
    label_id_mapping_file = base_folder + '/ILSVRC2015_devkit/data/map_det.txt'
    output_folder = base_folder
    min_pixel = 256
    
    #inverse = True
    inverse = False
    
    shuffle_in_folder = True
    #shuffle_in_folder = False
    
    preserve_ar = 'preserve'
    #preserve_ar = 'ignore'
    
    #exclude_unknown = True
    exclude_unknown = False

    #extract_features = True
    extract_features = False
    
    #channel_no = 1
    channel_no = 3
    
    if inverse:
        inverse_string = '/inverse'
    else:
        inverse_string = ''
    
    if preserve_ar == 'preserve':
        ar_string = 'min'
    elif preserve_ar == 'ignore':
        ar_string = 'size'
        
    train_db_name = '%s/db%s/train-db-%s-%s' % (output_folder, inverse_string, ar_string, min_pixel)
    valid_db_name = '%s/db%s/valid-db-%s-%s' % (output_folder, inverse_string, ar_string, min_pixel)
    test_db_name = '%s/db%s/test-db-%s-%s' % (output_folder, inverse_string, ar_string, min_pixel)

    class_db_name = '%s/db%s/class-names' % (output_folder, inverse_string)
    test_file_name_db_name = '%s/db%s/test-file-names' % (output_folder, inverse_string)
    test_output_pickle_path = output_folder + "/pickles/" + inverse_string + "/" + test_data_folder.split("/")[-1]
    train_list_file = '%s/db/train_list.txt' % (output_folder)
    valid_list_file = '%s/db/valid_list.txt' % (output_folder)
    
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    preserve_ar_str = ''
    
    if extract_features:
        train_db_name += '-features'
        valid_db_name += '-features'
    
    if channel_no > 1:
        train_db_name += '-channel-' + str(channel_no)
        valid_db_name += '-channel-' + str(channel_no)
        test_db_name += '-channel-' + str(channel_no)
        test_output_pickle_path += '-channel-' + str(channel_no)
        
    if exclude_unknown:
        train_db_name += '-exclude-unknown'
        valid_db_name += '-exclude-unknown'
        
    if shuffle_in_folder:
        train_db_name += '-shuffle2'
        valid_db_name += '-shuffle2'
    else:
        train_db_name += '-shuffle'
        valid_db_name += '-shuffle'
        

    converter = Converter(label_id_mapping_file)

    """
    converter.generate_data_list(train_data_folder, valid_data_folder, 
                                  train_list_file, valid_list_file, 
                                  valid_label_file, shuffle_in_folder)
    """

    converter.convert_data_to_db(train_data_folder, valid_data_folder, min_pixel, 
                       train_db_name, valid_db_name, 
                       train_list_file, valid_list_file,
                       channel_no, preserve_ar)

    

    """    
    print "Computing image mean and variance..."
    if not os.path.exists('%s/pickles' % train_db_name):
        os.makedirs('%s/pickles' % train_db_name)
    mean_output_file = '%s/pickles/mean-size-%i%s.pickle' % (output_folder, min_pixel, preserve_ar_str)
    var_output_file = '%s/pickles/var-size-%i%s.pickle' % (output_folder, min_pixel, preserve_ar_str)
    converter.make_mean_var_file(train_db_name, mean_output_file, var_output_file, min_pixel, min_pixel)
    """

    """
    converter.convert_test_data(test_data_folder, min_pixel, test_db_name, test_output_pickle_path, 
                       inverse, channel_no)
    """
    #converter.save_test_file_names(test_data_folder, test_file_name_db_name)

