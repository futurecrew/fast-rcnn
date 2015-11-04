import os
import struct
import shutil
import numpy
import leveldb
import caffe
import matplotlib.pyplot as plt
import cPickle
from caffe.proto import caffe_pb2

def read_leveldb(figure_no, db_file_name, channel_no, class_names=None, dim_format='chw'):
    db = leveldb.LevelDB(db_file_name)

    display_no = 4
    
    plt.figure(figure_no, figsize=(display_no, display_no), dpi=100)
    
    if channel_no == 1:
        plt.gray();
    plt.suptitle(db_file_name)
    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    """
    count = [0, 0, 0, 0, 0]
    
    try: 
        for i in range(100000):
            datum = caffe_pb2.Datum.FromString(db.Get("%08d" % i))
            label = datum.label
            
            count[label] += 1
    except:
        pass

    total = sum(count)    
    for i in range(5):
        print '%s : %s, %.4f' % (i, count[i], (float(count[i]) / float(total)))
    """
    
    count = 0
    for k,v in db.RangeIter():
        count += 1
        
        if count > display_no * display_no:
            break
        
        datum = caffe_pb2.Datum.FromString(v)

        """
        data = datum.data.split('!@#$')
        image_data = data[0]
        featrues = data[1]
        """          
        image_data = datum.data  
        
        #image_data = numpy.fromstring(image_data, numpy.uint8) * 0.00390625
        image_data = numpy.fromstring(image_data, numpy.uint8)
        
        if image_data.shape[0] != datum.height * datum.width * channel_no:
            print 'Size error : image_data.shape : %s' % image_data.shape[0]
            print 'datum.height : %s' % datum.height
            print 'datum.width : %s' % datum.width
            print 'channel_no : %s' % channel_no
            continue
        
        print "channels : %d, height : %d, width : %d" % (datum.channels, datum.height, datum.width)
        
        #dim_format = 'hwc'
        
        if channel_no == 3:
            if dim_format == 'chw':
                bb = numpy.reshape(image_data, (channel_no, datum.height, datum.width))
                bb = numpy.transpose(bb, (1, 2, 0))
            elif dim_format == 'hwc':
                bb = numpy.reshape(image_data, (datum.height, datum.width, channel_no))
        elif channel_no == 1:
            bb = numpy.reshape(image_data, (datum.height, datum.width))
        
        label = datum.label
        if class_names != None:
            print "label : %s, class : %s" % (label, class_names[label])
        else:
            print "label : %s" % (label)
            
        #print features
        
        plt.subplot(display_no, display_no, count)
        plt.title(datum.label)
        plt.axis('off')
        plt.imshow(bb)

        """
        count += 1

        plt.subplot(4, 4, count)
        plt.title(datum.label)
        plt.axis('off')
        plt.imshow(bb[1])
        
        count += 1

        plt.subplot(4, 4, count)
        plt.title(datum.label)
        plt.axis('off')
        plt.imshow(bb[2])
        """
    plt.show()

            
         
def read_leveldb_cifar10(figure_no, db_file_name, index_size):

    db = leveldb.LevelDB(db_file_name)

    """
    data_no = 0
    while True:
        datum = caffe_pb2.Datum.FromString(db.Get("%05d" % data_no))
        data_no += 1
        print data_no
    """

    plt.figure(figure_no)

    for i in range(10):
        count = i + 1
        
        if index_size == 5:
            index = "%05d" % i
        elif index_size == 8:
            index = "%08d" % i
        datum = caffe_pb2.Datum.FromString(db.Get(index))
        
        print "cifar10 label : %d" % (datum.label)
        #print "count : %d, label : %d, len(data) : %d" % (count, datum.label, len(datum.data))
        #print "channels : %d, height : %d, width : %d" % (datum.channels, datum.height, datum.width)
        
        #for j in range(10):
        #    print "data[%d] : %s" % (j, ord(datum.data[j]))
                
        aa = numpy.fromstring(datum.data, numpy.uint8)
        bb = numpy.reshape(aa, (datum.channels, datum.height, datum.width))
        bb = numpy.swapaxes(bb, 0, 2)
        bb = numpy.swapaxes(bb, 0, 1)
        
        plt.subplot(1, 10, count)
        plt.title(datum.label)
        plt.axis('off')
        plt.imshow(bb)

    plt.show()

def display_mean_image():
    mean_file = 'E:/data/VOC2007/db/train-db-min-224-channel-3/pickles/mean-min-224.pickle'
    #mean_file = 'E:/data/VOC2007/db/train-db-min-224-channel-3/pickles/var-min-224.pickle'
    size = 224
    channel_no = 3
    with open(mean_file, 'rb') as fp:
        mean_data = cPickle.load(fp)
        
    #mean_data = numpy.reshape(mean_data, (96, 96))
    #plt.imshow(mean_data, cmap = cm.Greys_r)

    mean_data = numpy.reshape(mean_data, (size, size, channel_no))
    plt.imshow(mean_data)
    
    plt.show()
            

if __name__ == '__main__':

    class_names = None
    
    #display_mean_image()

    """
    db_file_name = 'E:/data/cifar10/cifar10_train_leveldb'
    #db_file_name = 'E:/data/cifar100/cifar100_train_leveldb'
    index_size = 5
    read_leveldb_cifar10(2, db_file_name, index_size)
    """

    """
    db_file_name = 'E:/data/eye/db/partial-train-db-size-96-ratio-0.9-channel-3'
    channel_no = 3
    """
    
    """
    db_file_name = 'E:/data/plankton/db/inverse/partial-train-db-size-96-ratio-0.9-shuffle2'
    channel_no = 1
    class_names = "acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified".split(',')
    """
    
    """
    db_file_name = 'E:/data/mnist/mnist_train_leveldb'
    channel_no = 1
    """

    
    db_file_name = '/home/dj/big/data/ilsvrc14/db/train-db-min-256-channel-3-shuffle2/'
    #db_file_name = '/home/dj/big/data/ilsvrc14/db/valid-db-min-256-channel-3-shuffle2/'
    channel_no = 3
    

    """
    #db_file_name = 'E:/data/ilsvrc14/db/train-db-min-256-channel-3-shuffle2'
    #db_file_name = 'E:/data/ilsvrc14/db/train-db-size-256-channel-3-shuffle2'
    #db_file_name = 'E:/data/ilsvrc14/db/train-db-size-256'
    db_file_name = 'E:/data/ilsvrc14/db/valid-db-size-256'
    channel_no = 3
    """
    
    read_leveldb(1, db_file_name, channel_no, class_names)
