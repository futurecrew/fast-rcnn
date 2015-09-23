import glob
import os
import thread
import time
import ctypes
import pylab
from shutil import copyfile
#import psutil

ES_CONTINUOUS = 0x80000000
ES_AWAYMODE_REQUIRED = 0x00000040
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
  
def start_prevent_sleep():
    try:
        while True:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            time.sleep(40)
    except:
        pass

def prevent_sleep():
    print "prevent_sleep() started"
    thread.start_new(start_prevent_sleep, ())

def start_check_exit_key(callback):
    while True:
        a = raw_input()
        if a == 'exit':
            print 'exit key is pressed'
            callback()
            break
    
def check_exit_key(callback):
    thread.start_new(start_check_exit_key, (callback, ))
         
def copy_files(src_folder, dest_folder, file_no_per_folder):
    directory_names = list(set(glob.glob(os.path.join(src_folder, "*"))).difference(set(glob.glob(os.path.join(src_folder,"*.*")))))
    copied_file_no = 0
    
    for folder in directory_names:
        print "folder : " + folder
        folder_name = folder.split('\\')[-1]
        for fileNameDir in os.walk(folder):
            new_folder = os.path.join(dest_folder, folder_name)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
                
            for index, fileName in enumerate(fileNameDir[2]):
                if index >= file_no_per_folder:
                    break
                copyfile(os.path.join(src_folder, folder_name, fileName), os.path.join(dest_folder, folder_name, fileName))
                copied_file_no += 1
            break

    print "Copied %s files from %s to %s." % (copied_file_no, src_folder, dest_folder)
"""
def wait_previous_python_process():
    # If there is python.exe with more than 1 Gbytes
    # then wait until the process finishes.

    message_printed = False
        
    while True:
        plist = psutil.pids()
        
        has_previous_process = False
        for pid in plist:
            p = psutil.Process(pid)
            
            try:
                if p.name() == 'python.exe':
                    mem_info = p.memory_info()
                    if mem_info[0] >= 1024 * 1024 * 500:    # 500 Mbytes
                        has_previous_process = True
                        break
            except:
                pass
            

        if has_previous_process:
            if message_printed == False:                                    
                print 'There is a running python.exe.'
                print 'Wait for the process ends.'
                message_printed = True
            time.sleep(60)
        else:
            print 'There is no running python.exe.'
            print 'Now stop waiting.'
            break
"""
def wait_previous_python_process2():
    message_printed = False
        
    while True:
        has_previous_process = os.path.exists('processing.txt')

        if has_previous_process:
            if message_printed == False:                                    
                print 'There is a running python.exe.'
                print 'Wait for the process ends.'
                message_printed = True
            time.sleep(60)
        else:
            try:
                # Make a file to mark a process is running             
                f = open('processing.txt', 'w')
                f.close()
            except:
                print 'Cannot create processing.txt'
                print 'Continue waiting'
                continue
            
            break

def notify_end_python_process():
    # Remove this file to mark the process finished running
    try:
        os.remove('processing.txt')
    except:
        pass
    

def print_stat(object, predicts, y_valid, exclude_unknown):    
    
    label_list = "acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified".split(',')
    labels_name_to_index = {}
    labels_index_to_name = {}
    for i, label in enumerate(label_list):
        labels_name_to_index[label] = i
        labels_index_to_name[i] = label

    UNKNOWN_THRESHOLD = 0.9
    UNKNOWN_PROBABILITY = 0.3
    
    if exclude_unknown:
        output_dim = 120
        EXCLUDE_UNKNOWN = True
    else:
        output_dim = 121
        EXCLUDE_UNKNOWN = False

    changed_to_unknown_no = 0
    changed_to_unknown_success_no = 0
    changed_to_unknown_fail_no = 0
        
    printlog("UNKNOWN_THRESHOLD : %s" % UNKNOWN_THRESHOLD)
    printlog("UNKNOWN_PROBABILITY : %s" % UNKNOWN_PROBABILITY)

    column_no = output_dim
    if EXCLUDE_UNKNOWN:
        column_no += 1
        
    result_table = np.zeros((column_no, 3), dtype=np.int)
    
    for i in range(int(predicts.shape[0])):
        predicted_class = np.argmax(predicts[i])
        max_prob = np.max(predicts[i])
        label = y_valid[i]

        if hasattr(object, 'exclude_index'):
            if label == object.exclude_index and predicted_class != label:                    
                class_name = labels_index_to_name[label]
                print '----------------------------------------------------'
                print 'label\t\t: %s (%s)' % (label, class_name)
                print 'best guess\t: %s (%s) with %s' % (predicted_class, labels_index_to_name[predicted_class], predict[i][predicted_class])
                print 'predict label\t: %s (%s) with %s' % (label, class_name, predicts[i][label])

        
        changed_to_unknown = False
        if EXCLUDE_UNKNOWN and max_prob < UNKNOWN_THRESHOLD:
        #if max_prob < UNKNOWN_THRESHOLD:
                predicts[i], predicted_class, max_prob = \
                    object.get_predict_with_unknown(predicts[i], predicted_class, max_prob, UNKNOWN_PROBABILITY)
                changed_to_unknown_no += 1
                
                changed_to_unknown = True
        
        result_table[label][0] += 1
        
        if predicted_class != label:
            result_table[label][2] += 1
            if changed_to_unknown:
                changed_to_unknown_fail_no += 1
        else:
            result_table[label][1] += 1
            if changed_to_unknown:
                changed_to_unknown_success_no += 1
            
    score = object.get_score(predicts, y_valid)
        
    
    total_sum = 0
    total_right = 0
    total_wrong = 0
    
    print '=========================================================================================='
    print 'class\ttotal\tright\twrong\tacc\tname'
    for i in range(result_table.shape[0]):
        sum = result_table[i][0]
        right = result_table[i][1]
        wrong = result_table[i][2]
        
        total_right += right
        total_wrong += wrong
        total_sum += sum
        
        acc = float(right) / float(sum) if sum != 0 else -1
        
        if acc >= 0 and acc < 0.5:
            error = True
        else:
            error = False
        
        printlog('[%s]\t%s\t%s\t%s\t%.2f\t%s' % (i, sum, right, wrong, acc, labels_index_to_name[i]))
    print '=========================================================================================='
    total_percent = float(total_right) / float(total_sum)
    print '[total]\t%s\t%s\t%s\t%.2f' % (total_sum, total_right, total_wrong, total_percent)
    print '[score]\t%.5f' % score
    print '[changed_to_unknown_no]\t\t%s' % changed_to_unknown_no
    print '[changed_to_unknown_success_no]\t%s' % changed_to_unknown_success_no
    print '[changed_to_unknown_fail_no]\t%s' % changed_to_unknown_fail_no
    print '=========================================================================================='
    print ''   
         
def show_weights(layer):        
    while isinstance(layer, lasagne.layers.InputLayer) == False:
        if isinstance(layer, dnn.Conv2DDNNLayer):
            first_conv_layer =  layer
        layer = layer.input_layer
    
    weights = first_conv_layer.get_params()[0].get_value()
    weights_no = weights.shape[0]
    
    display_size = int(math.sqrt(weights_no)) + 1
    
    print 'display_size : %s' % display_size
    
    pylab.gray() 
    for i in range(display_size):
        for j in range(display_size):
            index = i * display_size + j + 1
            
            if index >= weights_no:
                break

            print 'index : %s' % index
    
            one_weight = weights[index][0]
            pylab.subplot(display_size, display_size, index) 
            pylab.axis('off') 
            pylab.imshow(one_weight)
    
    pylab.show()
    
if __name__ == '__main__':
    #copy_files('../data/train', '../data/train_medium', 40)
    copy_files('E:/caffe-windows/data/plankton/train', 
               'E:/caffe/caffe-windows/data/plankton/train_20', 20)