import os
import cPickle

def read_label_file(self, folder):
    
    cache_file = os.path.join('data/cache/voc2007_labels.pkl')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            labels = cPickle.load(fid)
        print 'VOC2007 labels loaded from {}'.format(cache_file)
        return labels
    
    labels = {}
    
    for index, file_name in enumerate(os.listdir(folder)):
        if file_name[-4:] != ".xml":
            continue

        file_path = "{0}{1}{2}".format(folder, os.sep, file_name)
        
        with open(file_path) as f:
            content = f.read()

        root = ET.fromstring(content)
        label_string = ''
        label_resized_string = ''
        
        image_width = root.find('size').find('width').text
        image_height = root.find('size').find('height').text

        if preserve_ar == 'preserve':
            if image_width > image_height:
                hpercent = (min_pixel/float(image_height))
                wsize = int((float(image_width)*float(hpercent)))
                new_image_width = wsize
            else:
                new_image_width = min_pixel
        elif preserve_ar == 'ignore':
            new_image_width = min_pixel 
            
        resize_ratio = float(new_image_width) / float(image_width)
        
        for object in root.findall('object'):
            label_name = object.find('name').text
            xmin = float(object.find('bndbox').find('xmin').text)
            ymin = float(object.find('bndbox').find('ymin').text)
            xmax = float(object.find('bndbox').find('xmax').text)
            ymax = float(object.find('bndbox').find('ymax').text)

            label_index = self.labels_name_to_index[label_name]
            if label_string != '':
                label_string += '-'
            label_string += '{0}:({1},{2},{3},{4})'.format(label_index, int(xmin), int(ymin), int(xmax), int(ymax))

            xmin_resized = xmin * resize_ratio
            ymin_resized = ymin * resize_ratio
            xmax_resized = xmax * resize_ratio
            ymax_resized = ymax * resize_ratio

            if label_resized_string != '':
                label_resized_string += '-'
            label_resized_string += '{0}:({1},{2},{3},{4})'.format(label_index, int(xmin_resized), int(ymin_resized), int(xmax_resized), int(ymax_resized))
             
        labels[file_name.split('.')[0]] = label_string + '|' + label_resized_string
        
        if index > 0 and index % 1000 == 0:
            print '%s label files done' % index
            
    return labels