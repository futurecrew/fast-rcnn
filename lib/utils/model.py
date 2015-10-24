import math

def conv_size(pad, kernel, stride, input_size):
    return math.floor(float(input_size + 2 * pad - kernel) / float(stride)) + 1 

def pool_size(pad, kernel, stride, input_size):
    return math.ceil(float(input_size + 2 * pad - kernel) / float(stride)) + 1 
    
def last_conv_size(input_size, model):
    if model == 'VGG_CNN_M_1024':
        conv1 = conv_size(0, 7, 2, input_size)
        pool1 = pool_size(0, 3, 2, conv1)
        conv2 = conv_size(1, 5, 2, pool1)
        pool2 = pool_size(0, 3, 2, conv2)
        
        scale = input_size / pool2
        return int(pool2), scale
    elif model == 'VGG16':
        pool1 = pool_size(0, 2, 2, input_size)
        pool2 = pool_size(0, 2, 2, pool1)
        pool3 = pool_size(0, 2, 2, pool2)
        pool4 = pool_size(0, 2, 2, pool3)
        
        scale = input_size / pool4
        return int(pool4), scale
    elif model == 'VGG19':
        pool1 = pool_size(0, 2, 2, input_size)
        pool2 = pool_size(0, 2, 2, pool1)
        pool3 = pool_size(0, 2, 2, pool2)
        pool4 = pool_size(0, 2, 2, pool3)
        
        scale = input_size / pool4
        return int(pool4), scale
    elif model == 'GOOGLENET':
        conv1 = conv_size(3, 7, 2, input_size)      # conv1/7x7_s2
        pool2 = pool_size(0, 3, 2, conv1)           # pool1/3x3_s2
        pool3 = pool_size(0, 3, 2, pool2)           # pool2/3x3_s2
        pool4 = pool_size(0, 3, 2, pool3)           # pool3/3x3_s2
        
        scale = input_size / pool4
        return int(pool4), scale
    elif model == 'GOOGLENET_BN':
        conv1 = conv_size(3, 7, 2, input_size)      # conv1/7x7_s2
        pool2 = pool_size(0, 3, 2, conv1)           # pool1/3x3_s2
        pool3 = pool_size(0, 3, 2, pool2)           # pool2/3x3_s2
        pool4 = pool_size(0, 3, 2, pool3)           # pool3/3x3_s2
        
        scale = input_size / pool4
        return int(pool4), scale
    elif model == 'GOOGLENET2':
        conv1 = conv_size(3, 7, 2, input_size)      # conv1/7x7_s2
        pool2 = pool_size(0, 3, 2, conv1)           # pool1/3x3_s2
        pool3 = pool_size(0, 3, 2, pool2)           # pool2/3x3_s2
        pool4 = pool_size(0, 3, 2, pool3)           # pool3/3x3_s2
        
        scale = input_size / pool4
        return int(pool4), scale
    elif model == 'GOOGLENET3':
        conv1 = conv_size(3, 7, 2, input_size)      # conv1/7x7_s2
        pool2 = pool_size(0, 3, 2, conv1)           # pool1/3x3_s2
        pool3 = pool_size(0, 3, 2, pool2)           # pool2/3x3_s2
        pool4 = pool_size(0, 3, 2, pool3)           # pool3/3x3_s2
        
        scale = input_size / pool4
        return int(pool4), scale
    elif model == 'GOOGLENET4':
        conv1 = conv_size(3, 7, 2, input_size)      # conv1/7x7_s2
        pool2 = pool_size(0, 3, 2, conv1)           # pool1/3x3_s2
        pool3 = pool_size(0, 3, 2, pool2)           # pool2/3x3_s2
        pool4 = pool_size(0, 3, 2, pool3)           # pool3/3x3_s2
        
        scale = input_size / pool4
        return int(pool4), scale
    elif model == 'GOOGLENET5':
        conv1 = conv_size(3, 7, 2, input_size)      # conv1/7x7_s2
        pool2 = pool_size(0, 3, 2, conv1)           # pool1/3x3_s2
        pool3 = pool_size(0, 3, 2, pool2)           # pool2/3x3_s2
        pool4 = pool_size(0, 3, 2, pool3)           # pool3/3x3_s2
        
        scale = input_size / pool4
        return int(pool4), scale
        
if __name__ == '__main__':
    print last_conv_size(600, 'VGG_CNN_M_1024')
    print last_conv_size(600, 'VGG16')
    print last_conv_size(224, 'VGG16')
    print last_conv_size(224, 'GOOGLENET')
    print last_conv_size(224, 'GOOGLENET_BN')
    print last_conv_size(224, 'GOOGLENET2')
    print last_conv_size(224, 'GOOGLENET3')
    print last_conv_size(224, 'GOOGLENET4')
    print last_conv_size(224, 'GOOGLENET5')
    