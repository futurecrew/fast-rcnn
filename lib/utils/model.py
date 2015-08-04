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
        
if __name__ == '__main__':
    print last_conv_size(600, 'VGG_CNN_M_1024')
    print last_conv_size(901, 'VGG_CNN_M_1024')
    print last_conv_size(1000, 'VGG16')
    