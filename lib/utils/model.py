import math

def conv_size(pad, kernel, stride, input_size):
    return math.floor((input_size + 2 * pad - kernel) / stride) + 1 

def pool_size(pad, kernel, stride, input_size):
    return math.ceil((input_size + 2 * pad - kernel) / stride) + 1 
    
def last_conv_size(input_size, model='VGG_CNN_M_1024'):
    conv1 = conv_size(0, 7, 2, input_size)
    pool1 = pool_size(0, 3, 2, conv1)
    conv2 = conv_size(1, 5, 2, pool1)
    pool2 = pool_size(0, 3, 2, conv2)
    
    return pool2
    
    