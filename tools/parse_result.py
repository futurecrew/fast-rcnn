
print_term = 500
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151019_062800.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151021_013200.txt'

#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151015_180600.txt'

#VGG16 step 1
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151013_001400.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151021_013200.txt'

# VGG16 step 2
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151023_063200.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151025_133500.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151027_080500.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151027_122700.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151027_181500.txt'

# VGG16 step 2 again
file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151103_055800.txt'

# VGG16 step 2 previous best in hospital
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151017_200700.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151019_062800.txt'

# VGG16 step 3
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151028_101900.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151029_184100.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151101_081100.txt'

# VGG16 step 4
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151031_160400.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151101_131900.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151101_170500.txt'


# Googlenet poly
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151025_083000.txt'
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151027_181900.txt'

# Googlenet
#file_path = '/home/dj/big/workspace/fast-rcnn/experiments/logs/faster_rcnn/log_20151028_215800.txt'


with open(file_path) as f:
    lines = f.readlines()
    
    for token in ['loss_cls = ', 'loss_bbox =']:
        print ''
        print token
        score_sum = 0
        sum_count = 0
        
        for line in lines:
            try:
                if (token in line) == False:
                    continue
                
                parsed = line.split(' ')
                for i, item in enumerate(parsed):
                    if item == 'loss_cls' or item == 'loss_bbox' or item == 'loss':
                        score = float(parsed[i+2])
                        score_sum += score
                        sum_count += 1
                        
                        if sum_count == print_term:
                            print '%.3f' % (score_sum / sum_count)
                            score_sum = 0 
                            sum_count = 0
                
            except:
                pass
        
        if sum_count > 0:
            print '%.3f' % (score_sum / sum_count)
        