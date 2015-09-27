
print_term = 50
#file_path = 'E:/project/fast-rcnn/experiments/logs/default_vgg_cnn_m_1024.txt.2015-04-29_10-42-14'
file_path = 'E:/project/fast-rcnn/experiments/logs/faster_rcnn/log_20150927_015100.txt'

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
                    if item == 'loss_cls' or item == 'loss_bbox':
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
        