
print_term = 500
i = 0
score_sum = 0
file_path = 'E:/project/fast-rcnn/experiments/logs/default_vgg16.txt.2015-04-29_10-43-20'

with open(file_path) as f:
    lines = f.readlines()
    for line in lines:
        try:
            if ('loss_cls = ' in line) == False:
                continue
            
            score = float(line.split(' ')[15])
            score_sum += score
            i += 1
            
            if i % print_term == 0:
                print '%d : %.3f' % (20 * i, score_sum / print_term)
                score_sum = 0 
            
        except:
            pass