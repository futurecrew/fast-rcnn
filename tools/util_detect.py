import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io
from PIL import Image, ImageDraw, ImageFont
import psutil

def draw_rect(draw, left, top, right, bottom, color, text = None, score = None):    
    draw.line((left, top, left, bottom), fill=color, width=2)
    draw.line((left, top, right, top), fill=color, width=2)
    draw.line((right, top, right, bottom), fill=color, width=2)
    draw.line((left, bottom, right, bottom), fill=color, width=2)
    if text != None:
        draw.text((left+5, top+5), ' %s %.1f' % (text, score), fill=color)

def display_prediction(image_path, ground_rects, pred_result_list):
    base = Image.open(image_path).convert('RGBA')
    
    draw = ImageDraw.Draw(base)
    
    for ground_rect in ground_rects:
        draw_rect(draw, ground_rect.left, ground_rect.top, ground_rect.right, ground_rect.bottom, 'blue', None)
    
    i = 0
    for pred_rect, pred_class, class_name, score in pred_result_list:
        draw_rect(draw, pred_rect[0], pred_rect[1], pred_rect[2], pred_rect[3], 'red', class_name, score)
    
    base.show()

    for proc in psutil.process_iter():
        if proc.name == "display":
            proc.kill()

def area(left, top, right, bottom):
    return (right - left) * (bottom - top)

def iou(rect1_left, rect1_top, rect1_right, rect1_bottom, 
        rect2_left, rect2_top, rect2_right, rect2_bottom):
    area1 = area(rect1_left, rect1_top, rect1_right, rect1_bottom)
    area2 = area(rect2_left, rect2_top, rect2_right, rect2_bottom)
    
    inter_left = max(rect1_left, rect2_left);
    inter_right = min(rect1_right, rect2_right);
    inter_top = max(rect1_top, rect2_top);
    inter_bottom = min(rect1_bottom, rect2_bottom);
    
    if inter_left >= inter_right or inter_top >= inter_bottom:
        return 0
    else: 
        inter_area = area(inter_left, inter_top, inter_right, inter_bottom)
        iou = float(inter_area) / float(area1 + area2 - inter_area)
        
        """
        if iou >= 0.5:
            print '(%s, %s, %s, %s) and (%s, %s, %s, %s)' % (rect1_left, rect1_top, rect1_right, rect1_bottom, rect2_left, rect2_top, rect2_right, rect2_bottom)
            print iou
        """
        
        return iou
    
def check_match(ground_class_list, ground_rect_list, pred_class, pred_rect, match_threshold):
    i = 0
    for ground_class, ground_rect in zip(ground_class_list, ground_rect_list): 
        if ground_class == pred_class and iou(pred_rect[0], pred_rect[1], pred_rect[2], pred_rect[3], 
                        ground_rect.left, ground_rect.top, ground_rect.right, ground_rect.bottom) >= match_threshold:
                return i
        i += 1
            
    return -1

def ap(ground_class_list, ground_rect_list, pred_result_list, match_threshold):
    """
    Computes the average precision.
    This function computes the average prescision between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    Returns
    -------
    score : double
            The average precision over the input lists
    """
    score = 0.0
    num_hits = 0.0
    match_index = [False] * len(ground_class_list)
    ground_no = 0
    predict_no = 0
    correct_no = 0
    wrong_no = 0
    i = 0
    
    ground_no += len(ground_class_list)
    predict_no += len(pred_result_list)
    for pred_rect, pred_class, class_name, pred_score in pred_result_list:
        match = check_match(ground_class_list, ground_rect_list, pred_class, pred_rect, match_threshold)
        if match > -1 and match_index[match] == False:
        #if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
            correct_no += 1
            match_index[match] = True
        else:
            wrong_no += 1
        i += 1
        
    if len(ground_class_list) == 0:
        ap = 1.0
    else:
        ap = score / len(ground_class_list)
    return ap, ground_no, predict_no, correct_no, wrong_no

def map(ground_class_list, ground_rect_list, pred_result_list, match_threshold):
    """
    Computes the mean average precision.
    This function computes the mean average prescision between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    Returns
    -------
    score : double
            The mean average precision over the input lists
    """
    total_ground_no = 0
    total_predict_no = 0
    total_correct_no = 0
    total_wrong_no = 0
    ap_list = []
    for gc, gr, pr in zip(ground_class_list, ground_rect_list, pred_result_list):
        ap_val, ground_no, predict_no, correct_no, wrong_no = ap(gc, gr, pr, match_threshold)
        ap_list.append(ap_val)
        total_ground_no += ground_no
        total_predict_no += predict_no
        total_correct_no += correct_no
        total_wrong_no += wrong_no
    return np.mean(ap_list), total_ground_no, total_predict_no, total_correct_no, total_wrong_no

def test():
    from matplotlib.patches import Rectangle
    someX, someY = 2, 3
    currentAxis = plt.gca()
    currentAxis.add_patch(Rectangle((someX - .5, someY - .5), 1, 1, facecolor="grey"))

if __name__ == '__main__':
    ground_rects = []
    predict_rect_list = []
    class_name_list = []
    
    class test_class:
        def __init__(self, left, top, right, bottom):
            self.left = left
            self.top = top
            self.right = right
            self.bottom = bottom
        
    test1 = test_class(20, 20, 100, 100)
    test2 = test_class(50, 50, 300, 300)
    ground_rects = [test1, test2]
    predict_rect_list = [[[0, 3, 100, 200, 1],
     [10, 10, 200, 120, 2],
     [100, 30, 230, 300, 3]],
     
     [[40, 30, 430, 370, 4],
     [310, 10, 500, 320, 5]]
    ]
    class_name_list = ['grass', 'car']
    
    display_prediction('E:\data\VOC2007\JPEGImages/000010.jpg', ground_rects, predict_rect_list, class_name_list)
    #test()
    
    