import numpy as np
import cv2
import os
import shutil
import time
import threading
import argparse
import json
import os.path as osp
import base64

#from maskrcnn_benchmark.modeling.rrpn.anchor_generator import draw_anchors
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.config import cfg

from predictor import Predictor
#from PIL import Image

import sys
sys.path.append('./apex/')

#ld_path = os.getenv('LD_LIBRARY_PATH')
#my_lib_path = './lib/'
os.environ['LD_LIBRARY_PATH'] += './lib'

print(os.environ['LD_LIBRARY_PATH'])


#指定卡号
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def mat_inter(box1, box2):
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def solve_coincide(box1, box2):
    # box=(xA,yA,xB,yB)
    coincide = 0
    coincide1 = 0
    coincide2 = 0
    if mat_inter(box1, box2) == True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01 + 1) * (y02 - y01 + 1)
        area2 = (x12 - x11 + 1) * (y12 - y11 + 1)
        coincide = intersection / (area1 + area2 - intersection)
        coincide1 = intersection / area1
        coincide2 = intersection / area2
    return coincide, coincide1, coincide2


def moveDuplicates(cls_boxes, scores, masks, tag):
    ifPrint = tag == 'print'

    eraseFlag = [0 for i in range(len(cls_boxes))]
    cls_boxes_temp = cls_boxes

    #print('cls_boxes_temp type ', type(cls_boxes_temp))
    score_temp = scores
    masks_temp = masks
    boxes_count = len(cls_boxes)

    for i in range(len(cls_boxes) - 1):
        scoreI = scores[i]

        if eraseFlag[i] == 1:
            continue
        
        boxI = cls_boxes[i]
        if(boxI[0]==boxI[2] or boxI[1] == boxI[3]):
            eraseFlag[i] = 1
            continue

        for j in range(i + 1, len(cls_boxes)):
            scoreJ = scores[j]
            
            if eraseFlag[j] == 1:
                continue
            
            boxJ = cls_boxes[j]
            
            if(boxJ[0] == boxJ[2] or boxJ[1] == boxJ[3]):
                eraseFlag[j] == 1
                continue
            ratioIJ, ratioI, ratioJ = solve_coincide(boxI, boxJ)
            if ratioIJ > 0.4 :
                #if(ifPrint) :
                if(False):    
                    eraseFlag[i] = 1
                    eraseFlag[j] = 1
                    x1 = min(boxI[0], boxJ[0])
                    y1 = min(boxI[1], boxJ[1])
                    x2 = max(boxI[2], boxJ[2])
                    y2 = max(boxI[3], boxJ[3])
                    cls_boxes_temp.append([x1, y1, x2, y2])
                    eraseFlag.append(0)
                    score_temp.append((scores[i] + scores[j]) / 2)
                    
                    masks_temp.append(masks[i])
                    boxes_count = boxes_count - 2                
                else :

                    if(scoreI > scoreJ  and eraseFlag[j] == 0):
                        eraseFlag[j] = 1
                        boxes_count = boxes_count - 1
                    elif scoreJ > scoreI and eraseFlag[i] == 0:
                        eraseFlag[i] = 1
                        boxes_count = boxes_count - 1

            elif ratioI > 0.7 or ratioJ > 0.7 :
                if(scoreI > scoreJ  and eraseFlag[j] == 0):
                    eraseFlag[j] = 1
                    boxes_count = boxes_count - 1
                elif scoreJ > scoreI and eraseFlag[i] == 0:
                    eraseFlag[i] = 1
                    boxes_count = boxes_count - 1
                    break



            continue
            
            
            


                
            if ratioIJ > 0.7:  # choose the max
                eraseFlag[i] = 1
                eraseFlag[j] = 1
                x1 = min(boxI[0], boxJ[0])
                y1 = min(boxI[1], boxJ[1])
                x2 = max(boxI[2], boxJ[2])
                y2 = max(boxI[3], boxJ[3])
                cls_boxes_temp.append([x1, y1, x2, y2])
                eraseFlag.append(0)
                scores_result.append((scores[i] + scores[j]) / 2)
                boxes_count = boxes_count - 1

            elif ratioI > 0.8 and eraseFlag[i] == 0:

                eraseFlag[i] = 1
                boxes_count = boxes_count - 1
            elif ratioJ > 0.8 and eraseFlag[j] == 0:
                eraseFlag[j] = 1
                boxes_count = boxes_count - 1


    cls_boxes_result =[]
    score_result = []
    mask_result = []
    cur_num = 0
    for i in range(len(eraseFlag)):
        if eraseFlag[i] == 1:
            continue
        cls_boxes_result.append(cls_boxes_temp[i])
        score_result.append(score_temp[i])
        mask_result.append(masks_temp[i])
        cur_num = cur_num + 1

    return cls_boxes_result, score_result, mask_result


def writeBoxResult(image_name, img, cls_boxes_hand, scores_hand, cls_boxes_print, scores_print, txt_save_dir):

    file_result = open(txt_save_dir + image_name , 'w')    
    for i in range(len(cls_boxes_hand)):
        box = list(cls_boxes_hand[i])
        box = list(map(str, box))
        label = 1
        score = scores_hand[i]
        write_str = ','.join(box) + ';POLYGON;'+str(score)+';'+str(label)
        file_result.write(write_str+'\n')

    for i in range(len(cls_boxes_print)):
        box = cls_boxes_print[i]
        box = list(cls_boxes_print[i])
        box = list(map(str, box))
        label = 2
        score = scores_print[i]
        write_str = ','.join(box) + ';POLYGON;'+str(score)+';'+str(label)
        file_result.write(write_str+'\n')
def saveBoxResult(image_name, image, cls_boxes_hand, score_hand, cls_boxes_print, score_print, save_dir):
    new_hand_count = len(cls_boxes_hand)
    new_print_count = len(cls_boxes_print)

    imageSrc = image.copy()

    for i in range(len(cls_boxes_hand)):
        box = cls_boxes_hand[i]
        score = score_hand[i]
        box = list(map(int, box))
        if(image_name.find('src')==-1):
            cv2.imwrite("./hand_cut/"+image_name +"_hand_"+ str(i) + ".jpg", image[box[1]:box[3], box[0]:box[2],:])

        cv2.rectangle(imageSrc, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
        cv2.putText(imageSrc, str(score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    imagePrint = image.copy()
    for j in range(len(cls_boxes_print)):
        box = cls_boxes_print[j]
        score = score_print[j]
        box = list(map(int, box))
        # if(abs(box[0]-588) < 5):
        #    continue
        # if(abs(box[0]-929) < 5):
        #    continue
        if(image_name.find('src')==-1):
            cv2.imwrite("./print_cut/"+image_name + "_print_"+ str(j) + ".jpg", image[box[1]:box[3], box[0]:box[2],:])

        cv2.rectangle(imageSrc, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)
        cv2.putText(imageSrc, str(score), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


    cv2.imwrite(save_dir + image_name, imageSrc)

def proBoxMask(poly_map, box_w, box_h):
   
    #start_time_proBoxMask = time.time()
    poly_map = cv2.resize(poly_map, (box_w, box_h))
    #poly_map = poly_map.astype(np.float32) / 255
    #poly_map = cv2.GaussianBlur(poly_map, (3, 3), sigmaX=3)
    ret, poly_map = cv2.threshold(poly_map, 128, 255, cv2.THRESH_BINARY)
    #SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #poly_map = cv2.erode(poly_map, SE1)
    #poly_map = cv2.dilate(poly_map, SE1)
    #poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
    #poly_map = (poly_map * 255).astype(np.uint8)
    
    poly_map = cv2.cvtColor(poly_map, cv2.COLOR_GRAY2BGR)
    #end_time_proBoxMask = time.time()
    return poly_map

def saveMaskResult(token, image, cls_boxes_hand, hand_masks, cls_boxes_print, print_masks, save_dir):
    image_name = token + '.jpg'
    im_poly_show = image.copy()
    b_channel, g_channel, r_channel = cv2.split(im_poly_show)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    alpha_channel[:, :int(b_channel.shape[0])] = 255
    im_poly_show_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    im_poly_show_color_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    hand_mask_points = []
    print_mask_points = []

    # 保存手写mask 
    for index in range(len(cls_boxes_hand)):
        time_start_index = time.time()
        box = cls_boxes_hand[index]
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]

        cls_polys = (hand_masks[index] * 255).astype(np.uint8)
                
        #poly_map = np.array(Image.fromarray(cls_polys))

        poly_map = proBoxMask(cls_polys, box_w, box_h)
        b_channel, g_channel, r_channel = cv2.split(poly_map)

        mask_points = []
        for x in range(0, box_h-1):
            for y in range(0, box_w-1):
                if(b_channel[x,y]==255):
                    mask_points.append(x)
                    mask_points.append(y)
        hand_mask_points.append(mask_points)

        continue

        #白色mask
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        alpha_channel[:, :int(b_channel.shape[0])] = 255
        poly_map_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        cv2.imwrite(save_dir + token +  "_" + str(index) + "_poly_mask.png", poly_map_BGRA)
        continue
        #imageROI = im_poly_show_BGRA[box[1]:box[3], box[0]:box[2]]
        imageROI = cv2.add(imageROI, poly_map_BGRA)
        im_poly_show_BGRA[box[1]:box[3], box[0]:box[2]] = imageROI
        time_end_index = time.time()
        #彩色mask
        #imageROI = im_poly_show_color_BGRA[box[1]:box[3], box[0]:box[2]]
        #imageROI = cv2.add(imageROI, poly_map_BGRA)
        #im_poly_show_color_BGRA[box[1]:box[3], box[0]:box[2]] = imageROI
       
    

    #保存打印体mask
    colorRandom = 0
    for index in range(len(cls_boxes_print)):
        box = cls_boxes_print[index]
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        cls_polys = (print_masks[index] * 255).astype(np.uint8)
        #poly_map = np.array(Image.fromarray(cls_polys))
        poly_map = proBoxMask(cls_polys, box_w, box_h)
        b_channel, g_channel, r_channel = cv2.split(poly_map)
        
        mask_points = []
        for x in range(0, box_h):
            for y in range(0, box_w):
                if(b_channel[x,y]==255):
                    mask_points.append(x)
                    mask_points.append(y)
        print_mask_points.append(mask_points)

        continue

        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        alpha_channel[:, :int(b_channel.shape[0])] = 255
        poly_map_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        if(colorRandom == 0):
            b_channel*= 200
            g_channel*= 0
            r_channel*=0
            colorRandom = 1
        elif(colorRandom == 1):
            g_channel *= 200
            b_channel *= 0
            r_channel *= 0
            colorRandom = 2
        else:
            r_channel *= 200
            b_channel *= 0
            g_channel *= 0
            colorRandom = 0

        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        alpha_channel[:, :int(b_channel.shape[0])] = 255
        poly_map_color_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        
        imageROI = im_poly_show_BGRA[box[1]:box[3], box[0]:box[2]]
        imageROI = cv2.add(imageROI, poly_map_color_BGRA)

        imageROI = cv2.addWeighted(imageROI, 0.8, poly_map_color_BGRA, 0.2, 0)
        im_poly_show_color_BGRA[box[1]:box[3], box[0]:box[2]] = imageROI
        cv2.putText(im_poly_show_color_BGRA, 'h_'+str(index), (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1) 

    #cv2.imwrite(save_dir + token + '_poly_mask_all.jpg', im_poly_show_color_BGRA)
    #cv2.imwrite(save_dir + token + '_poly_mask.jpg', im_poly_show_BGRA)
    return hand_mask_points, print_mask_points


def proBoxWuYue(cls_boxes_hand, scores_hand, mask_hand, cls_boxes_print, scores_print,mask_print):

    eraseFlagHand = [0 for i in range(len(cls_boxes_hand))]
    eraseFlagPrint = [0 for i in range(len(cls_boxes_print))]

    new_hand_count = len(cls_boxes_hand)
    new_print_count = len(cls_boxes_print)

    for hI in range(len(cls_boxes_hand)):
        score_h = scores_hand[hI]
        box_h = cls_boxes_hand[hI].astype(int)
        
        
        if score_h < 0.1 and eraseFlagHand[hI] == 0:
            eraseFlagHand[hI] = 1
            new_hand_count = new_hand_count - 1
            continue
        
        no_coincide = True  # 没有相交的打印体
        for pI in range(len(cls_boxes_print)):
            score_p = scores_print[pI]
            box_p = cls_boxes_print[pI].astype(int)
            ratiohp, ratiop, ratioh = solve_coincide(box_p, box_h)
            ratio_thresh_hand = 0.5
            
            
            if (ratiohp > 0.5):

                if score_h > score_p and eraseFlagPrint[pI] == 0:
                    eraseFlagPrint[pI] = 1  # erase this print boxes
                    new_print_count = new_print_count - 1

                elif score_h < score_p and eraseFlagHand[hI] == 0:
                    eraseFlagHand[hI] = 1  # erase this hand boxes
                    new_hand_count = new_hand_count - 1
                
            elif ratiohp > 0.1:
                if score_p < 0.2 and score_h < score_p:
                    eraseFlagHand[hI] = 1
                    new_hand_count = new_hand_count - 1
                elif score_p < 0.4 and score_p < score_h:
                    eraseFlagPrint[pI] = 1
                    new_print_count = new_print_count - 1

    new_boxes_hand = []
    new_boxes_print = []

    new_mask_hand = []
    new_mask_print = []

    new_scores_hand = []
    new_scores_print = []

    cur_hand_pos = 0
    cur_print_pos = 0

    for i in range(len(eraseFlagPrint)):
        if eraseFlagPrint[i] == 0:
            new_boxes_print.append(cls_boxes_print[i].astype(int))
            new_scores_print.append(scores_print[i])
            new_mask_print.append(mask_print[i])
            cur_print_pos = cur_print_pos + 1

    for i in range(len(eraseFlagHand)):
        if eraseFlagHand[i] == 0:
            new_boxes_hand.append(cls_boxes_hand[i].astype(int))
            new_scores_hand.append(scores_hand[i])
            new_mask_hand.append(mask_hand[i])
            cur_hand_pos = cur_hand_pos + 1

    
    new_boxes_print, new_scores_print , new_mask_print = moveDuplicates(new_boxes_print, new_scores_print, new_mask_print, 'print')
    cur_print_pos = len(new_boxes_print)

    new_boxes_hand, new_scores_hand , new_mask_hand = moveDuplicates(new_boxes_hand, new_scores_hand, new_mask_hand, 'hand')

    return new_boxes_hand, new_scores_hand , new_mask_hand,new_boxes_print, new_scores_print , new_mask_print


label_list = {}
label_list['h'] = 1
label_list['p'] = 2

label_to_str = {v: k for k, v in label_list.items()}

def test_one_image_server(model, image_name, img):


    label_indexs = model.label_indexs
    result = []
    result_js = {"token":image_name, "result": result}

    if img is None:
        print("Could not find %s" % (image_name))
        return json.dumps(result_js) 

    hSrc, wSrc, c = img.shape
    start_time = time.time()

    cls_boxes_hand = []
    cls_boxes_print = []
    scores_hand = []
    scores_print = []
    mask_hand = []
    mask_print = []


    data = model.run_on_opencv_image(img)

    end_time = time.time()
    if not data:
        print("No predictions for image")
        return json.dumps(result_js)
    scores = data["scores"]
    bboxes = data["bboxes"]
    has_labels = "labels" in data
    has_rrects = "rrects" in data
    has_masks = "masks" in data
    has_bbox = 'bboxes' in data
    bboxes = np.round(bboxes).astype(np.int32)
    print('len bboxes ', len(bboxes))
    for ix, (bbox, score) in enumerate(zip(bboxes, scores)):
        if has_labels:
            label = data["labels"][ix]
            label_str = label_indexs[str(int(label)-1)]
            label = label_list[label_str]
        if has_bbox:
            bbox = data["bboxes"][ix]
            if (label == 1):
                cls_boxes_hand.append(bbox)
                scores_hand.append(score)
                mask = data["masks"][ix]
                mask_hand.append(mask)

            elif (label == 2):
                cls_boxes_print.append(bbox)
                scores_print.append(score)
                mask = data["masks"][ix]
                mask_print.append(mask)
            else:
                print('wrong label')
                exit()

    pos = image_name.rfind('.')
    token = image_name[0:pos]

    print('src box , hand  count : ', len(cls_boxes_hand), len(scores_hand), ', print count : ', len(cls_boxes_print), len(scores_print))

    cls_boxes_hand, scores_hand, mask_hand, cls_boxes_print, scores_print, mask_print = proBoxWuYue(cls_boxes_hand,  
                                                                                               scores_hand, mask_hand,
                                                                                               cls_boxes_print,
                                                                                               scores_print, mask_print)
    img_save_dir = './result/show'


    hand_mask_points, print_mask_points = saveMaskResult(token, img, cls_boxes_hand, mask_hand, cls_boxes_print, mask_print, img_save_dir)
    
    #saveBoxResult(token + '_box.jpg', img, cls_boxes_hand, scores_hand, cls_boxes_print, scores_print, './result/show/')
    #writeBoxResult(token + '.jpg.txt', img, cls_boxes_hand, scores_hand, cls_boxes_print, scores_print, './result/txt')
    for i in range(len(cls_boxes_hand)):
        
        x = {}
        label = 2
        score = scores_hand[i]
        box = []

        box.append(int(cls_boxes_hand[i][0]))
        box.append(int(cls_boxes_hand[i][1]))
        box.append(int(cls_boxes_hand[i][2]))
        box.append(int(cls_boxes_hand[i][1]))
        box.append(int(cls_boxes_hand[i][2]))
        box.append(int(cls_boxes_hand[i][3]))
        box.append(int(cls_boxes_hand[i][0]))
        box.append(int(cls_boxes_hand[i][3]))
        #box.append(round(float(score),2))
        #box.append(label)
        #result.append(box)

        x['class' ] = label
        x['score'] = round(float(score),2)
        x['box'] = box
        x['mask'] = hand_mask_points[i]
        result.append(x)

    for i in range(len(cls_boxes_print)):
        score = scores_print[i]
        label = 1
        box = []
        x={}
        box.append(int(cls_boxes_print[i][0]))
        box.append(int(cls_boxes_print[i][1]))
        box.append(int(cls_boxes_print[i][2]))
        box.append(int(cls_boxes_print[i][1]))
        box.append(int(cls_boxes_print[i][2]))
        box.append(int(cls_boxes_print[i][3]))
        box.append(int(cls_boxes_print[i][0]))
        box.append(int(cls_boxes_print[i][3]))
        #box.append(round(float(score), 2))
        #box.append(label)
        #result.append(box)
        x['class'] = label
        x['score'] = round(float(score),2)
        x['box'] = box
        x['mask'] = print_mask_points[i]
        result.append(x)
    
    result_js["result"] = result

    result_str = json.dumps(result_js)
    return result_str

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wait',
        dest='wait',
        help='wait until net file exists',
        default=True,
        type=bool
    )
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true'
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference',
        action='store_true'
    )
    parser.add_argument(
        '--range',
        dest='range',
        help='start (inclusive) and end (exclusive) indices',
        default=None,
        type=int,
        nargs=2
    )
    parser.add_argument(
        'opts',
        help='See lib/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def base64_to_image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img 

def test_one_base64(model,image_name , base64_data):
    img = base64_to_image(base64_data)
    return   test_one_image_server(model, image_name, img)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    
    parser.add_argument(
        "-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    
    args = parser.parse_args()

    model_file = './save/mask_rcnn_0410/model_0120000.pth'
    config_file = './save/mask_rcnn_0410/config.yml'
    label_path = './save/mask_rcnn_0410/labels.json'	

    #模型初始化
    prediction_model = Predictor(config_file, label_path)
    prediction_model.load_weights(model_file)

    f = open('./test_images/1EnglishBlank_1.jpg','rb')

    base64_data = base64.b64encode(f.read())
    base64_data = base64_data.decode()
    image_name = 'test'
    
    #参数  初始化好的模型, 图片名字, base64图像数据, 返回 json字符串
    result_str = test_one_base64(prediction_model, image_name, base64_data)
    print(result_str)
  
    '''    
    img_dir = './test_images/'
    count = 0
    time_total = 0
    for image_name in open('test.list','r'):
    #for image_name in os.listdir(img_dir):
        image_name = image_name.strip()
        print(image_name)   
        img = cv2.imread(img_dir + image_name)
        time_start = time.time()
        result_str = test_one_image_server(prediction_model, image_name, img)
        time_end = time.time()
        print(result_str)
        count = count + 1
        time_total += time_end

    time_avg = time_total / count
    print(time_avg)
    '''
    
