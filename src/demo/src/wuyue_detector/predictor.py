import numpy as np
import cv2
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
from torchvision import transforms
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list

from maskrcnn_benchmark.utils.model_serialization import load_state_dict
import json

import time
import random

class MyResize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        im_scale = 1.0
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w), im_scale

        if w < h:
            ow = size
            oh = int(size * h / w)
            im_scale = float(ow) / w
        else:
            oh = size
            ow = int(size * w / h)
            im_scale = float(oh) / h

        return (oh, ow), im_scale

    def __call__(self, image):
        w,h,_= image.shape
        size , im_scale = self.get_size((w,h))
        image = cv2.resize(image, size)
        
        return image

def normalize (image, mean, std):

    c, height, width = image.shape
    
    image_mean = np.ones( image.shape, dtype=np.float32 )   
    #image_mean = np.array(height, width, 3)
    image_mean [0,:,:] = mean[0]
    image_mean [1,:,:] = mean[1]
    image_mean [2,:,:] = mean[2]
     
    image = image - image_mean

    return image

                              
class PseudoTarget(object):
    def __init__(self):
        self.x = {}

    def add_field(self, k, v):
        self.x[k] = v

    def get_field(self, k):
        return self.x[k]

    def resize(self, *args):
        return self

    def transpose(self, *args):
        return self



class ImageTransformer(object):
    def __init__(self, cfg):
        self.cfg = cfg.clone()
	
        #现在
        self.resize_func = MyResize(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
        

    '''
    def build_transform(self):
        transforms = build_transforms(self.cfg, is_train=False)
        print('transforms : ', transforms)

        return transforms
    '''

    def transform_image(self, original_image):
        
        time_tr_1 = time.time()
        #1204 修改
        image = self.resize_func(original_image)
        image = np.transpose(image, (2,0,1))
        image = image [[2,1,0]]
        
        image = normalize(image, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        
        image = torch.Tensor(image)

        time_tr_2 = time.time()

        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        
        return image_list
    

def paste_mask_on_image(mask, box, im_h, im_w, thresh=None, interp=cv2.INTER_LINEAR, rotated=False):

    box = np.round(box).astype(np.int32)
    if rotated:
        assert len(box) == 5  # xc,yc,w,h,angle
        w = box[2]
        h = box[3]
    else:
        assert len(box) == 4  # x1,y1,x2,y2
        w = box[2] - box[0] + 1
        h = box[3] - box[1] + 1

    w = max(w, 1)
    h = max(h, 1)

    resized = cv2.resize(mask, (w, h), interpolation=interp)

    if thresh is not None:#thresh >= 0:
        resized = (resized > thresh).astype(np.float32)
    canvas = np.zeros((im_h, im_w), dtype=np.float32)

    if rotated:
        from maskrcnn_benchmark.modeling.rotate_ops import paste_rotated_roi_in_image

        canvas = paste_rotated_roi_in_image(canvas, resized, box)

    else:
        x_0 = max(box[0], 0)
        x_1 = min(box[2] + 1, im_w)
        y_0 = max(box[1], 0)
        y_1 = min(box[3] + 1, im_h)

        canvas[y_0:y_1, x_0:x_1] = resized[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]
    # cv2.imshow("canvas", canvas)
    # cv2.waitKey(0)
    return canvas


def select_top_predictions(predictions, confidence_threshold=0.7, score_field="scores"):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field(score_field)
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    if len(keep) == 0:
        return []
    predictions = predictions[keep]
    scores = predictions.get_field(score_field)
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]


def load_model(model, f):
    #print('begin load model')
    checkpoint = torch.load(f, map_location=torch.device("cpu"))
    #checkpoint = checkpoint.module
    load_state_dict(model, checkpoint.pop("model"))
    #print("Loaded %s"%(f))


class Predictor(object):
    def __init__(self, config_file, label_path, min_score=0.8, mask_thresh=0.5, device="cuda"):

        """
        mask_thresh: [0,1] or None. 
        If value is [0,1], performs binary thresh
        If value is None, ignore binary thresholding
        """
        
        cfg.merge_from_file(config_file)
        self.label_indexs = json.load(open(label_path))
        #print(self.label_indexs)

        self.cfg = cfg
        self.min_score = min_score
        self.mask_thresh = mask_thresh

        self.device = device
        self.cpu_device = torch.device("cpu")

        self.model = self.build_model()
        self.img_transformer = ImageTransformer(self.cfg)

        self.cnt = 0

        self.score_field = "scores"
        if cfg.MODEL.RPN_ONLY:
            self.score_field = "objectness"
        elif cfg.MODEL.MASKIOU_ON:
            self.score_field = "mask_scores"

    def get_data_from_prediction(self, predictions, img_height, img_width):

        data = {
            "scores": predictions.get_field(self.score_field).numpy(),  # from roi box head,
            "bboxes": predictions.bbox.numpy()
        }
        
        if not self.cfg.MODEL.RPN_ONLY:
            data["labels"] = predictions.get_field("labels").numpy()

        rotated = self.cfg.MODEL.ROTATED
        if rotated:
            data["rrects"] = predictions.get_field("rrects").rbox.cpu().numpy()

        if self.cfg.MODEL.MASK_ON:
            masks = predictions.get_field("mask").numpy().squeeze(1)

            N = len(masks)

            boxes = data["bboxes"] if not rotated else data["rrects"]
            assert N == len(boxes)

            is_pp_mask = self.cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS

            #final_masks = np.zeros((N, img_height, img_width), dtype=np.float32)
            final_masks = []
            for ix in range(N):
                box = boxes[ix]
                #if not is_pp_mask:
                #    mask = paste_mask_on_image(masks[ix], box, img_height, img_width, thresh=self.mask_thresh, rotated=rotated)
                #else:
                #mask = cv2.resize(masks[ix], (img_width, img_height))
                final_masks.append(masks[ix])
                #final_masks[ix] = masks[ix]

            data["masks"] = final_masks

        return data

    def run_on_opencv_image(self, img):
        # assert len(img.shape) == 3 and img.shape[-1] == 3  # image must be in (H,W,3) dims
        #print('run on opencv image')
        run_start_time = time.time()

        height, width, cn = img.shape
        #print('height : ', height,'width :', width)
        if self.cnt == 0:
            self.model.to(self.device)
        

        self.cnt += 1

        # Change BGR to RGB, since torchvision.transforms use PIL image (RGB default...)

        time1 = time.time()

        image_list = self.img_transformer.transform_image(img[:,:,::-1])
        

        time2 = time.time()
        print('transform time : ', time2 - time1)


        image_list = image_list.to(self.device)
        time3 = time.time()

        prediction_start_time = time.time()
        with torch.no_grad():
            predictions = self.model(image_list)
        prediction_end_time = time.time()
        print('predictions time : ', prediction_end_time - prediction_start_time)

        predictions = [o.to(self.cpu_device) for o in predictions]

        time4 = time.time()

        if len(predictions) == 0:
            return None
        
        predictions = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        predictions = predictions.resize((width, height))
        #predictions = select_top_predictions(predictions, self.min_score, self.score_field)

        if len(predictions) == 0:
            return []
        
        time5 = time.time()
        data = self.get_data_from_prediction(predictions, height, width)
        run_end_time = time.time()
        print('run on opencv time : ', run_end_time - run_start_time)


        return data

    def build_model(self):
        # BASE MODEL
        model = build_detection_model(self.cfg)
        model.eval()
  
        return model

    def load_weights(self, model_file):
        load_model(self.model, model_file)
