# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import cv2
import numpy as np
class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        print('begin build self.backbone')
        self.backbone = build_backbone(cfg)
        print('begin build rpn')
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        print('begin build roi_heads')
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        print('build roi heads end')
    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
       
        ''' 
        c,h,w = images.tensors[0].size()
        b_channel = images.tensors[0].cpu().numpy()[0]+100
        g_channel = images.tensors[0].cpu().numpy()[1]+100
        r_channel = images.tensors[0].cpu().numpy()[2]+100

        cvImage = cv2.merge((b_channel, g_channel, r_channel))


        color = (0,0,255)
        for box in proposals[0].bbox:
            x0,y0,x1,y1 = list(map(int,box.cpu().numpy()))
            if((x0 > 320 and y0 > 540 and x1  < 800 and y1  < 850)):
                    #(x0 > 600 and y0 > 600 and x1 < 800 and y1 < 800)):
                cv2.rectangle(cvImage, (x0,y0,(x1-x0+1),(y1-y0+1)), color, 1,1)
                if(color[2]==255):
                    color=(255,0,0)
                else:
                    color = (0,0,255)
                    
                    
        cv2.imwrite('image.png', cvImage)
        '''
        

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
