      # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import torch.nn as nn
#from maskrcnn_benchmark.modeling.roi_heads.box_head.model_utils import CenterLoss
#from maskrcnn_benchmark.modeling.roi_heads.box_head.CenterLoss import CenterLoss
from maskrcnn_benchmark.modeling.roi_heads.box_head.center_loss import CenterLoss
from torch.autograd import Variable

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.roi_heads.box_head import net as net_utils
#import ipdb

def ohem_loss(batch_size, cls_pred, cls_target, loc_pred, loc_target):   
    """    Arguments:
     batch_size (int): number of sampled rois for bbox head training      
     loc_pred (FloatTensor): [R, 4], location of positive rois      
     loc_target (FloatTensor): [R, 4], location of positive rois   
     pos_mask (FloatTensor): [R], binary mask for sampled positive rois   
     cls_pred (FloatTensor): [R, C]     
     cls_target (LongTensor): [R]  
     Returns:    
           cls_loss, loc_loss (FloatTensor)
    """

    #print(loc_pred.size(), loc_target.size())
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)
    #ohem_loc_loss = smooth_l1_loss(loc_pred, loc_target).sum(dim=1)

    ohem_loc_loss = smooth_l1_loss(
        loc_pred, loc_target,
        size_average=False,
        beta=1,
    )

    #sl1_loss_bbox = sl1_loss_bbox / labels.numel()
    # 这里先暂存下正常的分类loss和回归loss
    #print(ohem_cls_loss.shape, ohem_loc_loss.shape)
    loss = ohem_cls_loss + ohem_loc_loss
    # 然后对分类和回归loss求和
    
    sorted_ohem_loss, idx = torch.sort(loss, descending=True)   
    # 再对loss进行降序排列
    
    #print(sorted_ohem_loss.size()[0],'vs', batch_size)

    keep_num = min(sorted_ohem_loss.size()[0], batch_size)    
    # 得到需要保留的loss数量
    
    if keep_num < sorted_ohem_loss.size()[0]:    
        # 这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留
    
        keep_idx_cuda = idx[:keep_num]        # 保留到需要keep的数目
        print(keep_idx_cuda)
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]      
        ohem_loc_loss = ohem_loc_loss[keep_idx_cuda]        # 分类和回归保留相同的数目

        print('keep_num : ', keep_num)
        
    cls_loss = ohem_cls_loss.sum() / keep_num   
    loc_loss = ohem_loc_loss.sum() / keep_num    # 然后分别对分类和回归loss求均值
    return cls_loss, loc_loss

class FocalSigmoidLossFunc(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    def forward(ctx, logits, label, alpha, gamma, reduction):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha

        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1. - probs) ** gamma

        ctx.coeff = coeff
        ctx.probs = probs
        ctx.log_probs = log_probs
        ctx.log_1_probs = log_1_probs
        ctx.probs_gamma = probs_gamma
        ctx.label = label
        ctx.gamma = gamma
        ctx.reduction = reduction

        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        if reduction == 'mean':
            loss = loss.mean()
        if reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        coeff = ctx.coeff
        probs = ctx.probs
        log_probs = ctx.log_probs
        log_1_probs = ctx.log_1_probs
        probs_gamma = ctx.probs_gamma
        probs_1_gamma = ctx.probs_1_gamma
        label = ctx.label
        gamma = ctx.gamma
        reduction = ctx.reduction

        term1 = (1. - probs - gamma * probs * log_probs).mul_(probs_1_gamma).neg_()
        term2 = (probs - gamma * (1. - probs) * log_1_probs).mul_(probs_gamma)

        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(grad_output)
        if reduction == 'mean':
            grads = grads.div_(label.numel())
        if reduction == 'sum':
            grads = grads
        return grads, None, None, None, None

class FocalLossV2(nn.Module):
    '''
    This use better formula to compute the gradient, which has better numeric stability
    '''
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        return FocalSigmoidLossFunc.apply(logits, label, self.alpha, self.gamma, self.reduction)


class AMSoftmax(nn.Module):
    def __init__(self):
        super(AMSoftmax, self).__init__()

    def forward(self, input, target, scale=2, margin=0.5):
        # self.it += 1
        
        cos_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= margin
        
        
        output = output * scale

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss

class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        cfg,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.cfg = cfg
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        
        centerloss = CenterLoss(num_classes=3, feat_dim=1024)
        self.centerloss = centerloss.cuda()
        self.AmSoftmax = AMSoftmax()
        
	#self.focalloss = FocalLossV2()
    def match_targets_to_proposals(self, proposal, target):
        
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            #这个Encode的作用是实现RCNN中提到的边框回归，其中的回归目标（regression target)t*
            #        的计算，主要是计算候选框与与之相关的基准框的偏差
            # 和 detectron 中的 roi_data/fast_rcnn.py _sample_rois 功能一样
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
        return labels, regression_targets


    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
      
         
        labels, regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    # shanhaijiao add x_features

    def __call__(self, class_logits, box_regression, x_features):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        #box_regression 为输入 fastrcnn 的 predictor 结果
        #regression_targets 为subsample之后的结果 经过了BoxCoder

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)

        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )
        #print('in loss box_regression : ', box_regression)
        #print('in loss regression_targets : ', regression_targets)
        #print('in loss labels:', labels)
        classification_loss = F.cross_entropy(class_logits, labels)

        #classification_loss = self.focalloss(class_logits, labels)

        #loss_0 =  F.cross_entropy(class_logits, labels)
        #loss_1  = self.centerloss(labels, x_features)
        #loss_1  = self.centerloss(x_features, labels)
        #loss_weight = 0.01*0.001
        #classification_loss = loss_0 + loss_weight*loss_1

        '''
        nllloss = nn.CrossEntropyLoss()
        nllloss = nllloss.cuda()
        centerloss = CenterLoss(num_classes=3, feat_dim=3, use_gpu = True )
        centerloss = centerloss.cuda()
        self.criterion = [nllloss, centerloss]
        '''
        #before
        classification_loss = F.cross_entropy(class_logits, labels)
        #classification_loss = self.AmSoftmax(class_logits, labels) 


        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
            
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)
    
        map_inds_src =  4 * labels[:, None] + torch.tensor([0, 1, 2, 3], device=device)
        
        #*************原始计算box_loss代码
        # regression_targets size: 512 * 4 
        #box_regression size 512*12, 其中12维表示 3个类*4个边框数值
        # map_inds 内容为 4,5,6,7 或  8,9,10,11 表示取 第一类或者第二类的索引
        
        sl1_loss_bbox = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        sl1_loss_bbox = sl1_loss_bbox / labels.numel()
        #***************原始代码结束

        #ohem
        #cls_loss, loc_loss = ohem_loss(int(labels.numel()/2), class_logits, labels, box_regression[sampled_pos_inds_subset[:, None], map_inds],regression_targets[sampled_pos_inds_subset])

        #print('old :', classification_loss, sl1_loss_bbox, cls_loss, loc_loss)

        #return cls_loss, loc_loss

       
        bbox_pred = box_regression # 
        #bbox_targets = regression_targets
        #print('in loss , regression_targets', regression_targets)
        
        
        bbox_targets = torch.zeros(regression_targets.size()[0],4*self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, device='cuda') 
        #bbox_targets[:, map_inds_src] = regression_targets
        for index in range(bbox_targets.size()[0]):
            for p in range(4):
                bbox_targets[index,map_inds_src[0]] = regression_targets[index]
        
        
        bbox_outside_weights = torch.zeros(regression_targets.size()[0],4*self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, device='cuda') 
        bbox_outside_weights[[sampled_pos_inds_subset[:, None], map_inds]] = 1.0
         
        bbox_inside_weights = torch.zeros(regression_targets.size()[0],4*self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, device='cuda')
        bbox_inside_weights[[sampled_pos_inds_subset[:, None], map_inds]] =1.0
  
         
        #sl1_loss_bbox = net_utils.smooth_l1_loss(
        #    bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        #transform_weights 已经转换过
        #iou_loss_bbox, giou_loss_bbox = net_utils.compute_giou(
        #    bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights,
        #    transform_weights=None)
       

        loss_type = 'smooth_l1'
        #loss_type = 'ciou'
        if loss_type == 'smooth_l1':
            loss_bbox = sl1_loss_bbox
        elif loss_type == 'iou':
            loss_bbox = iou_loss_bbox
        elif loss_type == 'giou':
            loss_bbox = giou_loss_bbox
        elif loss_type == 'diou':
            _, diou_loss_bbox = net_utils.compute_diou(
                bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights,
                transform_weights=None)
            loss_bbox = diou_loss_bbox
        elif loss_type == 'ciou':
            _, ciou_loss_bbox = net_utils.compute_ciou(
                bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights,
                transform_weights=None)
            #print('loss, ciou_loss_bbox : ', ciou_loss_bbox)
            loss_bbox = ciou_loss_bbox
        else:
            raise ValueError('Invalid loss type: ', loss_type )
        

        # class accuracy
        #cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
        #accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)
        

        #return loss_cls, loss_bbox, accuracy_cls, sl1_loss_bbox, iou_loss_bbox, giou_loss_bbox


        return classification_loss, loss_bbox


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        cfg,
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
