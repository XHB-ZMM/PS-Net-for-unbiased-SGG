# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import numpy as np
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.data import get_dataset_statistics
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..roi_heads.relation_head.SaveFasterRCNNFeats import FasterRCNNFeats
from ..roi_heads.relation_head.myLSTMContext import myLSTMContext,myFrequencyBias,myDecoderRNN

from ..roi_heads.relation_head.sampling import make_roi_relation_samp_processor
from ..roi_heads.relation_head.roi_relation_feature_extractors import make_roi_relation_feature_extractor

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks. #支持box和mask
    It consists of three main parts: 包含三个部分
    - backbone 主干提取网络
    - rpn rpn 特征金字塔
    - heads: takes the features + the proposals from the RPN and computes #从rpn获得建议框和对应特征，并计算
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg) #xhb:R-101-FPN
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)


        # self.statistics = get_dataset_statistics(cfg)
        # self.obj_classes = self.statistics['obj_classes']
        # self.rel_classes = self.statistics['rel_classes']
        # self.getFasterRCNNFeats = FasterRCNNFeats(cfg,self.obj_classes,self.rel_classes)
        #
        # self.ValueFeats_dict_list = []
        # self.PretrainingFeats_dict_list = []
        # self.TestFeats_dict_list = []

        # self.samp_processor = make_roi_relation_samp_processor(cfg)
        # self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, self.backbone.out_channels)
        # self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        # self.union_feats_list_dict = []


    def forward(self, images, targets=None, logger=None): #xhb：进这个forward算损失
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.#xhb：模型的输出
            在训练阶段：返回的就是损失，以字典的形式存储，数据格式为tensor
            在测试阶段：返回的不只是损失，还有分数、标签、mask之类的一些附加字段
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """


        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images) #xhb:将图像变成列表形式
        '''
        xhb:features一共有5个，形状分别是：尺寸下采样减半
                torch.Size([2, 256, 152, 256])
                torch.Size([2, 256, 76, 128])
                torch.Size([2, 256, 38, 64])
                torch.Size([2, 256, 19, 32])
                torch.Size([2, 256, 10, 16])
            proposals:box的尺寸是torch.Size([1016, 4])，此刻这张图像上有1016个框，每个框有4个参数
            proposal_losses有两个，一个是预测目标的损失，一个是box框的回归损失
                'loss_objectness'  ：tensor(0.0876, device='cuda:0')
                'loss_rpn_box_reg' ：tensor(0.0921, device='cuda:0')   
        '''
        features = self.backbone(images.tensors) #xhb:R-101-FPN
        proposals, proposal_losses = self.rpn(images, features, targets) #xhb：Faster-RCNN 输出的proposal
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, logger)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        #-------- save union_feature ---------#
        # if self.training:
        #     # relation subsamples and assign ground truth label during training
        #     with torch.no_grad():
        #         if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        #             _, rel_labels, rel_pair_idxs, _ = self.samp_processor.gtbox_relsample(result, targets)
        #         else:
        #             _, rel_labels, rel_pair_idxs, _ = self.samp_processor.detect_relsample(result, targets)
        # else:
        #     rel_labels, rel_binarys = None, None
        #     rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, result)
        #
        #
        # if self.use_union_box:
        #     union_features = self.union_feature_extractor(features, result, rel_pair_idxs)
        # else:
        #     union_features = None
        #
        # # ---------------save union_features in training-----------------#
        # if not self.training:
        #     with torch.no_grad():
        #         num_objs = [len(b) for b in rel_pair_idxs]
        #         union_feature = union_features.split(num_objs, dim=0)
        #         for i in range(len(union_feature)):
        #             image_id = targets[i].get_field("image_id")
        #             # 有关系的，save union feature，没关系的不存
        #             relation_map = targets[i].get_field("relation")
        #             relation_mat = relation_map.cpu().numpy() # cuda tensor no direct convert numpy
        #             gt_rel_idx = np.where(relation_mat != 0) # return idx (!=0) in relation_mat
        #             for k in range(len(gt_rel_idx[0])):
        #                 sub_id = gt_rel_idx[0][k]
        #                 obj_id = gt_rel_idx[1][k]
        #                 idx = None
        #                 for p in range(rel_pair_idxs[i].shape[0]):
        #                     if int(sub_id) == int(rel_pair_idxs[i][p][0]) and int(obj_id) == int(rel_pair_idxs[i][p][1]):
        #                         idx = p
        #                         break
        #                     else:
        #                         continue
        #                 union_feats_dict = {
        #                     "image_id": image_id,
        #                     "sub_id": sub_id,
        #                     "obj_id": obj_id,
        #                     "union_feats": union_feature[i][idx].data,
        #                 }
        #                 self.union_feats_list_dict.append(union_feats_dict)


        # # 得到训练集的特征文件
        # if self.training:
        #     with torch.no_grad():
        #         obj_pre_rep, obj_dists, obj_ctx = self.getFasterRCNNFeats(x,result) #得到一个batch的所有特征
        #         num_rois = [len(b) for b in result]  #得到每幅图像里box数量
        #         obj_pre_rep = obj_pre_rep.split(num_rois, dim=0)  #得到每幅图像的object feature
        #         obj_dists = obj_dists.split(num_rois, dim=0)
        #         obj_ctx = obj_ctx.split(num_rois, dim=0)
        #         # PretrainingFeats_dict_list = []
        #         for i in range(len(num_rois)):
        #             per_feats = obj_pre_rep[i] #第i张图像里的所有object特征
        #             per_obj_dists = obj_dists[i] #第i张图像里的所有objetc的预测概率分布
        #             per_obj_ctx = obj_ctx[i] #第i张图像里的所有object的上下文特征
        #             image_id = targets[i].get_field("image_id")  # 得到当前图像的id
        #             PretrainingFeats_dict = {
        #                 "image_id": image_id,
        #                 "feats": per_feats,
        #                 "obj_dist": per_obj_dists,
        #                 "obj_ctx": per_obj_ctx,
        #             }
        #             self.PretrainingFeats_dict_list.append(PretrainingFeats_dict) #将一个batch里的数据存储
        # else:
        #     # 得到评估集的特征文件
        #     obj_pre_rep, obj_dists, obj_ctx = self.getFasterRCNNFeats(x, result)  # 得到一个batch的所有特征
        #     num_rois = [len(b) for b in result]  # 得到每幅图像里box数量
        #     obj_pre_rep = obj_pre_rep.split(num_rois, dim=0)  # 得到每幅图像的object feature
        #     obj_dists = obj_dists.split(num_rois, dim=0)
        #     obj_ctx = obj_ctx.split(num_rois, dim=0)
        #
        #     for i in range(len(num_rois)):
        #         per_feats = obj_pre_rep[i]  # 第i张图像里的所有box特征 4424
        #         per_obj_dists = obj_dists[i]
        #         per_obj_ctx = obj_ctx[i]
        #
        #         image_id = targets[i].get_field("image_id")  # 得到当前图像的id
        #         PrevaluingFeats_dict = {
        #             "image_id": image_id,
        #             "feats": per_feats,
        #             "obj_dist": per_obj_dists,
        #             "obj_ctx": per_obj_ctx,
        #         }
        #         self.ValueFeats_dict_list.append(PrevaluingFeats_dict)  # 将一个batch里的数据存储


        # # 得到测试集的特征文件
        # obj_pre_rep, obj_dists, obj_ctx = self.getFasterRCNNFeats(x, result)  # 得到一个batch的所有特征
        # num_rois = [len(b) for b in result]  # 得到每幅图像里box数量
        # obj_pre_rep = obj_pre_rep.split(num_rois, dim=0)  # 得到每幅图像的object feature
        # obj_dists = obj_dists.split(num_rois, dim=0)
        # obj_ctx = obj_ctx.split(num_rois, dim=0)
        #
        # for i in range(len(num_rois)):
        #     per_feats = obj_pre_rep[i]  # 第i张图像里的所有box特征 4424
        #     per_obj_dists = obj_dists[i]
        #     per_obj_ctx = obj_ctx[i]
        #
        #     image_id = targets[i].get_field("image_id")  # 得到当前图像的id
        #     PretestingFeats_dict = {
        #         "image_id": image_id,
        #         "feats": per_feats,
        #         "obj_dist": per_obj_dists,
        #         "obj_ctx": per_obj_ctx,
        #     }
        #     self.TestFeats_dict_list.append(PretestingFeats_dict)  # 将一个batch里的数据存储



        if self.training:
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON:
                # During the relationship training stage, the rpn_head should be fixed, and no loss.
                losses.update(proposal_losses)
            return losses

        return result
