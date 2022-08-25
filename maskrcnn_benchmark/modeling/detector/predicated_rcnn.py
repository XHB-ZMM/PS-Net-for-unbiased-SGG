# Copyright (c) xuhongbo
"""
Implements the Predicated R-CNN framework
"""

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from maskrcnn_benchmark.data import get_dataset_statistics
from ..roi_heads.relation_head.SaveFasterRCNNFeats import FasterRCNNFeats
# from .utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, encode_box_info
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors,to_onehot
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_motifs import FrequencyBias
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_relation import layer_init

class PredicatedRCNN(nn.Module):


    def __init__(self, cfg):
        super(PredicatedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.feats_dim = 4424
        self.hidden_size = 1024
        self.num_obj_classes = 151
        self.num_rel_classes = 51
        self.ctx_dim = 512
        self.edge_hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM # 512
        self.embed_dim = cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM # 200
        self.post_emb = nn.Linear(self.edge_hidden_dim, self.edge_hidden_dim * 2)  # 512->1024
        self.edge_lin = nn.Linear((self.embed_dim + self.feats_dim+ self.ctx_dim)*2, self.edge_hidden_dim) # 10272->512


        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM # 4096
        self.post_cat = nn.Linear(self.edge_hidden_dim*2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_classes, bias=True)

        # initialize layer parameters
        # layer_init(self.post_emb, 10.0 * (1.0 / self.edge_hidden_dim) ** 0.5, normal=True)
        # layer_init(self.post_cat, xavier=True)
        # layer_init(self.edge_lin, xavier=True)
        # layer_init(self.rel_compress, xavier=True)

        # load class dict
        statistics = get_dataset_statistics(cfg)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_classes == len(obj_classes)
        assert self.num_rel_classes == len(rel_classes)

        # 映射输入物体的特征的维度
        self.out_obj = nn.Sequential( # 4424 -> 151
            nn.Linear(self.feats_dim, self.hidden_size),
            # nn.BatchNorm()
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.num_obj_classes),
            # nn.BatchNorm()
            nn.ReLU(inplace=True),
        )

        # 对obj—pred进行word vector编码
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim) # [151,200] 包含所有类
        self.edge_obj_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.sub_obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.sub_obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.edge_obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        # 计算交叉熵损失函数
        self.criterion_loss = nn.CrossEntropyLoss()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.use_bias = True
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(self.cfg, statistics)

        self.down_dim = nn.Linear(4096, 2048)
        layer_init(self.down_dim, xavier=True)
        self.up_dim = nn.Linear(2048, 4096)
        layer_init(self.up_dim, xavier=True)

    def forward(self, train_data, target, union_feats = None,logger=None): #xhb：进这个forward算损失

        # get object feature
        sub_feats = train_data['sub_feats']
        obj_feats = train_data['obj_feats']

        # get object pretraining dist
        pre_sub_dist = train_data['sub_dist']
        pre_obj_dist = train_data['obj_dist']

        # get objecy pretraining context feature
        pre_sub_ctx = train_data['sub_ctx']
        pre_obj_ctx = train_data['obj_ctx']


        # get object logits
        # sub_dists = self.out_obj(sub_feats) # 对sub进行标签分类，得到151维度的logits  torch.Size([151])
        # obj_dists = self.out_obj(obj_feats) # 对obj进行标签分类，得到151维度的logits  torch.Size([151])

        # get object pred
        sub_logits = F.softmax(pre_sub_dist.view(1,-1), dim=1)
        obj_logits = F.softmax(pre_obj_dist.view(1,-1), dim=1)
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL : # PREDCLS
            sub_pred = torch.tensor(int(target['sub_labels'])).long().view(1).to(self.device)
            obj_pred = torch.tensor(int(target['obj_labels'])).long().view(1).to(self.device)
        else: #SGCLS
            sub_pred = torch.argmax(sub_logits).view(1)
            obj_pred = torch.argmax(obj_logits).view(1)

        # get label embd feature 缺点：如果上面预测的不准，这就很不准
        sub_embed = self.edge_obj_embed(sub_pred.long())
        obj_embed = self.edge_obj_embed(obj_pred.long())

        # get edge features
        sub_rel_rep = torch.cat((sub_embed.view(1,-1), sub_feats.view(1,-1), pre_sub_ctx.view(1,-1)), -1) # 5136 = 200 + 4424 + 512
        obj_rel_rep = torch.cat((obj_embed.view(1,-1), obj_feats.view(1,-1), pre_obj_ctx.view(1,-1)), -1)

        # get relation head-tail feature
        edge_feats = torch.cat((sub_rel_rep,obj_rel_rep),-1) # 先将sub和obj的合并特征拼接起来
        edge_feats = self.edge_lin(edge_feats) # 映射到512 dim
        edge_rep = self.post_emb(edge_feats) # 再映射到1024 dim
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_hidden_dim) # [1,512] dim
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_hidden_dim) # [1,512] dim

        # get relation dist
        # head_rep = self.post_cat(head_rep) # 将head映射到4096维度
        # tail_rep = self.post_cat(tail_rep) # 将tail映射到4096维度
        # prod_rep = head_rep * tail_rep

        # get relation dist
        prod_rep = torch.cat((head_rep, tail_rep), dim=-1)
        prod_rep = self.post_cat(prod_rep)
        if union_feats != None:
            relation_rep = prod_rep * self.up_dim(self.down_dim(union_feats))  # 得到关系特征,先降维再升维
        else:
            relation_rep = prod_rep # 得到关系特征
        relation_dists = self.rel_compress(relation_rep) # 关系分类

        sub_label = torch.tensor(int(target['sub_labels'])).long().view(1).to(self.device)
        obj_label = torch.tensor(int(target['obj_labels'])).long().view(1).to(self.device)
        if self.use_bias:
            pair_pred = torch.stack((sub_label, obj_label), dim=-1)  # 用两个label是不是更好
            relation_dists = relation_dists + self.freq_bias.index_with_labels(pair_pred.long())

        relation_logits = F.softmax(relation_dists,dim=1)
        new_relation_logits = relation_logits.view(-1).tolist()[1:]
        new_relation_logits = torch.Tensor(new_relation_logits)
        predicate_pred = torch.argmax(new_relation_logits).view(1) # 不预测关系label = 0的三元组

        # get <subject predicate object> label
        rel_label = torch.tensor(int(target['predicate_label'])).long().view(1).to(self.device)

        # 获得分类置信度
        sub_index = int(sub_pred)
        lsit_sub = sub_logits.view(-1).tolist()
        sub_scores = lsit_sub[sub_index] # subject
        obj_index = int(obj_pred)
        lsit_obj = obj_logits.view(-1).tolist()
        obj_scores = lsit_obj[obj_index] # object
        predicate_index = int(predicate_pred) + 1 # 因为把0去掉了，所以其他类别得加1
        lsit_predicate = relation_logits.view(-1).tolist()
        predicate_scores = lsit_predicate[predicate_index] # predicate


        # 训练模式下，返回loss
        if self.training:
            # calculate object loss
            # sub_loss = self.criterion_loss(pre_sub_dist.view(1,-1),sub_label)
            # obj_loss = self.criterion_loss(pre_obj_dist.view(1,-1),obj_label)

            # calculate relation loss
            rel_loss = self.criterion_loss(relation_dists.view(1,-1), rel_label)

            # output_losses = dict(loss_rel=rel_loss, loss_refine_sub=sub_loss, loss_refine_obj=obj_loss)
            output_losses = dict(loss_rel=rel_loss)

            return output_losses

        # 评估模式下，返回预测结果
        elif self.cfg.SOLVER.TO_VAL:

            predictions = dict(sub_pred=sub_index,
                               obj_pred=obj_index,
                               predicate_pred=predicate_index,
                               sub_scores=sub_scores,
                               obj_scores=obj_scores,
                               predicate_scores=predicate_scores
                               )

            return predictions




