import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import encode_box_info,obj_edge_vectors
from .myLSTMContext import myLSTMContext


class FasterRCNNFeats(nn.Module):
    def __init__(self,cfg,obj_classes,rel_classes):
        super(FasterRCNNFeats, self).__init__()

        self.cfg = cfg
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        self.num_rel_classes = len(rel_classes)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.myLSTM = myLSTMContext(self.cfg,self.obj_classes,self.rel_classes, 4096)
        self.myLSTM.load_state_dict(torch.load("/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/modulepth/LSTMContext_state_dict_final_fortest.pth"))
        print("self.myLSTM.load_state_dict")

    def forward(self, x, proposals): # 传入4096的x特征 和 proposal

        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long()) # PredCls
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach() #SGCLS
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))

        obj_pre_rep = cat((x, obj_embed, pos_embed), -1)


        # 加载训练好的
        obj_dists, obj_preds, edge_ctx, obj_ctx, _ = self.myLSTM(x, proposals)



        return  obj_pre_rep, obj_dists, obj_ctx





