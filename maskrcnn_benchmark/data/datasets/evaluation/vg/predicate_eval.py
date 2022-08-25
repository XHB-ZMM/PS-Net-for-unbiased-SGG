import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce


from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps
from maskrcnn_benchmark.data.datasets.evaluation.vg.sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGNGMeanRecall, SGAccumulateRecall


def do_predicate_evaluation(cfg,model,val_dataloader_list_dict,val_image_id_list,triplet_image_count,count_image):
    '''
    :param cfg: 配置文件
    :param model: PredicatedRCNN
    :param val_dataloader_list_dict: 整个 val set 三元组的个数为1039344
    :param val_image_id_list: 5000个image_id的编号
    :param GT_count: 统计每一个image上GT谓词的数量，计算R和mR要用
    :param triplet_image_count: 5000个image，每个image里存放各自的三元组 [529,56,334,....]
    :param logger:
    :return:
    '''

    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)  # xhb：声明训练设备为gpu
    # 对原始数据集进行排序：val_dataloader_list_dict，以image为单位，按照三元组分数排序,得到一个新的val_dataloader_list_dict
    triplet_score_image = [[] for x in range(count_image)]  # 存放分数
    print("-------------start save triplet_score_image-------------")
    for i in tqdm(range(len(val_dataloader_list_dict))):
        with torch.no_grad():
            val_data_i = val_dataloader_list_dict[i][0]
            target_i = val_dataloader_list_dict[i][1]
            predictions = model(val_data_i, target_i)

            '''
              predictions = dict(
              sub_pred=sub_pred,
              obj_pred=obj_pred,
              predicate_pred=predicate_pred,
              sub_scores=sub_scores,
              obj_scores=obj_scores,
              predicate_scores=predicate_scores
            )
            '''
            # 按照置信度乘积，对每一副图像中的三元组排序，进而可以计算R@20/50/100 和 mR@20/50/100
            if  int(target_i['image_id']) in val_image_id_list:
                index = val_image_id_list.index(int(target_i['image_id']))
                if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL: # PredCls
                    triplet_score_i = predictions['predicate_scores']
                else:
                    triplet_score_i = predictions['sub_scores'] * predictions['obj_scores'] * predictions['predicate_scores']
                triplet_score_image[index].append(triplet_score_i) #把对应图像的三元组分数append对应的triplet_score_image的子列表里

    # torch.save(triplet_score_image,"/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/triplet_score_image.pt")
    # triplet_score_image = torch.load("/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/triplet_score_image.pt")

    index_all = []  # 存放排完序的索引 列表长度为5000
    for i in range(len(triplet_score_image)):
        # _, index_image = torch.sort(triplet_score_image[i]) # 对每一个image里的三元组按照分数排序，得到索引
        index_image = sorted(range(len(triplet_score_image[i])), key=lambda k: triplet_score_image[i][k], reverse=True)
        index_all.append(index_image)

    # 将原val_dataloader_list_dict按照image为单位，分成5000/26446份，存于new_val_dataloader_list_dict
    split_val_dataloader_list_dict = []  # list长度为5000/26446，但是每一个元素依然是个list
    flag = 0
    for triplet_count in triplet_image_count:
        image_i_triplet = val_dataloader_list_dict[flag:triplet_count + flag]
        split_val_dataloader_list_dict.append(image_i_triplet)
        flag = flag + triplet_count

    # 对分离后的new_val_dataloader_list_dict 按照image为单位，按照索引排序
    sort_new_val_dataloader_list_dict = []
    for i in range(len(split_val_dataloader_list_dict)):
        for j in range(len(split_val_dataloader_list_dict[i])):
            triplet = split_val_dataloader_list_dict[i][index_all[i][j]]
            sort_new_val_dataloader_list_dict.append(triplet)

    # 将原sort_new_val_dataloader_list_dict按照image为单位，分成5000/26446份，存于split_new_val_dataloader_list_dict
    split_new_val_dataloader_list_dict = []  # list长度为5000/26446，但是每一个元素依然是个list
    flag_new = 0
    for triplet_count_ in triplet_image_count:
        image_i_triplet_ = sort_new_val_dataloader_list_dict[flag_new:triplet_count_ + flag_new]
        split_new_val_dataloader_list_dict.append(image_i_triplet_)
        flag_new = flag_new + triplet_count_

    # 谓词类别输入，返回该image中该谓词类别的数量
    # 得到每张image里的非0 GT标签，分别都是哪些，然后每当我预测出谓词类时，都可以除以相应的分母
    val_dataloader_GTPredicate = []  # 存放整个val set中，所有图像中GT谓词的编号，也就是[[31,31,22,34...], [31,2,12,34...],...]
    for image_i_triplet in split_val_dataloader_list_dict:
        image_GTPredicate = []  # 存放每一张图像中非0 GT谓词的编号，也就是[31,31,22,34...]
        for triplet_i in image_i_triplet:
            rel_label = int(triplet_i[1]['predicate_label'])
            if rel_label != 0:
                image_GTPredicate.append(rel_label)
        val_dataloader_GTPredicate.append(image_GTPredicate)

    Recall_image20 = [x * 0 for x in range(count_image)]  # 声明一个5000/26446的空列表，每个位置存放每一张图像的召回率20
    Recall_image50 = [x * 0 for x in range(count_image)]  # 声明一个5000/26446的空列表，每个位置存放每一张图像的召回率50
    Recall_image100 = [x * 0 for x in range(count_image)]  # 声明一个5000/26446的空列表，每个位置存放每一张图像的召回率100
    mRecall_predicate20 = [[] for x in range(51)]  # 声明一个51的空列表，每个位置存放每一谓词类别的平均召回率20，0处不存东西
    mRecall_predicate50 = [[] for x in range(51)]  # 声明一个51的空列表，每个位置存放每一谓词类别的平均召回率50
    mRecall_predicate100 = [[] for x in range(51)]  # 声明一个51的空列表，每个位置存放每一谓词类别的平均召回率100

    # with torch.no_grad():
    print("-------------start valuing-------------")
    for i in tqdm(range(len(split_new_val_dataloader_list_dict))):

        recall_count = [x*0 for x in range(51)] # 统计一副图像中谓词的个数，第0位统计其非0谓词总数
        for idx in range(len(split_new_val_dataloader_list_dict[i])):
            predicate_label = int(split_new_val_dataloader_list_dict[i][idx][1]['predicate_label'])
            if predicate_label != 0:
                recall_count[predicate_label] += 1
                recall_count[0] += 1

        recall_hit20 = [x * 0 for x in range(51)]  # 统计一副图像中被召回谓词的个数（带索引）第0位为召回的总数
        recall_hit50 = [x * 0 for x in range(51)]
        recall_hit100 = [x * 0 for x in range(51)]

        with torch.no_grad():
            for k in [20, 50, 100]:
                length = k
                if len(split_new_val_dataloader_list_dict[i]) >= 100:
                    length = k
                elif (50 <= len(split_new_val_dataloader_list_dict[i])) < 100 and k == 100:
                    length = len(split_new_val_dataloader_list_dict[i])
                elif (20 <= len(split_new_val_dataloader_list_dict[i]) < 50) and (k == 100 or k == 50):
                    length = len(split_new_val_dataloader_list_dict[i])
                elif (len(split_new_val_dataloader_list_dict[i]) < 20) and (k == 100 or k == 50 or k == 20):
                    length = len(split_new_val_dataloader_list_dict[i])

                for j in range(len(split_new_val_dataloader_list_dict[i][:length])):  # 每一图像为单位
                    val_data_i = split_new_val_dataloader_list_dict[i][j][0]
                    target_i = split_new_val_dataloader_list_dict[i][j][1]
                    if len(split_new_val_dataloader_list_dict[i][j]) == 3:
                        union_feats = split_new_val_dataloader_list_dict[i][j][2]
                    else:
                        union_feats = None
                    predictions = model(val_data_i, target_i,union_feats)

                    '''
                      predictions = dict(
                      sub_pred=sub_pred,
                      obj_pred=obj_pred,
                      predicate_pred=predicate_pred,
                      sub_scores=sub_scores,
                      obj_scores=obj_scores,
                      predicate_scores=predicate_scores
                    )
                    '''
                    # 以image-id为基本单位，进行召回率的统计
                    sub_pred = predictions['sub_pred']
                    obj_pred = predictions['obj_pred']
                    predicate_pred = predictions['predicate_pred']
                    sub_label = int(target_i['sub_labels'])
                    obj_label = int(target_i['obj_labels'])
                    predicate_label = int(target_i['predicate_label'])

                    # 所有label全对上才算预测成功，由于我是按照编号取的，所以不用对box
                    if sub_pred == sub_label and obj_pred == obj_label and predicate_pred == predicate_label and predicate_label != 0:

                        if k == 20:
                            recall_hit20[predicate_label] += 1
                            recall_hit20[0] += 1
                        elif k == 50:
                            recall_hit50[predicate_label] += 1
                            recall_hit50[0] += 1
                        elif k == 100:
                            recall_hit100[predicate_label] += 1
                            recall_hit100[0] += 1

            for n in range(51): # 把当前图像的R和mR统计进去
                if n == 0 and recall_count[0] > 0: # 把当前image的R填进总表
                    Recall_image20[i] = float(recall_hit20[n] / recall_count[n])
                    Recall_image50[i] = float(recall_hit50[n] / recall_count[n])
                    Recall_image100[i] = float(recall_hit100[n] / recall_count[n])
                    continue
                if recall_count[n] > 0: # 把当前image的mR填进总表
                    mRecall_predicate20[n].append(float(recall_hit20[n] / recall_count[n]))
                    mRecall_predicate50[n].append(float(recall_hit50[n] / recall_count[n]))
                    mRecall_predicate100[n].append(float(recall_hit100[n] / recall_count[n]))




    print('----------------------- calculate R ------------------------\n')
    print("R@20 = {:.4f}-----R@50 = {:.4f}-----R@100 = {:.4f}\n".format(np.mean(Recall_image20)*100,
                                                                        np.mean(Recall_image50)*100,
                                                                        np.mean(Recall_image100)*100))
    print('----------------------- calculate mR ------------------------\n')
    mR20 = []
    for i,mR_predicate in enumerate(mRecall_predicate20):
        if i == 0:
            continue
        if len(mR_predicate) == 0:
            continue
        image_mR20 = np.mean(mR_predicate)
        mR20.append(image_mR20)
    mR50 = []
    for i,mR_predicate in enumerate(mRecall_predicate50):
        if i == 0:
            continue
        if len(mR_predicate) == 0:
            continue
        image_mR50 = np.mean(mR_predicate)
        mR50.append(image_mR50)
    mR100 = []
    for i,mR_predicate in enumerate(mRecall_predicate100):
        if i == 0:
            continue
        if len(mR_predicate) == 0:
            continue
        image_mR100 = np.mean(mR_predicate)
        mR100.append(image_mR100)
    print("mR@20 = {:.4f}----mR@50 = {:.4f}----mR@100 = {:.4f}\n".format((np.sum(mR20))/0.5,
                                                                             (np.sum(mR50))/0.5,
                                                                             (np.sum(mR100))/0.5))
    R100 = [] # 打印每个谓词R@100
    for i,mR_predicate in enumerate(mRecall_predicate100):
        if i == 0:
            continue
        if len(mR_predicate) == 0:
            mR_predicate = 0.0
        image_mR100 = np.mean(mR_predicate)
        R100.append(image_mR100)



    print('----------------------- Per predicate R@100 ------------------------\n')
    print("(above:{:.4f}) (across:{:.4f}) (against:{:.4f}) (along:{:.4f}) (and:{:.4f}) \n"
          "(at:{:.4f}) (attached to:{:.4f}) (behind:{:.4f}) (belonging to:{:.4f}) (between:{:.4f}) \n"
          "(carrying:{:.4f}) (covered in:{:.4f}) (covering:{:.4f}) (eating:{:.4f}) (flying in:{:.4f}) \n"
          "(for:{:.4f}) (from:{:.4f}) (growing on:{:.4f}) (hanging from:{:.4f}) (has:{:.4f}) \n"
          "(holding:{:.4f}) (in:{:.4f}) (in front of:{:.4f}) (laying on:{:.4f}) (looking at:{:.4f})\n"
          "(lying on:{:.4f}) (made of:{:.4f}) (mounted on:{:.4f}) (near:{:.4f}) (of:{:.4f})\n"
          "(on:{:.4f}) (on back of:{:.4f}) (over:{:.4f}) (painted on:{:.4f}) (parked on:{:.4f}) \n"
          "(part of:{:.4f}) (playing:{:.4f}) (riding:{:.4f}) (says:{:.4f}) (sitting on:{:.4f})  \n"
          "(standing on:{:.4f}) (to:{:.4f}) (under:{:.4f}) (using:{:.4f}) (walking in:{:.4f})\n"
          "(walking on:{:.4f}) (watching:{:.4f}) (wearing:{:.4f}) (wears:{:.4f}) (with:{:.4f})\n"
          .format(R100[0],R100[1],R100[2],R100[3],R100[4],R100[5],R100[6]
                  ,R100[7],R100[8],R100[9],R100[1],R100[11],R100[12],R100[13]
                  ,R100[14],R100[15],R100[16],R100[17],R100[18],R100[19],R100[20]
                  ,R100[21],R100[22],R100[23],R100[24],R100[25],R100[26],R100[27]
                  ,R100[28],R100[29],R100[30],R100[31],R100[32],R100[33],R100[34]
                  ,R100[35],R100[36],R100[37],R100[38],R100[39],R100[40],R100[41]
                  ,R100[42],R100[43],R100[44],R100[45],R100[46],R100[47],R100[48],R100[49]))

    for i in range(len(mRecall_predicate100)):
        if mRecall_predicate100[i] == None:
            print("{} is empty \n".format(i))


    val_result = np.sum(mR100)/0.5

    return val_result
