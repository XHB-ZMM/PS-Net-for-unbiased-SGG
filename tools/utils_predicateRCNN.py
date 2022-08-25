import torch

import random
from tqdm import tqdm


def Combine_TrainGT(batch_size,epoch_iteration,GTPredicate_list,GTPredicate_probability,DatasetFeats_image_id,newDatasetFeats,UnionFeats_list_dict,UnionFeats_image_id):
    # batch_size = 2000
    # epoch_iteration = 12000

    train_dataloader_list_dict = []  # 存放4000个batch，batch里有2000个字典
    for n in tqdm(range(epoch_iteration)):  # 采样epoch_iteration次，得到epoch_iteration个batch
        batch_gt = random.choices(GTPredicate_list, weights=GTPredicate_probability, k=batch_size) # 按照GTPredicate_probability分布采样
        # batch_gt = random.sample(GTPredicate_list, batch_size) # 原始分布采样
        train_set_batch_list_dict = []  # 存放2000个字典，也就是1个batch
        for i in range(batch_size):
            batch_i_iamge_id = int(batch_gt[i]['image_id'])  # 得到第n个batch里第i个gt三元组的image-id
            batch_i_sub_id = int(batch_gt[i]['sub_id'])  # 得到第n个batch里第i个gt三元组的sub_id
            batch_i_obj_id = int(batch_gt[i]['obj_id'])  # 得到第n个batch里第i个gt三元组的obj_id
            if batch_i_iamge_id in DatasetFeats_image_id:
                index = DatasetFeats_image_id.index(batch_i_iamge_id)
                batch_i_sub_feats = newDatasetFeats[index]['feats'][batch_i_sub_id, :]  # 得到相应sub的特征（4424维度的）
                batch_i_obj_feats = newDatasetFeats[index]['feats'][batch_i_obj_id, :]  # 得到相应obj的特征（4424维度的）
                batch_i_sub_dist = newDatasetFeats[index]['obj_dist'][batch_i_sub_id, :]  # 得到相应sub的dist（151维度的）
                batch_i_obj_dist = newDatasetFeats[index]['obj_dist'][batch_i_obj_id, :]  # 得到相应obj的dist（151维度的）
                batch_i_sub_ctx = newDatasetFeats[index]['obj_ctx'][batch_i_sub_id, :]  # 得到相应sub的context（512维度的）
                batch_i_obj_ctx = newDatasetFeats[index]['obj_ctx'][batch_i_obj_id, :]  # 得到相应obj的context（512维度的）

                train_dataset_dict = {
                    "image_id": batch_i_iamge_id,
                    "sub_id": batch_i_sub_id,
                    "obj_id": batch_i_obj_id,
                    "sub_feats": batch_i_sub_feats,
                    "obj_feats": batch_i_obj_feats,
                    "sub_dist": batch_i_sub_dist,
                    "obj_dist": batch_i_obj_dist,
                    "sub_ctx": batch_i_sub_ctx,
                    "obj_ctx": batch_i_obj_ctx,
                }

                # union_feats = None
                # if batch_i_iamge_id in UnionFeats_image_id:
                #     index_u = UnionFeats_image_id.index(batch_i_iamge_id)
                #     if batch_i_sub_id == int(UnionFeats_list_dict[index_u]["sub_id"]) and batch_i_obj_id == int(
                #             UnionFeats_list_dict[index_u]["obj_id"]):
                #         union_feats = UnionFeats_list_dict[index_u]["union_feats"]
                #
                # if union_feats is not None:
                #     train_unit = (train_dataset_dict, batch_gt[i], union_feats)
                # else:
                #     train_unit = (train_dataset_dict, batch_gt[i])
                train_unit = (train_dataset_dict, batch_gt[i])
                train_set_batch_list_dict.append(train_unit)  # 把对应于gt的训练数据存于字典，最后append到一个2000长度的列表train_set_list_dict里
        train_dataloader_list_dict.append(train_set_batch_list_dict)

    return  train_dataloader_list_dict


def Combine_ValGT(GTValue_list,ValueFeats_image_id,Value_list_dict,UnionFeats_image_id,UnionFeats_list_dict):
    triplet_image_count = [x * 0 for x in range(5000)]  # 存放每个image中三元组的个数[529,200,44,...,361,...]，作为实际参数传递给评估函数
    # debug_print(logger, 'Remake Val Dataloader')
    val_dataloader_list_dict = []  # 存放整个value set所有三元组的训练集数据
    for i in tqdm(range(len(GTValue_list))):
        val_i_iamge_id = int(GTValue_list[i]['image_id'])  # 得到第i个gt三元组的image-id
        val_i_sub_id = int(GTValue_list[i]['sub_id'])  # 得到第i个gt三元组的sub_id
        val_i_obj_id = int(GTValue_list[i]['obj_id'])  # 得到第i个gt三元组的obj_id
        if val_i_iamge_id in ValueFeats_image_id:  # 在训练集中找到与之匹配的image
            index = ValueFeats_image_id.index(val_i_iamge_id)  # 取出image-id的位置
            batch_i_sub_feats = Value_list_dict[index]['feats'][val_i_sub_id, :]  # 得到相应sub的特征（4424维度的）
            batch_i_obj_feats = Value_list_dict[index]['feats'][val_i_obj_id, :]  # 得到相应obj的特征（4424维度的）
            batch_i_sub_dist = Value_list_dict[index]['obj_dist'][val_i_sub_id, :]  # 得到相应sub的dist（151维度的）
            batch_i_obj_dist = Value_list_dict[index]['obj_dist'][val_i_obj_id, :]  # 得到相应obj的dist（151维度的）
            batch_i_sub_ctx = Value_list_dict[index]['obj_ctx'][val_i_sub_id, :]  # 得到相应sub的context（512维度的）
            batch_i_obj_ctx = Value_list_dict[index]['obj_ctx'][val_i_obj_id, :]  # 得到相应obj的context（512维度的）

            triplet_image_count[index] = triplet_image_count[index] + 1
            val_dataset_dict = {  # 创建一个字典，存与gt对应的训练数据
                "image_id": val_i_iamge_id,
                "sub_id": val_i_sub_id,
                "obj_id": val_i_obj_id,
                "sub_feats": batch_i_sub_feats,
                "obj_feats": batch_i_obj_feats,
                "sub_dist": batch_i_sub_dist,
                "obj_dist": batch_i_obj_dist,
                "sub_ctx": batch_i_sub_ctx,
                "obj_ctx": batch_i_obj_ctx,
            }

            # union_feats = None
            # if val_i_iamge_id in UnionFeats_image_id:
            #     index_u = UnionFeats_image_id.index(val_i_iamge_id)
            #     if val_i_sub_id == int(UnionFeats_list_dict[index_u]["sub_id"]) and val_i_obj_id == int(
            #             UnionFeats_list_dict[index_u]["obj_id"]):
            #         union_feats = UnionFeats_list_dict[index_u]["union_feats"]
            #
            # if union_feats is not None:
            #     val_unit = (val_dataset_dict, GTValue_list[i], union_feats)
            # else:
            #     val_unit = (val_dataset_dict, GTValue_list[i])

            val_unit = (val_dataset_dict, GTValue_list[i])

            val_dataloader_list_dict.append(val_unit)  # 把对应于gt的训练数据存于字典，最后append到一个2000长度的列表train_set_list_dict里

    return triplet_image_count,val_dataloader_list_dict


def Combine_TestGT(GTTest_list,TestFeats_image_id,TestDatasetFeats,UnionFeats_image_id_test,UnionFeats_list_dict_test):
    triplet_image_count = [x * 0 for x in range(26446)]  # 存放每个image中三元组的个数[529,200,44,...,361,...]，作为实际参数传递给评估函数
    # debug_print(logger, 'Remake Val Dataloader')
    test_dataloader_list_dict = []  # 存放整个value set所有三元组的训练集数据
    for i in tqdm(range(len(GTTest_list))):
        test_i_iamge_id = int(GTTest_list[i]['image_id'])  # 得到第i个gt三元组的image-id
        test_i_sub_id = int(GTTest_list[i]['sub_id'])  # 得到第i个gt三元组的sub_id
        test_i_obj_id = int(GTTest_list[i]['obj_id'])  # 得到第i个gt三元组的obj_id
        if test_i_iamge_id in TestFeats_image_id:  # 在训练集中找到与之匹配的image
            index = TestFeats_image_id.index(test_i_iamge_id)  # 取出image-id的位置
            batch_i_sub_feats = TestDatasetFeats[index]['feats'][test_i_sub_id, :]  # 得到相应sub的特征（4424维度的）
            batch_i_obj_feats = TestDatasetFeats[index]['feats'][test_i_obj_id, :]  # 得到相应obj的特征（4424维度的）
            batch_i_sub_dist = TestDatasetFeats[index]['obj_dist'][test_i_sub_id, :]  # 得到相应sub的dist（151维度的）
            batch_i_obj_dist = TestDatasetFeats[index]['obj_dist'][test_i_obj_id, :]  # 得到相应obj的dist（151维度的）
            batch_i_sub_ctx = TestDatasetFeats[index]['obj_ctx'][test_i_sub_id, :]  # 得到相应sub的context（512维度的）
            batch_i_obj_ctx = TestDatasetFeats[index]['obj_ctx'][test_i_obj_id, :]  # 得到相应obj的context（512维度的）

            triplet_image_count[index] = triplet_image_count[index] + 1
            test_dataset_dict = {  # 创建一个字典，存与gt对应的训练数据
                "image_id": test_i_iamge_id,
                "sub_id": test_i_sub_id,
                "obj_id": test_i_obj_id,
                "sub_feats": batch_i_sub_feats,
                "obj_feats": batch_i_obj_feats,
                "sub_dist": batch_i_sub_dist,
                "obj_dist": batch_i_obj_dist,
                "sub_ctx": batch_i_sub_ctx,
                "obj_ctx": batch_i_obj_ctx,
            }

            # union_feats = None
            # if test_i_iamge_id in UnionFeats_image_id_test:
            #     index_u = UnionFeats_image_id_test.index(test_i_iamge_id)
            #     if test_i_sub_id == int(UnionFeats_list_dict_test[index_u]["sub_id"]) and test_i_obj_id == int(
            #             UnionFeats_list_dict_test[index_u]["obj_id"]):
            #         union_feats = UnionFeats_list_dict_test[index_u]["union_feats"]
            #
            # if union_feats is not None:
            #     test_unit = (test_dataset_dict, GTTest_list[i], union_feats)
            # else:
            #     test_unit = (test_dataset_dict, GTTest_list[i])

            test_unit = (test_dataset_dict, GTTest_list[i])
            test_dataloader_list_dict.append(test_unit)  # 把对应于gt的训练数据存于字典，最后append到一个2000长度的列表train_set_list_dict里

    return triplet_image_count,test_dataloader_list_dict


def randomDataloader(GTPredicate_list,predlicate_probability):
    random.shuffle(GTPredicate_list)
    GTPredicate_probability = [x*0 for x in range(len(GTPredicate_list))] # 用于存放9581817个概率
    for i in tqdm(range(len(GTPredicate_list))):
        id = int(GTPredicate_list[i]['predicate_label'])
        GTPredicate_probability[i] = predlicate_probability[id]

    return GTPredicate_probability,GTPredicate_list