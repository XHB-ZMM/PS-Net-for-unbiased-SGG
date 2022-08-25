
import csv
from tqdm import tqdm
import torch
from torch import nn
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation.vg.predicate_eval import do_predicate_evaluation
from maskrcnn_benchmark.modeling.detector.predicated_rcnn import PredicatedRCNN
from tools.utils_predicateRCNN import Combine_TestGT
from maskrcnn_benchmark.data import make_data_loader

def main():


    # xhb：加载数据集，获得训练用的数据集 57723个

    model = PredicatedRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/modelpth/model_0400.pth"
    model.load_state_dict(torch.load(output_dir))

    # 把test GT集存为列表字典 5132015
    GTTest_list = []
    with open("/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/test_GT.csv", 'r',encoding='utf-8') as fp_test:
        fp_key = csv.reader(fp_test)
        for csv_key in fp_key:  # 把key取出来
            csv_reader = csv.DictReader(fp_test, fieldnames=csv_key)
            for row in tqdm(csv_reader):
                GTTest_list.append(row)

    # 得到test集的特征文件
    TestDatasetFeats = torch.load("/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/PretestingFeats_ctx.pt")


    # 把所有的value set image-id append list里 [1,2,3,4,6,7,8,...4998,245664,498333]
    TestFeats_image_id = []
    for j in tqdm(range(len(TestDatasetFeats))):
        image_id = int(TestDatasetFeats[j]['image_id'])
        TestFeats_image_id.append(image_id)

    # load test union features
    UnionFeats_list_dict_test = torch.load(
        "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/UnionFeats_test.pt")

    # 得到image-id列表
    UnionFeats_image_id_test = []
    for j in tqdm(range(len(UnionFeats_list_dict_test))):
        image_id = int(UnionFeats_list_dict_test[j]['image_id'])
        UnionFeats_image_id_test.append(image_id)

    # 结合测试数据集和GT，和union feats
    triplet_image_count,test_dataloader_list_dict = Combine_TestGT(GTTest_list,
                                                                  TestFeats_image_id,
                                                                  TestDatasetFeats,
                                                                  UnionFeats_image_id_test,
                                                                  UnionFeats_list_dict_test)


    val_result = do_predicate_evaluation(cfg,
                                         model,
                                         test_dataloader_list_dict,
                                         TestFeats_image_id,
                                         triplet_image_count,
                                         26446)

if __name__ == "__main__":
    main()