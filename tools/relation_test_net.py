# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import csv
from tqdm import tqdm
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.data.datasets.evaluation.vg.predicate_eval import do_predicate_evaluation
from maskrcnn_benchmark.modeling.detector.predicated_rcnn import PredicatedRCNN
from tools.utils_predicateRCNN import Combine_TestGT,Combine_ValGT
# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode="test", is_distributed=distributed)


    # =========================================== My Model Test demo ============================================#
    mymodel = PredicatedRCNN(cfg)
    mymodel.to(cfg.MODEL.DEVICE)

    output_dir = "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/modelpth/model_0200.pth"
    mymodel.load_state_dict(torch.load(output_dir))

    # 把test GT集存为列表字典 5131933
    GTTest_list = []
    with open("/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/test_GT.csv", 'r',
              encoding='utf-8') as fp_test:
        fp_key = csv.reader(fp_test)
        for csv_key in fp_key:  # 把key取出来
            csv_reader = csv.DictReader(fp_test, fieldnames=csv_key)
            for row in tqdm(csv_reader):
                GTTest_list.append(row)

    # 得到test集的特征文件
    TestDatasetFeats = torch.load(
        "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/PretestingFeats_ctx.pt")

    # 把所有的value set image-id append list里 [1,2,3,4,6,7,8,...4998,245664,498333]
    TestFeats_image_id = [] # 26446
    for j in tqdm(range(len(TestDatasetFeats))):
        image_id = int(TestDatasetFeats[j]['image_id'])
        TestFeats_image_id.append(image_id)

    # load test union features
    # UnionFeats_list_dict_test = torch.load(
    #     "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/UnionFeats_test.pt")
    #
    # # 得到image-id列表
    # UnionFeats_image_id_test = []
    # for j in tqdm(range(len(UnionFeats_list_dict_test))):
    #     image_id = int(UnionFeats_list_dict_test[j]['image_id'])
    #     UnionFeats_image_id_test.append(image_id)

    UnionFeats_image_id_test = None
    UnionFeats_list_dict_test = None


    # 结合测试数据集和GT，和union feats
    print("#------------------------ Combine_TestGT ---------------------------#")
    triplet_image_count, test_dataloader_list_dict = Combine_TestGT(GTTest_list,
                                                                    TestFeats_image_id,
                                                                    TestDatasetFeats,
                                                                    UnionFeats_image_id_test,
                                                                    UnionFeats_list_dict_test)

    val_result = do_predicate_evaluation(cfg,
                                         mymodel,
                                         test_dataloader_list_dict,
                                         TestFeats_image_id,
                                         triplet_image_count,
                                         26446)

    # ============================= val ==============================#
    # # load 评估集特征文件
    # Value_list_dict = torch.load(
    #     "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/PrevaluingFeats_ctx_fortest.pt")
    #
    # # 把所有的value set image-id append list里 [1,2,3,4,6,7,8,...4998,245664,498333]
    # ValueFeats_image_id = []
    # for j in tqdm(range(len(Value_list_dict))):
    #     image_id = int(Value_list_dict[j]['image_id'])
    #     ValueFeats_image_id.append(image_id)
    #
    # # 把value GT数据集存为列表字典 1039344
    # GTValue_list = []
    # with open("/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/value_GT.csv", 'r',
    #           encoding='utf-8') as fp_val:
    #     fp_key = csv.reader(fp_val)
    #     for csv_key in fp_key:  # 把key取出来
    #         csv_reader = csv.DictReader(fp_val, fieldnames=csv_key)
    #         for row in tqdm(csv_reader):
    #             GTValue_list.append(row)
    #
    # # # load val union features
    # # UnionFeats_list_dict_value = torch.load(
    # #     "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/UnionFeats_value.pt")
    # #
    # # UnionFeats_image_id_value = []
    # # for j in tqdm(range(len(UnionFeats_list_dict_value))):
    # #     image_id = int(UnionFeats_list_dict_value[j]['image_id'])
    # #     UnionFeats_image_id_value.append(image_id)
    #
    # UnionFeats_list_dict_value  = None
    # UnionFeats_image_id_value = None
    # triplet_image_count, val_dataloader_list_dict = Combine_ValGT(GTValue_list,
    #                                                               ValueFeats_image_id,
    #                                                               Value_list_dict,
    #                                                               UnionFeats_image_id_value,
    #                                                               UnionFeats_list_dict_value)
    #
    # val_result = do_predicate_evaluation(cfg,
    #                                      mymodel,
    #                                      val_dataloader_list_dict,
    #                                      ValueFeats_image_id,
    #                                      triplet_image_count,
    #                                      5000)
    # ============================= val ==============================#

    # =========================================== My Model Test demo ============================================#

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()
