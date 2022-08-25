# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime
import csv
from tqdm import tqdm
#os.environ['CUDA_VISIBLE_DEVICES']='5'
import torch
from torch.nn.utils import clip_grad_norm_
import random
import matplotlib.pyplot as plt
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.data.datasets.evaluation.vg.predicate_eval import do_predicate_evaluation

from  maskrcnn_benchmark.modeling.detector.predicated_rcnn import PredicatedRCNN
from tools.utils_predicateRCNN import Combine_TrainGT,Combine_ValGT,randomDataloader,Combine_TestGT
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed, logger):
    debug_print(logger, 'prepare training') #xhb：训练前的准备工作：maskrcnn_benchmark INFO: #################### prepare training ####################
    model = build_detection_model(cfg)  #xhb：返回我们需要使用的模型，用的是GeneralizedRCNN
    debug_print(logger, 'end model construction') #xhb：maskrcnn_benchmark INFO: #################### end model construction ####################

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
    fix_eval_modules(eval_modules) #xhb:设置eval_modules的param.requires_grad = False，将三个模型混合起来

    # NOTE, we slow down the LR of the layers start with the names in slow_heads 下降学习率，不同的模型，下降方式不同
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",]
    else: #xhb：只要ROI_RELATION_HEAD.PREDICTOR不是IMPPredictor，slow_heads就都设置为空
        slow_heads = []

    # load pretrain layers to new layers #load_mapping就是下方这两个，如果MODEL.ATTRIBUTE_ON=True的话，就不是这两个了
    load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}
    
    if cfg.MODEL.ATTRIBUTE_ON: #xhb：如果MODEL.ATTRIBUTE_ON=True的话，检测器就变成roi_heads.attribute了
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    device = torch.device(cfg.MODEL.DEVICE) #xhb：声明训练设备为gpu
    model.to(device) #xhb：把模型放在cuda上训练

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1#xhb:得到训练时gpu的数量
    num_batch = cfg.SOLVER.IMS_PER_BATCH #xhb:得到batch_size的大小
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch)) #xhb：得到优化器SGD
    scheduler = make_lr_scheduler(cfg, optimizer, logger) #xhb：得到LR调节器，采用WarmupMultiStepLR或者WarmupReduceLROnPlateau其中一个
    debug_print(logger, 'end optimizer and shcedule') #xhb：输出日志，成功初始化完优化器和LR调节器


    # Initialize mixed-precision training 初始化混合精度训练，这个好像是apex加速，不过我好像没用到
    use_mixed_precision = cfg.DTYPE == "float16" #xhb：使用混合精度float16，还有一个float32这里不采用
    amp_opt_level = 'O1' if use_mixed_precision else 'O0' #xhb：如果使用混合精度，opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    #xhb：如果采用分布式训练，就要把调用DistributedDataParallel函数
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed') #xhb：日志输出，结束分布式配置

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    #xhb：如果output里有pth权重，那就加载当前存在的权重，如果没有那就加载预训练权重
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, 
                                       update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
    debug_print(logger, 'end load checkpointer') #xhb：输出日志：结束加载checkpointer的步骤

    # xhb：加载数据集，获得训练用的数据集 57723个
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    # xhb：加载数据集，获得评估用的数据集 5000个
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    )
    debug_print(logger, 'end dataloader') #xhb：打印日志：结束数据集的加载
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD #xhb：设置周期保存权重文件，此值为2000，可以根据需要更改
    ''' #xhb:注释掉训练前的评估#
    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed, logger)
    '''


# ##--------------------------------------------my train code demo--------------------------------------------------------------##

    # ============================= train ==============================#
    # 把train GT数据集存为列表字典 9581817
    debug_print(logger, 'Loading Predicate_GT')
    GTPredicate_list = []
    with open("/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Predicate_GT.csv", 'r',
              encoding='utf-8') as fp_train:
        fp_key = csv.reader(fp_train)
        for csv_key in fp_key:  # 把key取出来
            csv_reader = csv.DictReader(fp_train, fieldnames=csv_key)
            for row in tqdm(csv_reader):
                GTPredicate_list.append(row)

    # load 训练集数据，是一个list[dict]，list长度为图像数量=56224，里面有image-id个feats
    debug_print(logger, 'Loading PretrainingFeats56224_ctx')
    newDatasetFeats = torch.load(
        "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/PretrainingFeats56224_ctx_fortest.pt")
    debug_print(logger, 'end Loading PretrainingFeats56224_ctx')

    # 把所有的image-id append list里 [49935,49936,....] 目的是为了定位image-id的index
    DatasetFeats_image_id = []
    for j in tqdm(range(len(newDatasetFeats))):
        image_id = int(newDatasetFeats[j]['image_id'])
        DatasetFeats_image_id.append(image_id)

    # # load train union features
    # UnionFeats_list_dict_train = torch.load(
    #     "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/UnionFeats_02000.pt")
    #
    # UnionFeats_image_id_train = []
    # for j in tqdm(range(len(UnionFeats_list_dict_train))):
    #     image_id = int(UnionFeats_list_dict_train[j]['image_id'])
    #     UnionFeats_image_id_train.append(image_id)

    UnionFeats_list_dict_train = None
    UnionFeats_image_id_train = None
    # ============================= train ==============================#

    # 按照原始分布采样GT
    # m = 2000
    # origin_GTPredicate_batch_All = []  # 存放5000个batch
    # origin_predicate_distribution = [] # 存放每个batch里的谓词分布
    # for n in tqdm(range(5000)): # 采样5000次，得到5000个batch
    #     predicate_batch = random.sample(GTPredicate_list,m)
    #     origin_GTPredicate_batch_All.append(predicate_batch) #保存batch本身内容
    #     batch_predicate_count = [x*0 for x in range(51)] # 统计每一个batch里面，谓词分布情况
    #     for i in  range(len(predicate_batch)):
    #         id = int(predicate_batch[i]['predicate_label'])
    #         batch_predicate_count[id] = batch_predicate_count[id] + 1
    #     origin_predicate_distribution.append(batch_predicate_count)

    # # 统计GT 各个谓词数量，进而得到采样率
    # predlicate_to_count = [x*0 for x in range(51)]
    # for i in tqdm(range(len(GTPredicate_list))):
    #     id = int(GTPredicate_list[i]['predicate_label'])
    #     predlicate_to_count[id] = predlicate_to_count[id] + 1

    predlicate_probability =[0.0000707383, # background sample-ratio = 0.000000107383时，background将与其他保持实例平衡
    0.000224972,0.006993007,0.006289308,0.002923977,0.002469136,0.000831947,0.000958773,0.000117357,0.001831502,0.002873563,
    0.00101626,0.00310559,0.002680965,0.002824859,0.25,0.001517451,0.007092199,0.007462687,0.001912046,0.0000206637,0.00015006,
    0.0000655781,0.000380373,0.002298851,0.002079002,0.004716981,0.012658228,0.005524862,0.0000814001,0.000040657,0.00001259,
    0.004065041,0.001481481,0.008333333,0.002123142,0.002915452,0.02,0.000445831,0.045454545,0.000312402,0.000632111,0.004081633,
    0.000335233,0.003278689,0.005291005,0.001113586,0.003322259,0.000031001,0.000294291,0.00011711]

    # # 多数类，稍微多采样一点，has near of on in,少数类，少一点
    # predlicate_probability = [0.0000997383,  # background sample-ratio = 0.000000107383时，background将与其他保持实例平衡
    #                           0.000384972, 0.006993007, 0.006289308, 0.002923977, 0.002469136,
    #                           0.000831947, 0.000958773, 0.000357357, 0.001831502, 0.002873563,
    #                           0.00101626, 0.00310559, 0.002680965, 0.002824859, 0.25,
    #                           0.001517451, 0.007092199, 0.007462687, 0.001912046, 0.0001006637,
    #                           0.00023006, 0.000145578, 0.000380373, 0.002298851, 0.002079002,
    #                           0.004716981, 0.012658228, 0.005524862, 0.000181400, 0.000130657,
    #                           0.00014259, 0.004065041, 0.001481481, 0.008333333, 0.002123142,
    #                           0.002915452, 0.02, 0.000525831, 0.045454545, 0.000442402,
    #                           0.000632111, 0.004081633, 0.000425233, 0.003278689, 0.005291005,
    #                           0.001113586, 0.003322259, 0.000121001, 0.000294291, 0.00019711]

    # 不要多数类，或者让多数类更少，比少数类还少
    # predlicate_probability = [0.0000897383,  # background sample-ratio = 0.000000107383时，background将与其他保持实例平衡
    #                           0.000384972, 0.006993007, 0.006289308, 0.002923977, 0.002469136,
    #                           0.000831947, 0.000958773, 0.000357357, 0.001831502, 0.002873563,
    #                           0.00101626, 0.00310559, 0.002680965, 0.002824859, 0.25,
    #                           0.001517451, 0.007092199, 0.007462687, 0.001912046, 0.000001006637,
    #                           0.00023006, 0.00000145578, 0.000380373, 0.002298851, 0.002079002,
    #                           0.004716981, 0.012658228, 0.005524862, 0.00000181400, 0.00000130657,
    #                           0.0000014259, 0.004065041, 0.001481481, 0.008333333, 0.002123142,
    #                           0.002915452, 0.02, 0.000525831, 0.045454545, 0.000442402,
    #                           0.000632111, 0.004081633, 0.000425233, 0.003278689, 0.005291005,
    #                           0.001113586, 0.003322259, 0.0000121001, 0.000294291, 0.000019711]

    # 为每一个GT附上离散概率，服务于random.choices函数
    GTPredicate_probability = [x * 0 for x in range(len(GTPredicate_list))]  # 用于存放9581817个概率
    for i in tqdm(range(len(GTPredicate_list))):
        id = int(GTPredicate_list[i]['predicate_label'])
        GTPredicate_probability[i] = predlicate_probability[id]

    # # 按照指定采样频率采样，并获得采样后的分布predicate_distribution，用于验证是不是想要的分布
    # batch_size = 2000  # 目前设定每个batch里有2000组三元组数据
    # epoch_iteration = 500
    # GTPredicate_batch_All = []
    # predicate_distribution = []  # 存放每个batch里的谓词分布
    # for n in tqdm(range(epoch_iteration)):  # 采样epoch_iteration次，得到epoch_iteration个batch
    #     batch_sampler = random.choices(GTPredicate_list, weights=GTPredicate_probability, k=batch_size)
    #     GTPredicate_batch_All.append(batch_sampler)  # 保存batch本身内容
    #     batch_predicate_count = [x * 0 for x in range(51)]  # 统计每一个batch里面，谓词分布情况
    #     for i in range(len(batch_sampler)):
    #         id = int(batch_sampler[i]['predicate_label'])
    #         batch_predicate_count[id] = batch_predicate_count[id] + 1
    #     predicate_distribution.append(batch_predicate_count)
    #
    # # 查看平均每个谓词召回多少
    # sum = [x*0 for x in range(51)]
    # for i in range(len(predicate_distribution)):
    #     for j in  range(len(predicate_distribution[i])):
    #         sum[j] += predicate_distribution[i][j]
    # avg_predicate_distribution =[]
    # for i in sum:
    #     j = i/epoch_iteration
    #     avg_predicate_distribution.append(j)


    plt.bar(range(len(avg_predicate_distribution[1:])), avg_predicate_distribution[1:])
    plt.savefig("/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/bar_picture/avg_predicate_distribution.png")
    plt.show()

    # ============================= val ==============================#
    # load 评估集特征文件
    Value_list_dict = torch.load(
        "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/PrevaluingFeats_ctx_fortest.pt")

    # 把所有的value set image-id append list里 [1,2,3,4,6,7,8,...4998,245664,498333]
    ValueFeats_image_id = []
    for j in tqdm(range(len(Value_list_dict))):
        image_id = int(Value_list_dict[j]['image_id'])
        ValueFeats_image_id.append(image_id)

    # 把value GT数据集存为列表字典 1039344
    debug_print(logger, 'Loading value_GT')
    GTValue_list = []
    with open("/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/value_GT.csv", 'r',
              encoding='utf-8') as fp_val:
        fp_key = csv.reader(fp_val)
        for csv_key in fp_key:  # 把key取出来
            csv_reader = csv.DictReader(fp_val, fieldnames=csv_key)
            for row in tqdm(csv_reader):
                GTValue_list.append(row)

    # load val union features
    # debug_print(logger, 'Loading value union features')
    # UnionFeats_list_dict_value = torch.load(
    #     "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/UnionFeats_value.pt")
    #
    # UnionFeats_image_id_value = []
    # for j in tqdm(range(len(UnionFeats_list_dict_value))):
    #     image_id = int(UnionFeats_list_dict_value[j]['image_id'])
    #     UnionFeats_image_id_value.append(image_id)

    UnionFeats_list_dict_value = None
    UnionFeats_image_id_value = None
    # 将GT与val数据一一对应（绑在一起） [[val_data1,GT1,union-feats],[val_data2,GT2],....]
    debug_print(logger, 'Remake Val Dataloader')
    triplet_image_count_val, val_dataloader_list_dict = Combine_ValGT(GTValue_list,
                                                                      ValueFeats_image_id,
                                                                      Value_list_dict,
                                                                      UnionFeats_image_id_value,
                                                                      UnionFeats_list_dict_value)
    # ============================= val ==============================#

    # debug val代码用的
    # mymodel = PredicatedRCNN(cfg)
    # mymodel.to(device)
    # val_result = do_predicate_evaluation(cfg, mymodel, val_dataloader_list_dict, ValueFeats_image_id, triplet_image_count,5000)

    # ============================= test ==============================#
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
    TestFeats_image_id = []  # 26446
    for j in tqdm(range(len(TestDatasetFeats))):
        image_id = int(TestDatasetFeats[j]['image_id'])
        TestFeats_image_id.append(image_id)

    # # load test union features
    # UnionFeats_list_dict_test = torch.load(
    #     "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/UnionFeats_test.pt")
    #
    # # 得到image-id列表
    # UnionFeats_image_id_test = []
    # for j in tqdm(range(len(UnionFeats_list_dict_test))):
    #     image_id = int(UnionFeats_list_dict_test[j]['image_id'])
    #     UnionFeats_image_id_test.append(image_id)

    # No use union features : comment
    UnionFeats_image_id_test = None
    UnionFeats_list_dict_test = None

    # 结合测试数据集和GT，和union feats
    print("#------------------------ Combine_TestGT ---------------------------#")
    triplet_image_count_test, test_dataloader_list_dict = Combine_TestGT(GTTest_list,
                                                                         TestFeats_image_id,
                                                                         TestDatasetFeats,
                                                                         UnionFeats_image_id_test,
                                                                         UnionFeats_list_dict_test)

    # ============================= test ==============================#

    # training stating
    logger.info("Start training")  # xhb：输出日志：开始训练了
    meters = MetricLogger(delimiter="  ")
    end = time.time()
    mymodel = PredicatedRCNN(cfg)
    optimizer_Adam = torch.optim.Adam(mymodel.parameters(), lr=0.01)  # SGD也可备选，lr需要尝试初始学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_Adam,gamma=0.9,last_epoch=-1)
    # scheduler = make_lr_scheduler(cfg, optimizer_Adam, logger)
    mymodel.to(device)
    for epoch in range(5):
        debug_print(logger, 'Remake Dataloader (Thanks hch-ing + hsm-ed)')
        batch_size = 2000
        epoch_iteration = 1000
        # random GTPredicate_list
        GTPredicate_probability, GTPredicate_list = randomDataloader(GTPredicate_list, predlicate_probability)
        train_dataloader_list_dict = Combine_TrainGT(batch_size,
                                                     epoch_iteration,
                                                     GTPredicate_list,
                                                     GTPredicate_probability,
                                                     DatasetFeats_image_id,
                                                     newDatasetFeats,
                                                     UnionFeats_list_dict_train,
                                                     UnionFeats_image_id_train)
        max_iter = len(train_dataloader_list_dict)  # xhb：max_iter：4000
        for iteration, train_gt_batch in enumerate(train_dataloader_list_dict):
            data_time = time.time() - end
            iteration = iteration + 1
            real_iteration = iteration + epoch * epoch_iteration
            arguments["iteration"] = real_iteration
            rel_loss = torch.tensor(0)  # 初始化而已
            # sub_loss = torch.tensor(0)
            # obj_loss = torch.tensor(0)
            # batch_loss  = dict(loss_rel=rel_loss, loss_refine_sub=sub_loss, loss_refine_obj=obj_loss)
            batch_loss = dict(loss_rel=rel_loss)
            for i in range(len(train_gt_batch)):
                mymodel.train()
                train_data = train_gt_batch[i][0]
                target = train_gt_batch[i][1]
                if len(train_gt_batch[i]) == 3:
                    union_feats = train_gt_batch[i][2]
                else:
                    union_feats = None
                output_loss = mymodel(train_data, target, union_feats)
                batch_loss['loss_rel'] = output_loss['loss_rel'] + batch_loss['loss_rel']
                # batch_loss['loss_refine_sub'] = output_loss['loss_refine_sub'] + batch_loss['loss_refine_sub']
                # batch_loss['loss_refine_obj'] = output_loss['loss_refine_obj'] + batch_loss['loss_refine_obj']

            losses = sum(loss / len(train_gt_batch) for loss in batch_loss.values())

            meters.update(loss=losses, **batch_loss)

            optimizer_Adam.zero_grad()  # xhb：优化器清除梯度，准备损失梯度回传了

            losses.backward()

            optimizer_Adam.step()  # xhb：优化器开始工作

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if real_iteration % 50 == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=real_iteration,
                        meters=str(meters),
                        lr=optimizer_Adam.param_groups[-1]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

            # xhb：如果迭代次数正好到了要保存模型的时候，就save模型
            if real_iteration % 1000 == 0:
                print("save model to /data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/modelpth/model_{:04d}.pth".format(
                        real_iteration))
                torch.save(mymodel.state_dict(),
                           "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/modelpth/model_{:04d}.pth".format(
                               real_iteration))

            if cfg.SOLVER.TO_VAL and real_iteration % 1000 == 0:  # cfg.SOLVER.VAL_PERIOD和VAL_PERIOD 我都改成了500
                logger.info("Start validating")
                val_result_val = do_predicate_evaluation(cfg,
                                                     mymodel,
                                                     val_dataloader_list_dict,
                                                     ValueFeats_image_id,
                                                     triplet_image_count_val,
                                                     5000)
                logger.info("Validation Result: %.4f" % val_result_val)
                val_result_test = do_predicate_evaluation(cfg,
                                                     mymodel,
                                                     test_dataloader_list_dict,
                                                     TestFeats_image_id,
                                                     triplet_image_count_test,
                                                     26446)
                logger.info("Testing Result: %.4f" % val_result_test)
                for p in optimizer_Adam.param_groups:
                    p["lr"] *= 0.5

    # ##--------------------------------------------my train code demo --------------------------------------------------------------##


    logger.info("Start training") #xhb：输出日志：开始训练了
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader) #xhb：max_iter：50000
    start_iter = arguments["iteration"] #xhb：start_iter不定，如果是接着训练（output里有pth文件），那start_iter就是接着的数，如果是初次训练，start_iter就是0
    start_training_time = time.time() # 1641821079.0426774
    end = time.time()                 # 1641821079.0426843

    print_first_grad = True
    '''
    xhb: for循环就是训练的循环语句
        images    : 此刻就是图像，单纯的图像
        targets   : 里面存的是box GT框，还有目标的label
        iteration : 当前迭代次数
    '''
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter): #xhb：image已经是tensor了，target信息里有box的数量
        if any(len(target) < 1 for target in targets):#xhb：如果任何一个target不存在，就输出下面的错误日志
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.train()
        fix_eval_modules(eval_modules) #xhb：eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)

        images = images.to(device) #xhb：将输入图像放在cuda上
        targets = [target.to(device) for target in targets] #xhb：将GT也放在cuda上

        loss_dict = model(images, targets) #xhb：images是输入图像、targets是GT标签

        losses = sum(loss for loss in loss_dict.values()) #xhb：将loss值加起来

        # reduce losses over all GPUs for logging purposes
        '''
        xhb:
        :param loss_dict_reduced是一个字典，里面有5个key
                'loss_rel'         : tensor(0.0839,
                'loss_refine_obj'  : tensor(1.0486, 
                'auxiliary_ctx'    : tensor(0.0748, 
                'auxiliary_vis'    : tensor(0.0938, 
                'auxiliary_frq'    : tensor(0.1404, 
        '''
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values()) #xhb：losses_reduced是一个数值，把上面字典里的5个损失值加在一起
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad() #xhb：优化器清除梯度，准备损失梯度回传了
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # xhb：如果不采用混合精度，损失就不backward了吗？
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        
        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
        print_first_grad = False
        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

        optimizer.step() #xhb：优化器开始工作

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        #xhb：每迭代200次，输出一些东西，包括时间、迭代次数，学习率等等
        if iteration % 200 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        #xhb：如果迭代次数正好到了要保存模型的时候，就save模型
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            # torch.save(model.roi_heads.relation.predictor.context_layer.state_dict(),"/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/modulepth/LSTMContext_state_dict_{:05d}.pth".format(iteration))
            # torch.save(model.PretrainingFeats_dict_list,"/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/PretrainingFeats_ctx_fortest.pt")
            # torch.save(model.roi_heads.relation.union_feats_list_dict,
            #            "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/UnionFeats_{:05d}.pt".format(iteration))
            # torch.save(model.union_feats_list_dict,
            #        "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/UnionFeats_{:05d}.pt".format(
            #            iteration))

        # xhb：如果迭代次数到达了最大迭代次数，就保存模型为model_final
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        val_result = None # used for scheduler updating xhb：评估模型，主要是调用run_val函数
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start validating")
            val_result = run_val(cfg, model, val_data_loaders, distributed, logger)
            # torch.save(model.union_feats_list_dict,
            #        "/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/UnionFeats_value.pt")
            # torch.save(model.ValueFeats_dict_list,"/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/datasets/vg/Feats/PrevaluingFeats_ctx_fortest.pt")
            logger.info("Validation Result: %.4f" % val_result)
 
        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        # xhb：自动调整学习率，这里采用WarmupReduceLROnPlateau方法
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(val_result, epoch=iteration) #xhb：根据评估结果调整学习率
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                # torch.save(model.roi_heads.relation.predictor.context_layer.state_dict(),"/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/modulepth/LSTMContext_state_dict_final_fortest.pth")
                # torch.save(model.roi_heads.relation.predictor.context_layer.state_dict(),"/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_new/modulepth/LSTMContext.pth")
                break
        else:
            scheduler.step()

    total_training_time = time.time() - start_training_time #xhb：此时一次迭代训练就已经结束了
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    return model

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def run_val(cfg, model, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
                            cfg,
                            model,
                            val_data_loader,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=None,
                            logger=logger,
                        )
        synchronize()
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch.tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result>=0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result
    torch.cuda.empty_cache()
    return val_result

def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
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
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
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
            logger=logger,
        )
        synchronize()


def main():
    #xhb:解决通过命令行传入参数，从而避免了每次运行时都要改动代码的缺点。argparse.ArgumentParser就是干这个事的
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")#xhb:description告诉这段程序是干什么的，相当于注释，不会打印出来
    parser.add_argument(
        "--config-file", #xhb:config前面有两条杠，说明在命令行中，必须先把--config-file写出来，后面才可以跟参数，直接写参数不行
        default="",
        metavar="FILE",
        help="path to config file",#xhb:也相当于是注释，不会影响程序，表明这个参数是做什么用的
        type=str,#xhb:指定了输入参数的类型
    )
    '''
    xhb:命令：--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" 
    '''
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args() #xhb:使得修改后的参数生效，将设置的所有add_argument返回到args中，那么parser中增加的属性内容都会在args实例中，使用即可
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1 #xhb:得到GPU的数量
    args.distributed = num_gpus > 1

    if args.distributed: #xhb:如果训练时使用的GPU数量大于1，那就进行分布式训练
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    #xhb:cfg 就是 _C ，存放参数的实例
    cfg.merge_from_file(args.config_file)#xhb:merge_from_file是用来加载yaml文件的
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR #xhb：输出文件夹的目录
    if output_dir:
        mkdir(output_dir) #xhb：创建一个一级目录output_dir，也就是存放输出文件的地方

    #xhb：打印日志 maskrcnn_benchmark INFO: 这种的
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus)) #xhb：maskrcnn_benchmark INFO: Using 2 GPUs
    logger.info(args) #xhb：把配置的参数以日志的形式输出

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info()) #xhb：显示一些系统信息和cuda信息

    # maskrcnn_benchmark INFO: Loaded configuration file configs/e2e_relation_X_101_32_8_FPN_1x.yaml
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str) #xhb：把yaml文件打印出来了

    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml') #xhb：将输出路径拼接起来，

    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path) #xhb：将本次训练用到的参数，保存在output/config.yml里面

    #xhb：配置完参数并保存完参数，训练开始
    model = train(cfg, args.local_rank, args.distributed, logger)

    #xhb：如果没有跳过测试，那就测试
    if not args.skip_test:
        run_test(cfg, model, args.distributed, logger)


if __name__ == "__main__":
    main()
