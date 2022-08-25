# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .predicated_rcnn import PredicatedRCNN

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}
_PREDICATE_META_ARCHITECTURES = {"PredicatedRCNN": PredicatedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    # meta_arch = _PREDICATE_META_ARCHITECTURES[cfg.MODEL.META_PREDICATE]
    return meta_arch(cfg)
