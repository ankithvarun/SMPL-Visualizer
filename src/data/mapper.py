# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import random
import logging
from typing import Any, Dict, List, Tuple
import cloudpickle
import torch
from torch.utils.data import Dataset, IterableDataset

from ..utils.transform import apply_augmentations, transform_instance_annotations, annotations_to_instances, ResizeShortestEdge, RandomFlip, RandomRotation, DensePoseTransformData
from ..utils.image import read_image, check_image_size
from ..utils.structures import DensePoseList, DensePoseDataRelative

# from detectron2.data import MetadataCatalog
# from detectron2.data import detection_utils as utils
# from detectron2.data import transforms as T
# from detectron2.layers import ROIAlign
# from detectron2.structures import BoxMode
# from detectron2.utils.file_io import PathManager

# from densepose.structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData

MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
MAX_SIZE_TRAIN = 1333
MIN_SIZE_TEST = 800
MAX_SIZE_TEST = 1333
RANDOM_FLIP = "horizontal"
ROTATION_ANGLES = [0]

def build_augmentation(is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = MIN_SIZE_TRAIN
        max_size = MAX_SIZE_TRAIN
        sample_style = "choice"
    else:
        min_size = MIN_SIZE_TEST
        max_size = MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train:
        if RANDOM_FLIP != "none":
            augmentation.append(
                RandomFlip(
                    horizontal=RANDOM_FLIP == "horizontal",
                    vertical=RANDOM_FLIP == "vertical",
                )
            )
        random_rotation = RandomRotation(
            ROTATION_ANGLES, expand=False, sample_style="choice"
        )
        augmentation.append(random_rotation)
        print("DensePose-specific augmentation used in training: " + str(random_rotation))
    return augmentation

class PicklableWrapper(object):
    """
    Wrap an object to make it more picklable, note that it uses
    heavy weight serialization libraries that are slower than pickle.
    It's best to use it only on closures (which are usually not picklable).

    This is a simplified version of
    https://github.com/joblib/joblib/blob/master/joblib/externals/loky/cloudpickle_wrapper.py
    """

    def __init__(self, obj):
        while isinstance(obj, PicklableWrapper):
            # Wrapping an object twice is no-op
            obj = obj._obj
        self._obj = obj

    def __reduce__(self):
        s = cloudpickle.dumps(self._obj)
        return cloudpickle.loads, (s,)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getattr__(self, attr):
        # Ensure that the wrapped object can be used seamlessly as the previous object.
        if attr not in ["_obj"]:
            return getattr(self._obj, attr)
        return getattr(self, attr)

class _MapIterableDataset(IterableDataset):
    """
    Map a function over elements in an IterableDataset.

    Similar to pytorch's MapIterDataPipe, but support filtering when map_func
    returns None.

    This class is not public-facing. Will be called by `MapDataset`.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        for x in map(self._map_func, self._dataset):
            if x is not None:
                yield x

class MapDataset(Dataset):
    """
    Map a function over the elements in a dataset.
    """

    def __init__(self, dataset, map_func):
        """
        Args:
            dataset: a dataset where map function is applied. Can be either
                map-style or iterable dataset. When given an iterable dataset,
                the returned object will also be an iterable dataset.
            map_func: a callable which maps the element in dataset. map_func can
                return None to skip the data (e.g. in case of errors).
                How None is handled depends on the style of `dataset`.
                If `dataset` is map-style, it randomly tries other elements.
                If `dataset` is iterable, it skips the data and tries the next.
        """
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __new__(cls, dataset, map_func):
        is_iterable = isinstance(dataset, IterableDataset)
        if is_iterable:
            return _MapIterableDataset(dataset, map_func)
        else:
            return super().__new__(cls)

    def __getnewargs__(self):
        return self._dataset, self._map_func

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                print(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )

class DatasetMapper:
    """
    A customized version of `detectron2.data.DatasetMapper`
    """

    def __init__(self, dataset, is_train=True):
        self.augmentation = build_augmentation(is_train)

        # fmt: off
        self.img_format     = "BGR"

        # densepose_transform_srcs = [
        #     dataset.get_metadata().densepose_transform_src
        #     # MetadataCatalog.get(ds).densepose_transform_src
        #     # for ds in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
        # ]
        # assert len(densepose_transform_srcs) > 0
        # TODO: check that DensePose transformation data is the same for
        # all the datasets. Otherwise one would have to pass DB ID with
        # each entry to select proper transformation data. For now, since
        # all DensePose annotated data uses the same data semantics, we
        # omit this check.
        # densepose_transform_data_fpath = PathManager.get_local_path(densepose_transform_srcs[0])
        self.densepose_transform_data = DensePoseTransformData.load(dataset.get_metadata()["densepose_transform_src"])

        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = read_image(dataset_dict["file_name"], format=self.img_format)
        check_image_size(dataset_dict, image)

        image, transforms = apply_augmentations(self.augmentation, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        for anno in dataset_dict["annotations"]:
            anno.pop("segmentation", None)
            anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        # USER: Don't call transpose_densepose if you don't need
        annos = [
            self._transform_densepose(
                transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=None
                ),
                transforms,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        # if self.mask_on:
        #     self._add_densepose_masks_as_segmentation(annos, image_shape)

        instances = annotations_to_instances(annos, image_shape, mask_format="bitmask")
        densepose_annotations = [obj.get("densepose") for obj in annos]
        if densepose_annotations and not all(v is None for v in densepose_annotations):
            instances.gt_densepose = DensePoseList(
                densepose_annotations, instances.gt_boxes, image_shape
            )

        dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
        return dataset_dict

    def _transform_densepose(self, annotation, transforms):
        # Handle densepose annotations
        is_valid, reason_not_valid = DensePoseDataRelative.validate_annotation(annotation)
        if is_valid:
            densepose_data = DensePoseDataRelative(annotation, cleanup=True)
            densepose_data.apply_transform(transforms, self.densepose_transform_data)
            annotation["densepose"] = densepose_data
        else:
            # logger = logging.getLogger(__name__)
            # logger.debug("Could not load DensePose annotation: {}".format(reason_not_valid))
            DensePoseDataRelative.cleanup_annotation(annotation)
            # NOTE: annotations for certain instances may be unavailable.
            # 'None' is accepted by the DensePostList data structure.
            annotation["densepose"] = None
        return annotation
