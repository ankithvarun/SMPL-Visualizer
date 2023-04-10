# Copyright (c) Facebook, Inc. and its affiliates.
import os
import random
import itertools
import copy

from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch.utils.data.sampler import Sampler

from ..utils.structures import BoxMode, TorchSerializedList

from pycocotools.coco import COCO
from typing import Union, Callable

# COCO_TRAIN_IMAGE_DIR = "/scratch/coco/train2014"
# DENSEPOSE_METADATA_DIR = "../../metadata"
# DENSEPOSE_COCO_ANNOTATIONS_PATH = "../../annotations/densepose_coco_2014_train.json"

DENSEPOSE_MASK_KEY = "dp_masks"
DENSEPOSE_IUV_KEYS = ["dp_x", "dp_y", "dp_I", "dp_U", "dp_V"]
DENSEPOSE_KEYS = set(DENSEPOSE_IUV_KEYS + [DENSEPOSE_MASK_KEY])

# DENSEPOSE_MASK_KEY = "dp_masks"
# DENSEPOSE_IUV_KEYS_WITHOUT_MASK = ["dp_x", "dp_y", "dp_I", "dp_U", "dp_V"]
# DENSEPOSE_CSE_KEYS_WITHOUT_MASK = ["dp_x", "dp_y", "dp_vertex", "ref_model"]
# DENSEPOSE_ALL_POSSIBLE_KEYS = set(
#     DENSEPOSE_IUV_KEYS_WITHOUT_MASK + DENSEPOSE_CSE_KEYS_WITHOUT_MASK + [DENSEPOSE_MASK_KEY]
# )
# DENSEPOSE_METADATA_URL_PREFIX = "https://dl.fbaipublicfiles.com/densepose/data/"

class ParentDataset:
    def __init__(self, name, image_dir, metadata_dir, annotation_path, sampling_rate=1.0):
        self.name = name
        self.image_dir = image_dir
        self.metadata_dir = metadata_dir
        self.annotation_path = annotation_path
        self.sampling_rate = sampling_rate

    def get_metadata(self):
        """
        Returns metadata associated with COCO DensePose datasets

        Args:
        base_path: Optional[str]
            Base path used to load metadata from

        Returns:
        Dict[str, Any]
            Metadata in the form of a dictionary
        """
        meta = {
            "densepose_transform_src":  os.path.join(self.metadata_dir, "UV_symmetry_transforms.mat"),
            "densepose_smpl_subdiv": os.path.join(self.metadata_dir, "SMPL_subdiv.mat"),
            "densepose_smpl_subdiv_transform": os.path.join(self.metadata_dir, "SMPL_SUBDIV_TRANSFORM.mat")
        }
        return meta


    # def _load_coco_annotations(json_file: str):
    #     """
    #     Load COCO annotations from a JSON file

    #     Args:
    #         json_file: str
    #             Path to the file to load annotations from
    #     Returns:
    #         Instance of `pycocotools.coco.COCO` that provides access to annotations
    #         data
    #     """

    #     logger = logging.getLogger(__name__)
    #     timer = Timer()
    #     with contextlib.redirect_stdout(io.StringIO()):
    #         coco_api = COCO(json_file)
    #     if timer.seconds() > 1:
    #         logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))
    #     return coco_api


    # def _add_categories_metadata(self, categories: List[Dict[str, Any]]):
    #     self.meta_categories = {c["id"]: c["name"] for c in categories}
    #     print("Dataset {} categories: {}".format(self.name, self.meta_categories))


    # def _verify_annotations_have_unique_ids(json_file: str, anns: List[List[Dict[str, Any]]]):
    #     if "minival" in json_file:
    #         # Skip validation on COCO2014 valminusminival and minival annotations
    #         # The ratio of buggy annotations there is tiny and does not affect accuracy
    #         # Therefore we explicitly white-list them
    #         return
    #     ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    #     assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
    #         json_file
    #     )


    # def _maybe_add_bbox(obj: Dict[str, Any], ann_dict: Dict[str, Any]):
    #     if "bbox" not in ann_dict:
    #         return
    #     obj["bbox"] = ann_dict["bbox"]
    #     obj["bbox_mode"] = BoxMode.XYWH_ABS


    # def _maybe_add_segm(obj: Dict[str, Any], ann_dict: Dict[str, Any]):
    #     if "segmentation" not in ann_dict:
    #         return
    #     segm = ann_dict["segmentation"]
    #     if not isinstance(segm, dict):
    #         # filter out invalid polygons (< 3 points)
    #         segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
    #         if len(segm) == 0:
    #             return
    #     obj["segmentation"] = segm


    # def _maybe_add_keypoints(obj: Dict[str, Any], ann_dict: Dict[str, Any]):
    #     if "keypoints" not in ann_dict:
    #         return
    #     keypts = ann_dict["keypoints"]  # list[int]
    #     for idx, v in enumerate(keypts):
    #         if idx % 3 != 2:
    #             # COCO's segmentation coordinates are floating points in [0, H or W],
    #             # but keypoint coordinates are integers in [0, H-1 or W-1]
    #             # Therefore we assume the coordinates are "pixel indices" and
    #             # add 0.5 to convert to floating point coordinates.
    #             keypts[idx] = v + 0.5
    #     obj["keypoints"] = keypts


    # def _maybe_add_densepose(obj: Dict[str, Any], ann_dict: Dict[str, Any]):
    #     for key in DENSEPOSE_ALL_POSSIBLE_KEYS:
    #         if key in ann_dict:
    #             obj[key] = ann_dict[key]


    def combine_images_with_annotations(self, img_datas, ann_datas):
        ann_keys = ["iscrowd", "category_id"]
        dataset_dicts = []

        for img_dict, ann_dicts in zip(img_datas, ann_datas):
            record = {}
            record["file_name"] = os.path.join(self.image_dir, img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            record["image_id"] = img_dict["id"]
            # record["dataset"] = dataset_name
            objs = []
            for ann_dict in ann_dicts:
                assert ann_dict["image_id"] == record["image_id"]
                assert ann_dict.get("ignore", 0) == 0
                obj = {key: ann_dict[key] for key in ann_keys if key in ann_dict}
                
                # _maybe_add_bbox(obj, ann_dict)
                if "bbox" in ann_dict:
                    obj["bbox"] = ann_dict["bbox"]
                    obj["bbox_mode"] = BoxMode.XYWH_ABS

                # _maybe_add_segm(obj, ann_dict)
                if "segmentation" in ann_dict:
                    segm = ann_dict["segmentation"]
                    if not isinstance(segm, dict):
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            return
                    obj["segmentation"] = segm

                # _maybe_add_keypoints(obj, ann_dict)
                if "keypoints" in ann_dict:
                    keypts = ann_dict["keypoints"]  # list[int]
                    for idx, v in enumerate(keypts):
                        if idx % 3 != 2:
                            # COCO's segmentation coordinates are floating points in [0, H or W],
                            # but keypoint coordinates are integers in [0, H-1 or W-1]
                            # Therefore we assume the coordinates are "pixel indices" and
                            # add 0.5 to convert to floating point coordinates.
                            keypts[idx] = v + 0.5
                    obj["keypoints"] = keypts

                # _maybe_add_densepose(obj, ann_dict)
                for key in DENSEPOSE_KEYS:
                    if key in ann_dict:
                        obj[key] = ann_dict[key]

                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts


    # def get_contiguous_id_to_category_id_map(metadata):
    #     cat_id_2_cont_id = metadata.thing_dataset_id_to_contiguous_id
    #     cont_id_2_cat_id = {}
    #     for cat_id, cont_id in cat_id_2_cont_id.items():
    #         if cont_id in cont_id_2_cat_id:
    #             continue
    #         cont_id_2_cat_id[cont_id] = cat_id
    #     return cont_id_2_cat_id


    # def maybe_filter_categories_cocoapi(dataset_name, coco_api):
    #     meta = MetadataCatalog.get(dataset_name)
    #     cont_id_2_cat_id = get_contiguous_id_to_category_id_map(meta)
    #     cat_id_2_cont_id = meta.thing_dataset_id_to_contiguous_id
    #     # filter categories
    #     cats = []
    #     for cat in coco_api.dataset["categories"]:
    #         cat_id = cat["id"]
    #         if cat_id not in cat_id_2_cont_id:
    #             continue
    #         cont_id = cat_id_2_cont_id[cat_id]
    #         if (cont_id in cont_id_2_cat_id) and (cont_id_2_cat_id[cont_id] == cat_id):
    #             cats.append(cat)
    #     coco_api.dataset["categories"] = cats
    #     # filter annotations, if multiple categories are mapped to a single
    #     # contiguous ID, use only one category ID and map all annotations to that category ID
    #     anns = []
    #     for ann in coco_api.dataset["annotations"]:
    #         cat_id = ann["category_id"]
    #         if cat_id not in cat_id_2_cont_id:
    #             continue
    #         cont_id = cat_id_2_cont_id[cat_id]
    #         ann["category_id"] = cont_id_2_cat_id[cont_id]
    #         anns.append(ann)
    #     coco_api.dataset["annotations"] = anns
    #     # recreate index
    #     coco_api.createIndex()


    # def maybe_filter_and_map_categories_cocoapi(dataset_name, coco_api):
    #     meta = MetadataCatalog.get(dataset_name)
    #     category_id_map = meta.thing_dataset_id_to_contiguous_id
    #     # map categories
    #     cats = []
    #     for cat in coco_api.dataset["categories"]:
    #         cat_id = cat["id"]
    #         if cat_id not in category_id_map:
    #             continue
    #         cat["id"] = category_id_map[cat_id]
    #         cats.append(cat)
    #     coco_api.dataset["categories"] = cats
    #     # map annotation categories
    #     anns = []
    #     for ann in coco_api.dataset["annotations"]:
    #         cat_id = ann["category_id"]
    #         if cat_id not in category_id_map:
    #             continue
    #         ann["category_id"] = category_id_map[cat_id]
    #         anns.append(ann)
    #     coco_api.dataset["annotations"] = anns
    #     # recreate index
    #     coco_api.createIndex()


    # def create_video_frame_mapping(dataset_name, dataset_dicts):
    #     mapping = defaultdict(dict)
    #     for d in dataset_dicts:
    #         video_id = d.get("video_id")
    #         if video_id is None:
    #             continue
    #         mapping[video_id].update({d["frame_id"]: d["file_name"]})
    #     MetadataCatalog.get(dataset_name).set(video_frame_mapping=mapping)


    def load_coco_json(self):
        """
        Loads a JSON file with annotations in COCO instances format.
        Replaces `detectron2.data.datasets.coco.load_coco_json` to handle metadata
        in a more flexible way. Postpones category mapping to a later stage to be
        able to combine several datasets with different (but coherent) sets of
        categories.

        Args:

        annotations_json_file: str
            Path to the JSON file with annotations in COCO instances format.
        image_root: str
            directory that contains all the images
        dataset_name: str
            the name that identifies a dataset, e.g. "densepose_coco_2014_train"
        extra_annotation_keys: Optional[List[str]]
            If provided, these keys are used to extract additional data from
            the annotations.
        """
        coco_api = COCO(self.annotation_path)
        # _add_categories_metadata(dataset_name, coco_api.loadCats(coco_api.getCatIds()))
        self.meta_categories = {c["id"]: c["name"] for c in coco_api.loadCats(coco_api.getCatIds())}
        print("Dataset {} categories: {}".format(self.name, self.meta_categories))
        # sort indices for reproducible results
        img_ids = sorted(coco_api.imgs.keys())
        # imgs is a list of dicts, each looks something like:
        # {'license': 4,
        #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
        #  'file_name': 'COCO_val2014_000000001268.jpg',
        #  'height': 427,
        #  'width': 640,
        #  'date_captured': '2013-11-17 05:57:24',
        #  'id': 1268}

        # Randomly sample 10% of the images for faster training
        if self.sampling_rate < 1:
            img_ids = random.sample(img_ids, int(len(img_ids) * self.sampling_rate))
        

        imgs = coco_api.loadImgs(img_ids)
        # logger = logging.getLogger(__name__)
        print("Loaded {} images in COCO format from {}".format(len(imgs), self.annotation_path))
        # anns is a list[list[dict]], where each dict is an annotation
        # record for an object. The inner list enumerates the objects in an image
        # and the outer list enumerates over images.
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        # _verify_annotations_have_unique_ids(annotations_json_file, anns)
        if "minival" not in self.annotation_path:
            # Skip validation on COCO2014 valminusminival and minival annotations
            # The ratio of buggy annotations there is tiny and does not affect accuracy
            # Therefore we explicitly white-list them
            ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
            assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
                self.annotation_path
            )
            
        dataset_records = self.combine_images_with_annotations(imgs, anns)
        return dataset_records


    def register(self):
        """
        Registers provided COCO DensePose dataset

        Args:
        dataset_data: CocoDatasetInfo
            Dataset data
        datasets_root: Optional[str]
            Datasets root folder (default: None)
        """
        # annotations_fpath = maybe_prepend_base_path(datasets_root, dataset_data.annotations_fpath)
        # images_root = maybe_prepend_base_path(datasets_root, dataset_data.images_root)

        # def load_annotations():
        #     return load_coco_json(
        #         annotations_json_file=annotations_fpath,
        #         image_root=images_root,
        #         dataset_name=dataset_data.name,
        #     )

        # DatasetCatalog.register(dataset_data.name, load_annotations)
        # MetadataCatalog.get(dataset_data.name).set(
        #     json_file=annotations_fpath,
        #     image_root=images_root,
        #     **get_metadata(DENSEPOSE_METADATA_URL_PREFIX)
        # )

        self.dataset_dict = self.load_coco_json()
        self.meta = self.get_metadata()

        # print(self.dataset_dict[0])
        # print(self.meta_categories)

    # def register_datasets(
    #     datasets_data: Iterable[CocoDatasetInfo], datasets_root: Optional[str] = None
    # ):
    #     """
    #     Registers provided COCO DensePose datasets

    #     Args:
    #     datasets_data: Iterable[CocoDatasetInfo]
    #         An iterable of dataset datas
    #     datasets_root: Optional[str]
    #         Datasets root folder (default: None)
    #     """
    #     for dataset_data in datasets_data:
    #         register_dataset(dataset_data, datasets_root)

class DatasetFromList(Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(
        self,
        lst: list,
        copy: bool = True,
        serialize: Union[bool, Callable] = True,
    ):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool or callable): whether to serialize the stroage to other
                backend. If `True`, the default serialize method will be used, if given
                a callable, the callable will be used as serialize method.
        """
        self._lst = lst
        self._copy = copy
        if not isinstance(serialize, (bool, Callable)):
            raise TypeError(f"Unsupported type for argument `serailzie`: {serialize}")
        self._serialize = serialize is not False

        if self._serialize:
            serialize_method = (
                serialize
                if isinstance(serialize, Callable)
                else TorchSerializedList
            )
            print(f"Serializing the dataset using: {serialize_method}")
            self._lst = serialize_method(self._lst)

    def __len__(self):
        return len(self._lst)

    def __getitem__(self, idx):
        if self._copy and not self._serialize:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]

def _shard_iterator_dataloader_worker(iterable):
    # Shard the iterable if we're currently inside pytorch dataloader worker.
    worker_info = get_worker_info()
    if worker_info is None or worker_info.num_workers == 1:
        # do nothing
        yield from iterable
    else:
        yield from itertools.islice(iterable, worker_info.id, None, worker_info.num_workers)

class ToIterableDataset(IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    """

    def __init__(self, dataset, sampler, shard_sampler = True):
        """
        Args:
            dataset: an old-style dataset with ``__getitem__``
            sampler: a cheap iterable that produces indices to be applied on ``dataset``.
            shard_sampler: whether to shard the sampler based on the current pytorch data loader
                worker id. When an IterableDataset is forked by pytorch's DataLoader into multiple
                workers, it is responsible for sharding its data based on worker id so that workers
                don't produce identical data.

                Most samplers (like our TrainingSampler) do not shard based on dataloader worker id
                and this argument should be set to True. But certain samplers may be already
                sharded, in that case this argument should be set to False.
        """
        assert not isinstance(dataset, IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler
        self.shard_sampler = shard_sampler

    def __iter__(self):
        if not self.shard_sampler:
            sampler = self.sampler
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker. So we should only keep 1/N of the ids on
            # each worker. The assumption is that sampler is cheap to iterate so it's fine to
            # discard ids in workers.
            sampler = _shard_iterator_dataloader_worker(self.sampler)
        for idx in sampler:
            yield self.dataset[idx]

    def __len__(self):
        return len(self.sampler)

class AspectRatioGroupedDataset(IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                data = bucket[:]
                # Clear bucket first, because code after yield is not
                # guaranteed to execute
                del bucket[:]
                yield data