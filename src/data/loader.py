import operator
import torch
import torch.utils.data as torchdata

from ..utils.comm import get_world_size, seed_all_rng
from .dataset import DENSEPOSE_IUV_KEYS, ToIterableDataset, AspectRatioGroupedDataset, DatasetFromList
from .sampler import TrainingSampler
from .mapper import MapDataset, DatasetMapper

def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)

def keep_instance(instance):
    def has_annotations(instance):
        return "annotations" in instance

    def has_only_crowd_anotations(instance):
        for ann in instance["annotations"]:
            if ann.get("is_crowd", 0) == 0:
                return False
        return True

    def has_densepose_annotations(instance):
        for ann in instance["annotations"]:
            if all(key in ann for key in DENSEPOSE_IUV_KEYS):
                return True
        return False

    return has_annotations(instance) and not has_only_crowd_anotations(instance) and has_densepose_annotations(instance)

def build_train_loader(parent_dataset, total_batch_size, num_workers):
    dataset_dicts = [d for d in parent_dataset.dataset_dict if keep_instance(d)]
    mapper = DatasetMapper(parent_dataset, True)
    if isinstance(dataset_dicts, list):
        dataset_dicts = DatasetFromList(dataset_dicts, copy=False)
    dataset = MapDataset(dataset_dicts, mapper)
    sampler = TrainingSampler(len(dataset_dicts))

    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset = ToIterableDataset(dataset, sampler)

    data_loader = torchdata.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict
    data_loader = AspectRatioGroupedDataset(data_loader, batch_size)
    return data_loader
    # Filter images without usable annotations

