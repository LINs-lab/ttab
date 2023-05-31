# -*- coding: utf-8 -*-
from typing import Iterable, Optional, Tuple, Type, Union

import torch
from ttab.api import Batch, PyTorchDataset
from ttab.loads.datasets.datasets import WrapperDataset
from ttab.scenarios import Scenario

D = Union[torch.utils.data.Dataset, PyTorchDataset]


class BaseLoader(object):
    def __init__(self, dataset: PyTorchDataset):
        self.dataset = dataset

    def iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = False,
        ref_num_data: Optional[int] = None,
        num_workers: int = 1,
        sampler: Optional[torch.utils.data.Sampler] = None,
        generator: Optional[torch.Generator] = None,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> Iterable[Tuple[int, float, Batch]]:
        yield from self.dataset.iterator(
            batch_size,
            shuffle,
            repeat,
            ref_num_data,
            num_workers,
            sampler,
            generator,
            pin_memory,
            drop_last,
        )


def _init_dataset(dataset: D, device: str) -> PyTorchDataset:
    if isinstance(dataset, torch.utils.data.Dataset):
        return WrapperDataset(dataset, device)
    else:
        return dataset


def get_test_loader(dataset: D, device: str) -> Type[BaseLoader]:
    dataset: PyTorchDataset = _init_dataset(dataset, device)
    return BaseLoader(dataset)


def get_auxiliary_loader(dataset: D, device: str) -> Type[BaseLoader]:
    dataset: PyTorchDataset = _init_dataset(dataset, device)
    return BaseLoader(dataset)
