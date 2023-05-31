# -*- coding: utf-8 -*-
import itertools
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data.dataset import random_split

State = List[torch.Tensor]
Gradient = List[torch.Tensor]
Parameters = List[torch.Tensor]
Loss = float
Quality = Mapping[str, float]


class Batch(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __len__(self) -> int:
        return len(self._x)

    def to(self, device) -> "Batch":
        return Batch(self._x.to(device), self._y.to(device))

    def __getitem__(self, index):
        return self._x[index], self._y[index]


class GroupBatch(object):
    def __init__(self, x, y, g):
        self._x = x
        self._y = y
        self._g = g

    def __len__(self) -> int:
        return len(self._x)

    def to(self, device) -> "Batch":
        return GroupBatch(self._x.to(device), self._y.to(device), self._g.to(device))

    def __getitem__(self, index):
        return self._x[index], self._y[index], self._g[index]


class Dataset:
    def random_split(self, fractions: List[float]) -> List["Dataset"]:
        pass

    def iterator(
        self, batch_size: int, shuffle: bool, repeat=True
    ) -> Iterable[Tuple[float, Batch]]:
        pass

    def __len__(self) -> int:
        pass


class PyTorchDataset(object):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        device: str,
        prepare_batch: Callable,
        num_classes: int,
    ):
        self._set = dataset
        self._device = device
        self._prepare_batch = prepare_batch
        self._num_classes = num_classes

    def __len__(self):
        return len(self._set)

    def replace_indices(
        self,
        indices_pattern: str = "original",
        new_indices: List[int] = None,
        random_seed: int = None,
    ) -> None:
        """Change the order of dataset indices in a particular pattern."""
        if indices_pattern == "original":
            pass
        elif indices_pattern == "random_shuffle":
            rng = np.random.default_rng(random_seed)
            rng.shuffle(self.dataset.indices)
        elif indices_pattern == "new":
            if new_indices is None:
                raise ValueError("new_indices should be specified.")
            self.dataset.update_indices(new_indices=new_indices)
        else:
            raise NotImplementedError

    def query_dataset_attr(self, attr_name: str) -> Any:
        return getattr(self._set, attr_name, None)

    @property
    def dataset(self):
        return self._set

    @property
    def num_classes(self):
        return self._num_classes

    def no_split(self) -> List[Dataset]:
        return [
            PyTorchDataset(
                dataset=self._set,
                device=self._device,
                prepare_batch=self._prepare_batch,
                num_classes=self._num_classes,
            )
        ]

    def random_split(self, fractions: List[float], seed: int = 0) -> List[Dataset]:
        lengths = [int(f * len(self._set)) for f in fractions]
        lengths[0] += len(self._set) - sum(lengths)
        return [
            PyTorchDataset(
                dataset=split,
                device=self._device,
                prepare_batch=self._prepare_batch,
                num_classes=self._num_classes,
            )
            for split in random_split(
                self._set, lengths, torch.Generator().manual_seed(seed)
            )
        ]

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
        _num_batch = 1 if not drop_last else 0
        if ref_num_data is None:
            num_batches = int(len(self) / batch_size + _num_batch)
        else:
            num_batches = int(ref_num_data / batch_size + _num_batch)
        if sampler is not None:
            shuffle = False

        loader = torch.utils.data.DataLoader(
            self._set,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            num_workers=num_workers,
            sampler=sampler,
            generator=generator,
        )

        step = 0
        for _ in itertools.count() if repeat else [0]:
            for i, batch in enumerate(loader):
                step += 1
                epoch_fractional = float(step) / num_batches
                yield step, epoch_fractional, self._prepare_batch(batch, self._device)

    def record_class_distribution(
        self,
        targets: Union[List, np.ndarray],
        indices: Union[List, np.ndarray],
        print_fn: Callable = print,
        is_train: bool = True,
        display: bool = True,
    ):
        targets_np = np.array(targets)
        unique_elements, counts_elements = np.unique(
            targets_np[indices] if indices is not None else targets_np,
            return_counts=True,
        )
        element_counts = list(zip(unique_elements, counts_elements))

        if display:
            print_fn(
                f"\tThe histogram of the targets in {'train' if is_train else 'test'}: {element_counts}"
            )
        return element_counts
