# -*- coding: utf-8 -*-
from typing import Callable, List, NamedTuple, Optional

import numpy as np
import torch
from PIL import Image

# map data_name to a distribution shift.
# Let's say the data_name could be in the form of 1) <a>, 2), <a>_<b>, and 3) <a>_<b>_<c>

data2shift = dict(
    cifar10="no_shift",
    cifar100="no_shift",
    cifar10_c="synthetic",
    cifar100_c="synthetic",
    cifar10_1="natural",
    cifar10_shiftedlabel="natural",
    cifar100_shiftedlabel="natural",
    cifar10_temporal="temporal",
    imagenet="no_shift",
    imagenet_c="synthetic",
    imagenet_a="natural",
    imagenet_r="natural",
    # imagenet_v2_<test-set-letter (version)>
    imagenet_v2="natural",
    officehome_art="natural",
    officehome_clipart="natural",
    officehome_product="natural",
    officehome_realworld="natural",
    pacs_art="natural",
    pacs_cartoon="natural",
    pacs_photo="natural",
    pacs_sketch="natural",
    mnist="no_shift",
    coloredmnist="synthetic",
    waterbirds="natural",
    yearbook="natural",
)


class SyntheticShiftProperty(NamedTuple):
    shift_degree: int
    shift_name: str

    version: str = "stochastic"
    has_shift: bool = True


class NaturalShiftProperty(NamedTuple):
    version: str = None
    has_shift: bool = True


class NoShiftProperty(NamedTuple):
    has_shift: bool = False


class ShiftedData(object):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def update_indices(self, new_indices: List[int]) -> None:
        """Update the indices of the dataset after applying shift."""
        self.dataset.indices = new_indices
        self.dataset.data_size = len(self.indices)

    @property
    def data(self):
        return self.dataset.data

    @property
    def targets(self):
        return self.dataset.targets

    @property
    def indices(self):
        return self.dataset.indices

    @property
    def data_size(self):
        return self.dataset.data_size

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_index(self):
        return self.dataset.class_to_index

    @property
    def group_array(self):
        return getattr(self.dataset, "group_array", None)


class NoShiftedData(ShiftedData):
    """
    Dataset-like object, but only access a subset of it.
    And it applies NO shift to the data when __getitem__.
    """

    def __init__(self, data_name, dataset: torch.utils.data.Dataset):
        super().__init__(dataset)
        # initialize corruption class
        self.data_name = data_name


class NaturalShiftedData(ShiftedData):
    """
    Dataset-like object, but only access a subset of it.
    And it will reload the data with natural shift. It will apply NO shift to the data when __getitem__.
    """

    def __init__(
        self,
        data_name,
        dataset: torch.utils.data.Dataset,
        new_data: torch.utils.data.Dataset,
    ):
        super().__init__(dataset)
        # replace original data/targets with new data/targets
        self.dataset.data = new_data.data
        self.dataset.targets = new_data.targets
        self.data_name = data_name


class SyntheticShiftedData(ShiftedData):
    """
    Dataset-like object, but only access a subset of it.
    And it applies corruptions to the data when __getitem__.
    """

    def __init__(
        self,
        data_name: str,
        dataset: torch.utils.data.Dataset,
        seed: int,
        synthetic_class: Callable,
        version: str,
        **kwargs,
    ):
        super().__init__(dataset)
        self.data_name = data_name
        self.version = version  # either stochastic or determinstic

        # initialize corruption class
        if any([name in data_name for name in ["cifar", "imagenet"]]):
            self.synthetic_ops = synthetic_class(data_name, seed, kwargs["severity"])
        elif "mnist" in data_name:
            self.synthetic_ops = synthetic_class
        else:
            NotImplementedError(
                f"synthetic shift for {data_name} is not supported in TTAB."
            )

    def apply_corruption(self):
        """Apply corruption to the clean dataset."""
        corrupted_imgs = []

        for index in range(self.dataset.data_size):
            img_array = self.dataset.data[index]
            img = Image.fromarray(img_array)
            img = self.synthetic_ops.apply(img)
            corrupted_imgs.append(img)

        # replace data.
        self.dataset.data = np.stack(corrupted_imgs, axis=0)

    def prepare_colored_mnist(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        return self.synthetic_ops.apply(self.dataset, transform, target_transform)
