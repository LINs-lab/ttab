# -*- coding: utf-8 -*-
import functools
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch import randperm
from torch._utils import _accumulate
from ttab.api import Batch, GroupBatch, PyTorchDataset
from ttab.configs.datasets import dataset_defaults
from ttab.loads.datasets.dataset_shifts import (
    NaturalShiftedData,
    NoShiftedData,
    SyntheticShiftedData,
)
from ttab.loads.datasets.utils.preprocess_toolkit import get_transform

group_attributes = {
    "waterbirds": 4,  # number of groups in the dataset.
}


class WrapperDataset(PyTorchDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, device: str = "cuda"):
        # init other utility functions.
        super().__init__(
            dataset,
            device=device,
            prepare_batch=WrapperDataset.prepare_batch,
            num_classes=None,
        )

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class CIFARDataset(PyTorchDataset):
    """A class to load different CIFAR datasets for training and testing.

    CIFAR10-C/CIFAR100-C: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
        https://arxiv.org/abs/1903.12261

    CIFAR10.1: Do CIFAR-10 Classifiers Generalize to CIFAR-10?
        https://arxiv.org/abs/1806.00451
    """

    def __init__(
        self,
        root: str,
        data_name: str,
        split: str,
        device: str = "cuda",
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
        input_size: int = None,
        data_size: int = None,
    ):

        # setup data.
        if "10" in data_name and "100" not in data_name:
            num_classes = dataset_defaults["cifar10"]["statistics"]["n_classes"]
            normalize = {
                "mean": dataset_defaults["cifar10"]["statistics"]["mean"],
                "std": dataset_defaults["cifar10"]["statistics"]["std"],
            }
            dataset_fn = datasets.CIFAR10
        elif "100" in data_name:
            num_classes = dataset_defaults["cifar100"]["statistics"]["n_classes"]
            normalize = {
                "mean": dataset_defaults["cifar100"]["statistics"]["mean"],
                "std": dataset_defaults["cifar100"]["statistics"]["std"],
            }
            dataset_fn = datasets.CIFAR100
        else:
            raise NotImplementedError(f"invalid data_name={data_name}.")

        # data transform.
        if input_size is None:
            input_size = 32
        is_train = True if split == "train" else False
        augment = True if data_augment else False
        if augment:
            scale_size = 40 if input_size == 32 else None
        else:
            scale_size = input_size

        self.transform = get_transform(
            data_name,
            input_size=input_size,
            scale_size=scale_size,
            normalize=normalize,
            augment=augment,
        )
        self.target_transform = None

        # init dataset.
        basic_conf = {
            "root": root,
            "train": is_train,
            "transform": self.transform,
            "target_transform": self.target_transform,
            "download": True,
        }

        if "deterministic" in data_name:
            data_shift_class = functools.partial(
                NoShiftedData, data_name=data_name
            )  # deterministic data is directly loaded from extrinsic files.

        # basic check.
        assert data_shift_class is not None, "data_shift_class is required."

        # configure dataset.
        clean_dataset = dataset_fn(**basic_conf)
        num_samples = len(clean_dataset) if data_size is None else data_size
        if issubclass(data_shift_class.func, NoShiftedData):
            if "deterministic" in data_name:
                # get names
                # support string like "cifar10_c_deterministic-gaussian_noise-5"
                _new_data_names = data_name.split("_", 2)
                _shift_name = _new_data_names[-1].split("-")[1]
                _shift_degree = int(_new_data_names[-1].split("-")[-1])

                # get data
                data_raw = self._load_deterministic_cifar_c(
                    root, _shift_name, _shift_degree
                )

                # construct data_class
                dataset = ImageArrayDataset(
                    data=data_raw[:num_samples],
                    targets=clean_dataset.targets[:num_samples],
                    classes=clean_dataset.classes,
                    class_to_index=clean_dataset.class_to_idx,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
            else:
                dataset = ImageArrayDataset(
                    data=clean_dataset.data[:num_samples],
                    targets=clean_dataset.targets[:num_samples],
                    classes=clean_dataset.classes,
                    class_to_index=clean_dataset.class_to_idx,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
            dataset = data_shift_class(dataset=dataset)
        elif issubclass(data_shift_class.func, SyntheticShiftedData):
            dataset = ImageArrayDataset(
                data=clean_dataset.data[:num_samples],
                targets=clean_dataset.targets[:num_samples],
                classes=clean_dataset.classes,
                class_to_index=clean_dataset.class_to_idx,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            dataset = data_shift_class(dataset=dataset)
            dataset.apply_corruption()
        elif issubclass(data_shift_class.func, NaturalShiftedData):
            dataset = ImageArrayDataset(
                data=clean_dataset.data,
                targets=clean_dataset.targets,
                classes=clean_dataset.classes,
                class_to_index=clean_dataset.class_to_idx,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            dataset = data_shift_class(dataset=dataset)
            num_samples = min(num_samples, len(dataset.data))
            new_indices = list([x for x in range(0, num_samples)])
            dataset.update_indices(new_indices)
        else:
            NotImplementedError(
                f"invalid data_shift_class={data_shift_class} for {data_name}."
            )

        # init other utility functions.
        super().__init__(
            dataset,
            device=device,
            prepare_batch=CIFARDataset.prepare_batch,
            num_classes=num_classes,
        )

    def _download_cifar_c(self):
        pass

    def _load_deterministic_cifar_c(
        self, root: str, shift_name: str, shift_degree: int
    ) -> np.ndarray:
        domain_path = os.path.join(root + "_c", shift_name + ".npy")

        if not os.path.exists(domain_path):
            # download data from website: https://zenodo.org/record/2535967#.YxS6D-wzY-R
            raise ValueError("Please download cifar_c data from the website.")

        data_raw = np.load(domain_path)
        data_raw = data_raw[(shift_degree - 1) * 10000 : shift_degree * 10000]
        return data_raw

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class ImageNetDataset(PyTorchDataset):
    def __init__(
        self,
        root,
        data_name: str,
        split: str = "test",
        device: str = "cuda",
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
    ):
        # setup data.
        is_train = True if split == "train" else False
        self.transform = get_transform(
            "imagenet", augment=any([is_train, data_augment]), color_process=False
        )
        self.target_transform = None
        num_classes = dataset_defaults["imagenet"]["statistics"]["n_classes"]

        if "deterministic" in data_name:
            data_shift_class = functools.partial(
                NoShiftedData, data_name=data_name
            )  # deterministic data is directly loaded from extrinsic files.

        # basic check.
        assert data_shift_class is not None, "data_shift_class is required."

        # configure dataset.
        if issubclass(data_shift_class.func, NoShiftedData):
            if "deterministic" in data_name:
                _new_data_names = data_name.split(
                    "_", 2
                )  # support string like "cifar10_c_deterministic-gaussian_noise-5"
                _shift_name = _new_data_names[-1].split("-")[1]
                _shift_degree = _new_data_names[-1].split("-")[-1]

                validdir = os.path.join(root, "imagenet-c", _shift_name, _shift_degree)
                dataset = ImageFolderDataset(
                    root=validdir,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
            else:
                # TODO: how to load in-distribution imagenet test data?
                validdir = os.path.join(root, "val")
                dataset = ImageFolderDataset(
                    root=validdir,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
            dataset = data_shift_class(dataset=dataset)
        elif issubclass(data_shift_class.func, SyntheticShiftedData):
            validdir = os.path.join(root, "val")
            dataset = ImageFolderDataset(
                root=validdir,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            dataset = data_shift_class(dataset=dataset)
            dataset.apply_corruption()

        super().__init__(
            dataset,
            device=device,
            prepare_batch=ImageNetDataset.prepare_batch,
            num_classes=num_classes,
        )

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class OfficeHomeDataset(PyTorchDataset):
    """
    A class to load officehome dataset for training and testing.
    Deep Hashing Network for Unsupervised Domain Adaptation: https://paperswithcode.com/paper/deep-hashing-network-for-unsupervised-domain
    """

    DOMAINS: list = ["art", "clipart", "product", "realworld"]

    def __init__(
        self,
        root: str,
        device: str = "cuda",
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
        data_size: int = None,
        random_seed: int = None,
    ):
        # some basic dataset configuration.
        normalize = transforms.Normalize(
            dataset_defaults["officehome"]["statistics"]["mean"],
            dataset_defaults["officehome"]["statistics"]["std"],
        )
        num_classes = dataset_defaults["officehome"]["statistics"]["n_classes"]
        self.transform = get_transform(
            "officehome", normalize=normalize, augment=data_augment, color_process=False
        )
        self.target_transform = None

        # set up data.
        dataset = ImageFolderDataset(
            root=root, transform=self.transform, target_transform=self.target_transform
        )
        if data_size is not None:
            dataset.trim_dataset(data_size, random_seed)

        if data_shift_class is not None:
            dataset = data_shift_class(dataset=dataset)

        super().__init__(
            dataset=dataset,
            device=device,
            prepare_batch=OfficeHomeDataset.prepare_batch,
            num_classes=num_classes,
        )

    def split_data(
        self, fractions: List[float], augment: List[bool], seed: int = None
    ) -> List[PyTorchDataset]:
        """This function is used to divide the dataset into two or more than two splits."""
        assert len(fractions) == len(augment)
        lengths = [int(f * len(self.dataset)) for f in fractions]
        lengths[0] += len(self.dataset) - sum(lengths)

        indices = randperm(
            sum(lengths), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        sub_indices = [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]

        normalize = transforms.Normalize(
            dataset_defaults["officehome"]["statistics"]["mean"],
            dataset_defaults["officehome"]["statistics"]["std"],
        )
        sub_datasets = [
            SubDataset(
                data=self.dataset.data,
                targets=self.dataset.targets,
                indices=sub_indices[i],
                transform=get_transform(
                    "officehome",
                    normalize=normalize,
                    augment=augment[i],
                    color_process=False,
                ),
                target_transform=None,
            )
            for i in range(len(sub_indices))
        ]

        return [
            PyTorchDataset(
                dataset=dataset,
                device=self._device,
                prepare_batch=OfficeHomeDataset.prepare_batch,
                num_classes=self.num_classes,
            )
            for dataset in sub_datasets
        ]

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class PACSDataset(PyTorchDataset):
    """
    A class to load officehome dataset for training and testing.
    Deep Hashing Network for Unsupervised Domain Adaptation: https://paperswithcode.com/paper/deep-hashing-network-for-unsupervised-domain
    """

    DOMAINS: list = ["art", "cartoon", "photo", "sketch"]

    def __init__(
        self,
        root: str,
        device: str = "cuda",
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
        data_size: int = None,
        random_seed: int = None,
    ):
        # some basic dataset configuration.
        normalize = transforms.Normalize(
            dataset_defaults["pacs"]["statistics"]["mean"],
            dataset_defaults["pacs"]["statistics"]["std"],
        )
        num_classes = dataset_defaults["pacs"]["statistics"]["n_classes"]
        self.transform = get_transform(
            "pacs", normalize=normalize, augment=data_augment, color_process=False
        )
        self.target_transform = None

        # set up data.
        dataset = ImageFolderDataset(
            root=root, transform=self.transform, target_transform=self.target_transform
        )
        if data_size is not None:
            dataset.trim_dataset(
                data_size, random_seed
            )  # trim indices, so it will also control new data.

        if data_shift_class is not None:
            dataset = data_shift_class(dataset=dataset)

        super().__init__(
            dataset=dataset,
            device=device,
            prepare_batch=OfficeHomeDataset.prepare_batch,
            num_classes=num_classes,
        )

    def split_data(
        self, fractions: List[float], augment: List[bool], seed: int = None
    ) -> List[PyTorchDataset]:
        """This function is used to divide the dataset into two or more than two splits."""
        assert len(fractions) == len(augment)
        lengths = [int(f * len(self.dataset)) for f in fractions]
        lengths[0] += len(self.dataset) - sum(lengths)

        indices = randperm(
            sum(lengths), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        sub_indices = [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]

        normalize = transforms.Normalize(
            dataset_defaults["pacs"]["statistics"]["mean"],
            dataset_defaults["pacs"]["statistics"]["std"],
        )
        sub_datasets = [
            SubDataset(
                data=self.dataset.data,
                targets=self.dataset.targets,
                indices=sub_indices[i],
                transform=get_transform(
                    "officehome",
                    normalize=normalize,
                    augment=augment[i],
                    color_process=False,
                ),
                target_transform=None,
            )
            for i in range(len(sub_indices))
        ]

        return [
            PyTorchDataset(
                dataset=dataset,
                device=self._device,
                prepare_batch=OfficeHomeDataset.prepare_batch,
                num_classes=self.num_classes,
            )
            for dataset in sub_datasets
        ]

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class ColoredMNIST(PyTorchDataset):
    def __init__(
        self,
        root: str,
        data_name: str,
        split: str,
        device: str = "cuda",
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
    ):
        self.data_name = data_name
        self.split = split
        # set up transform
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    dataset_defaults["coloredmnist"]["statistics"]["mean"],
                    dataset_defaults["coloredmnist"]["statistics"]["std"],
                ),
            ]
        )
        self.target_transform = None

        # set up data.
        original_dataset = datasets.mnist.MNIST(
            root,
            train=True,
            download=True,
        )
        num_classes = 2

        # init dataset.
        assert issubclass(
            data_shift_class.func, SyntheticShiftedData
        ), "ColoredMNIST belongs to synthetic shift type."
        dataset = data_shift_class(
            dataset=original_dataset,
        )

        dataset = dataset.prepare_colored_mnist(
            transform=self.transform, target_transform=self.target_transform
        )

        # init other utility functions.
        super().__init__(
            dataset=data_shift_class(dataset=dataset[split]),
            device=device,
            prepare_batch=ColoredMNIST.prepare_batch,
            num_classes=num_classes,
        )

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class WBirdsDataset(PyTorchDataset):
    def __init__(
        self,
        root: str,
        split: str,
        device: str,
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
    ):
        # some basic dataset configuration.
        normalize = transforms.Normalize(
            dataset_defaults["waterbirds"]["statistics"]["mean"],
            dataset_defaults["waterbirds"]["statistics"]["std"],
        )
        num_classes = dataset_defaults["waterbirds"]["statistics"]["n_classes"]
        self.transform = get_transform(
            "waterbirds", normalize=normalize, augment=data_augment, color_process=False
        )
        self.target_transform = None

        # set up data
        assert os.path.exists(
            root
        ), f"{root} does not exist yet, please generate the dataset first."

        # read in metadata.
        metadata_df = pd.read_csv(os.path.join(root, "metadata.csv"))

        split_dict = {
            "train": 0,  # the distribution of cofounder effect: 95%-5%
            "val": 1,  # 50%-50%
            "test": 2,  # 50%-50%
        }
        split_array = metadata_df["split"].values
        filename_array = metadata_df["img_filename"].values[
            split_array == split_dict[split]
        ]

        # Get the y values
        y_array = metadata_df["y"].values[split_array == split_dict[split]]
        self.target_name = "waterbird_complete95"

        # waterbirds dataset has only one confounder: places.
        confounder_array = metadata_df["place"].values[split_array == split_dict[split]]
        self.n_confounders = 1
        self.confounder_names = ["forest2water2"]
        # map to groups
        self.n_groups = pow(2, 2)
        group_array = (y_array * (self.n_groups / 2) + confounder_array).astype("int")
        self._group_counts = (
            (torch.arange(self.n_groups).unsqueeze(1) == torch.LongTensor(group_array))
            .sum(1)
            .float()
        )

        classes = [
            "0 - landbird",
            "1 - waterbird",
        ]
        class_to_index = {
            "0 - landbird": 0,
            "1 - waterbird": 1,
        }

        dataset = ConfounderDataset(
            root=root,
            data=None,
            filename_array=filename_array,
            targets=list(y_array),
            group_array=group_array,
            classes=classes,
            class_to_index=class_to_index,
            transform=self.transform,
            target_transform=self.target_transform,
        )

        if data_shift_class is not None:
            dataset = data_shift_class(dataset=dataset)

        super().__init__(
            dataset=dataset,
            device=device,
            prepare_batch=WBirdsDataset.prepare_batch,
            num_classes=num_classes,
        )

    def split_dataset(
        self, fractions: List[float], augment: List[bool], seed: int = None
    ) -> List[PyTorchDataset]:
        """This function is used to divide the dataset into two or more than two splits."""
        assert len(fractions) == len(augment)
        lengths = [int(f * len(self.dataset)) for f in fractions]
        lengths[0] += len(self.dataset) - sum(lengths)

        indices = randperm(
            sum(lengths), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        sub_indices = [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]

        normalize = transforms.Normalize(
            dataset_defaults["waterbirds"]["statistics"]["mean"],
            dataset_defaults["waterbirds"]["statistics"]["std"],
        )
        sub_datasets = [
            SubDataset(
                data=self.dataset.data,
                targets=self.dataset.targets,
                indices=sub_indices[i],
                transform=get_transform(
                    "waterbirds",
                    normalize=normalize,
                    augment=augment[i],
                    color_process=False,
                ),
                target_transform=None,
            )
            for i in range(len(sub_indices))
        ]

        return [
            PyTorchDataset(
                dataset=dataset,
                device=self._device,
                prepare_batch=WBirdsDataset.prepare_batch,
                num_classes=self.num_classes,
            )
            for dataset in sub_datasets
        ]

    def group_str(self, group_idx: int) -> str:
        y = group_idx // (self.n_groups / self.num_classes)
        c = group_idx % (self.n_groups // self.num_classes)

        group_name = f"{self.target_name} = {int(y)}"
        bin_str = format(int(c), f"0{self.n_confounders}b")[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f", {attr_name} = {bin_str[attr_idx]}"
        return group_name

    def group_counts(self):
        return self._group_counts

    @staticmethod
    def prepare_batch(batch, device):
        return GroupBatch(*batch).to(device)


class MergeMultiDataset(PyTorchDataset):
    """MergeMultiDataset combines a list of sub-datasets as one augmented dataset"""

    def __init__(self, datasets: List[PyTorchDataset]):
        self.datasets = datasets
        self.device = datasets[0]._device

        # some basic dataset configuration TODO: add a warning to the log
        self.transform = datasets[0].transform
        self.target_transform = datasets[0].target_transform
        num_classes = datasets[0].num_classes
        classes = datasets[0].query_dataset_attr("classes")
        class_to_index = datasets[0].query_dataset_attr("class_to_index")
        data_shift_class = functools.partial(NoShiftedData, data_name="MergedDataset")

        (
            merged_data,
            merged_targets,
            merged_indices,
            merged_group_arrays,
        ) = self.merge_datasets(datasets)
        if isinstance(self.datasets[0], WBirdsDataset):
            dataset = ConfounderDataset(
                root=None,
                data=merged_data,
                filename_array=None,
                targets=merged_targets,
                group_array=merged_group_arrays,
                classes=classes,
                class_to_index=class_to_index,
                transform=self.transform,
                target_transform=self.target_transform,
            )
        else:
            dataset = ImageArrayDataset(
                data=merged_data,
                targets=merged_targets,
                classes=classes,
                class_to_index=class_to_index,
                transform=self.transform,
                target_transform=self.target_transform,
            )
        dataset = data_shift_class(dataset=dataset)
        dataset.update_indices(merged_indices)

        super().__init__(
            dataset=dataset,
            device=self.device,
            prepare_batch=self.datasets[0].prepare_batch,
            num_classes=num_classes,
        )

    @staticmethod
    def merge_datasets(
        datasets: List[PyTorchDataset],
    ) -> Tuple[Union[List, np.ndarray], List[int], List[int], np.ndarray]:
        """Merge a list of datasets into one dataset through concatenating data, targets, indices and group_array."""
        merged_data = []
        merged_targets = []
        merged_indices = []
        if isinstance(datasets[0], WBirdsDataset):
            merged_group_arrays = []
        else:
            merged_group_arrays = None
        cumulative_size = 0

        all_has_same_type = all(
            isinstance(dataset, type(datasets[0])) for dataset in datasets
        )
        if not all_has_same_type:
            raise ValueError("All datasets to be merged should be of the same type.")

        for dataset in datasets:
            data = dataset.query_dataset_attr("data")
            indices = dataset.query_dataset_attr("indices")
            if type(data) == list:
                merged_data += data
            elif type(data) == np.ndarray:
                merged_data.append(data)
            else:
                raise NotImplementedError
            merged_targets += dataset.query_dataset_attr("targets")
            if merged_group_arrays is not None:
                merged_group_arrays.append(dataset.query_dataset_attr("group_array"))
            merged_indices += [i + cumulative_size for i in indices]
            cumulative_size += len(data)

        if type(data) == np.ndarray:
            merged_data = np.concatenate(merged_data, axis=0)
        if merged_group_arrays is not None:
            merged_group_arrays = np.concatenate(merged_group_arrays, axis=0)
        return merged_data, merged_targets, merged_indices, merged_group_arrays

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class ImageFolderDataset(torch.utils.data.Dataset):
    EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

        Support conventional image formats when reading local images: ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        # prepare info
        self.transform = transform
        self.target_transform = target_transform
        self.loader = datasets.folder.default_loader

        # setup of data and targets
        self.classes, self.class_to_index = self._find_classes(root)
        self.data, self.targets = self._make_dataset(
            root=root,
            class_to_idx=self.class_to_index,
            is_allowed_file=self._has_file_allowed_extension,
        )
        self.data_size = len(self.data)
        self.indices = list([x for x in range(0, self.data_size)])

        self.label_statistics = self._count_label_statistics(labels=self.targets)
        # print label statistics---------------------------------------------------------
        # for (i, v) in self.label_statistics.items():
        #     print(f"category={i}: {v}.\n")

    def __getitem__(self, index):
        data_idx = self.indices[index]
        img_path = self.data[data_idx]
        img = self.loader(img_path)
        target = self.targets[data_idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _find_classes(root) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.
        Ensures no class is a subdirectory of another.

        Args:
            root (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        """
        classes = [cls.name for cls in os.scandir(root) if cls.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _has_file_allowed_extension(self, filename: str) -> bool:
        """Checks if a file is an allowed extension."""
        return filename.lower().endswith(self.EXTENSIONS)

    @staticmethod
    def _make_dataset(
        root: str,
        class_to_idx: Dict[str, int],
        is_allowed_file: Callable[[str], bool],
    ) -> Tuple[List[str], List[int]]:
        imgs = []
        labels = []
        root = os.path.expanduser(root)

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for dir, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(dir, fname)
                    if is_allowed_file(path):
                        imgs.append(path)
                        labels.append(class_index)
                    else:
                        raise NotImplementedError(
                            f"The extension = {fname.split('.')[-1]} is not supported yet."
                        )

        return imgs, labels

    def _count_label_statistics(self, labels: list) -> Dict[str, int]:
        """
        This function returns the statistics of label category.
        """
        label_statistics = {}

        if self.class_to_index is not None:
            for k, v in sorted(self.class_to_index.items(), key=lambda item: item[1]):
                num_occurrence = labels.count(v)
                label_statistics[k] = num_occurrence
        else:
            # get the number of categories.
            num_categories = len(set(labels))
            for i in range(num_categories):
                num_occurrence = labels.count(i)
                label_statistics[str(i)] = num_occurrence

        return label_statistics

    def trim_dataset(self, data_size: int, random_seed: int = None) -> None:
        """trim dataset in a random manner given a data size"""
        assert data_size <= len(
            self
        ), "given data size should be smaller than the original data size."
        rng = np.random.default_rng(random_seed)
        indices_to_keep = rng.choice(len(self), size=data_size, replace=False)
        self.indices = self.indices[indices_to_keep]
        self.data_size = len(self.indices)


class ImageArrayDataset(ImageFolderDataset):
    """
    ImageArrayDataset supports dataset downloaded from torchvision library, and all other datasets
    that have processed raw images into image arrays such as CIFAR10-C and CIFAR10.1.
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: List[int],
        classes: Optional[List[str]] = None,
        class_to_index: Optional[dict] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data = data
        self.targets = targets
        self.data_size = len(self.data)
        self.indices = list([x for x in range(0, self.data_size)])
        if classes is not None:
            assert class_to_index is not None, "class_to_index needs to be specified "
            self.classes = classes
            self.class_to_index = class_to_index

        self.transform = transform
        self.target_transform = target_transform
        self.loader = datasets.folder.default_loader

        self.label_statistics = self._count_label_statistics(labels=self.targets)
        # print label statistics---------------------------------------------------------
        # for (i, v) in self.label_statistics.items():
        #     print(f"category={i}: {v}.\n")

    def __getitem__(self, index):
        data_idx = self.indices[index]
        img = self.data[data_idx]
        # data in some datasets is in the form of string path.
        if type(img) == str:
            img = np.array(self.loader(img))
        img = Image.fromarray(img)
        target = self.targets[data_idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class ConfounderDataset(ImageFolderDataset):
    """This class is designed for datasets such as waterbirds that need to consider confounders."""

    def __init__(
        self,
        root: str,
        data: Optional[np.ndarray],
        filename_array: Optional[np.ndarray],
        targets: List[int],
        group_array: np.ndarray,
        classes: Optional[list] = None,
        class_to_index: Optional[dict] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.loader = datasets.folder.default_loader
        assert (data is not None) or (
            filename_array is not None
        ), "either data or filename_array should be specified."
        if data is None:
            self.data = self._make_dataset(
                root=root,
                filename_array=filename_array,
                is_allowed_file=self._has_file_allowed_extension,
            )
        else:
            self.data = data
        self.targets = targets
        self.group_array = group_array
        self.data_size = len(self.data)
        self.indices = list([x for x in range(0, self.data_size)])
        self.classes = classes
        self.n_classes = len(self.classes)
        self.class_to_index = class_to_index

        self.transform = transform
        self.target_transform = target_transform

        self.label_statistics = self._count_label_statistics(labels=self.targets)
        # print label statistics---------------------------------------------------------
        # for (i, v) in self.label_statistics.items():
        #     print(f"category={i}: {v}.\n")

    def __getitem__(self, index):
        data_idx = self.indices[index]
        img = self.data[data_idx]
        if isinstance(img, str):
            img = np.array(self.loader(img))
        img = Image.fromarray(img)
        target = self.targets[data_idx]
        group = self.group_array[data_idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, group

    @staticmethod
    def _make_dataset(
        root: str,
        filename_array: np.ndarray,
        is_allowed_file: Callable[[str], bool],
    ) -> List[str]:
        imgs = []
        root = os.path.expanduser(root)

        for i in range(len(filename_array)):
            img_filename = filename_array[i]
            if is_allowed_file(img_filename):
                abs_imgpath = os.path.join(root, img_filename)
                imgs.append(abs_imgpath)
            else:
                raise NotImplementedError(
                    f"The extension = {img_filename.split('.')[-1]} is not supported yet."
                )

        return imgs


class SubDataset(torch.utils.data.Dataset):
    """
    It aims to support the split of the original dataset into training and test datasets
    that may encounter in pretraining. This dataset class is designed for datasets like
    Waterbirds, OfficeHome, and PACS that do not have available training and test data.
    """

    def __init__(
        self,
        data: List[str],
        targets: List[int],
        indices: List[int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data = data
        self.targets = targets
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform
        self.loader = datasets.folder.default_loader

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        img_path = self.data[data_idx]
        img = self.loader(img_path)
        target = self.targets[self.indices[idx]]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.indices)
