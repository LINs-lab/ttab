# -*- coding: utf-8 -*-
import copy
import functools
import os
import random
from typing import Callable, List, Tuple

import numpy as np
import torch

import ttab.loads.datasets.loaders as loaders
from ttab.api import PyTorchDataset
from ttab.loads.datasets.cifar import CIFAR10_1, CIFARSyntheticShift, LabelShiftedCIFAR
from ttab.loads.datasets.dataset_sampling import DatasetSampling
from ttab.loads.datasets.dataset_shifts import (
    NaturalShiftedData,
    NoShiftedData,
    NoShiftProperty,
    SyntheticShiftedData,
)
from ttab.loads.datasets.datasets import (
    CIFARDataset,
    ColoredMNIST,
    ImageFolderDataset,
    ImageNetDataset,
    MergeMultiDataset,
    OfficeHomeDataset,
    PACSDataset,
    WBirdsDataset,
    YearBookDataset,
)
from ttab.loads.datasets.imagenet import ImageNetSyntheticShift, ImageNetValNaturalShift
from ttab.loads.datasets.mnist import ColoredSyntheticShift
from ttab.scenarios import (
    CrossMixture,
    HeterogeneousNoMixture,
    HomogeneousNoMixture,
    InOutMixture,
    Scenario,
    TestCase,
    TestDomain,
)


class MergeMultiTestDatasets(object):
    @staticmethod
    def _intra_shuffle_dataset(
        dataset: PyTorchDataset, random_seed: int = None
    ) -> PyTorchDataset:
        """shuffle the dataset."""
        dataset.replace_indices(
            indices_pattern="random_shuffle", random_seed=random_seed
        )
        return dataset

    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
    @staticmethod
    def _intra_non_iid_shift(
        dataset: PyTorchDataset,
        non_iid_pattern: str = "class_wise_over_domain",
        non_iid_ness: float = 0.1,
        random_seed: int = None,
    ) -> PyTorchDataset:
        """make iid dataset non-iid through applying dirichlet distribution."""
        indices = dataset.query_dataset_attr("indices")
        targets = dataset.query_dataset_attr(
            "targets"
        )  # targets are always the same, which is the original label list.
        targets = [
            targets[i] for i in indices
        ]  # use indices to get the targets of interest.
        new_indices = []

        if non_iid_pattern == "class_wise_over_domain":
            dirichlet_numchunks = dataset.num_classes
            min_size = -1
            N = len(dataset)
            min_size_threshold = 5  # hyperparameter.
            while (
                min_size < min_size_threshold
            ):  # prevent any chunk having too less data
                idx_batch = [[] for _ in range(dirichlet_numchunks)]
                idx_batch_cls = [
                    [] for _ in range(dirichlet_numchunks)
                ]  # contains data per each class
                for k in range(dataset.num_classes):
                    targets_np = torch.Tensor(targets).numpy()
                    idx_k = np.where(targets_np == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(
                        np.repeat(non_iid_ness, dirichlet_numchunks)
                    )

                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < N / dirichlet_numchunks)
                            for p, idx_j in zip(proportions, idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                    ]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

                    # store class-wise data
                    for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                        idx_j.append(idx)

            sequence_stats = []
            # create temporally correlated toy dataset by shuffling classes
            for chunk in idx_batch_cls:
                cls_seq = list(range(dataset.num_classes))
                np.random.shuffle(cls_seq)
                for cls in cls_seq:
                    idx = chunk[cls]
                    new_indices.extend(idx)
                    sequence_stats.extend(list(np.repeat(cls, len(idx))))

            num_samples = len(new_indices)
            new_indices = new_indices[:num_samples]
            dataset.replace_indices(indices_pattern="new", new_indices=new_indices)
        return dataset

    def _merge_datasets(self, datasets: List[PyTorchDataset]) -> PyTorchDataset:
        return MergeMultiDataset(datasets=datasets)

    def _merge_two_datasets(
        self,
        left_dataset: PyTorchDataset,
        right_dataset: PyTorchDataset,
        ratio: float,
        random_seed: int = None,
    ) -> PyTorchDataset:
        random.seed(random_seed)
        left_dataset = DatasetSampling.uniform_sample(left_dataset, ratio, random_seed)
        right_dataset = DatasetSampling.uniform_sample(
            right_dataset, ratio, random_seed
        )
        return self._merge_datasets([left_dataset, right_dataset])

    def merge(
        self,
        test_case: TestCase,
        test_datasets: List[PyTorchDataset],
        src_dataset: PyTorchDataset = None,
        random_seed: int = None,
    ) -> PyTorchDataset:
        if isinstance(test_case.inter_domain, HomogeneousNoMixture):
            return self._merge_datasets(
                [
                    self._intra_shuffle_dataset(test_dataset, random_seed)
                    if test_case.intra_domain_shuffle
                    else test_dataset
                    for test_dataset in test_datasets
                ]
            )
        elif isinstance(test_case.inter_domain, HeterogeneousNoMixture):
            return self._merge_datasets(
                [
                    self._intra_non_iid_shift(
                        dataset=dataset,
                        non_iid_pattern=test_case.inter_domain.non_iid_pattern,
                        non_iid_ness=test_case.inter_domain.non_iid_ness,
                        random_seed=random_seed,
                    )
                    for dataset in test_datasets
                ]
            )
        elif isinstance(test_case.inter_domain, InOutMixture):
            if isinstance(src_dataset, WBirdsDataset):
                raise ValueError(
                    "WBirdsDataset does not support InOutMixture since it does not have an available ID test set."
                )
            return self._merge_datasets(
                [
                    self._intra_shuffle_dataset(
                        self._merge_two_datasets(
                            src_dataset,
                            test_dataset,
                            ratio=test_case.inter_domain.ratio,
                            random_seed=random_seed,
                        ),
                        random_seed=random_seed,
                    )
                    for test_dataset in test_datasets
                ]
            )
        elif isinstance(test_case.inter_domain, CrossMixture):
            return self._intra_shuffle_dataset(
                self._merge_datasets(test_datasets), random_seed=random_seed
            )


class ConstructTestDataset(object):
    def __init__(self, config):
        self.meta_conf = config
        self.data_path = self.meta_conf.data_path
        self.base_data_name = self.meta_conf.base_data_name
        self.seed = self.meta_conf.seed
        self.device = self.meta_conf.device
        self.input_size = (
            int(self.meta_conf.model_name.split("_")[-1])
            if "vit" in self.meta_conf.model_name
            else 32
        )

    def get_test_datasets(
        self, test_domains: List[TestDomain], data_augment: bool = False
    ) -> List[PyTorchDataset]:
        """This function defines the target domain dataset(s) for evaluation."""
        helper_fn = self._get_target_domain_helper()
        return [
            DatasetSampling(test_domain).sample(helper_fn(test_domain, data_augment))
            for test_domain in test_domains
        ]

    @staticmethod
    def get_src_domain(scenario: Scenario) -> TestDomain:
        """
        Create a TestDomain object for the source domain

        Attention: use the same sampling parameters as the target domain
        """
        # Select the first test domain as a base domain
        test_domain = copy.deepcopy(scenario.test_domains[0])
        data_name = scenario.src_data_name
        shift_type = test_domain.shift_type
        shift_property = test_domain.shift_property
        if any([x in test_domain.base_data_name for x in ["cifar", "imagenet"]]):
            shift_type = "no_shift"
            shift_property = NoShiftProperty(has_shift=False)

        return TestDomain(
            base_data_name=test_domain.base_data_name,
            data_name=data_name,
            shift_type=shift_type,
            shift_property=shift_property,
        )

    def get_src_dataset(
        self, scenario: Scenario, data_augment: bool = False
    ) -> PyTorchDataset:
        # get test_domain
        split = "test"
        if scenario.src_data_name in ["imagenet", "coloredmnist"]:
            split = "val"
        elif scenario.src_data_name == "waterbirds":
            # waterbirds has no availble ID test set
            split = "train"
        src_domain = self.get_src_domain(scenario)
        helper_fn = self._get_target_domain_helper()
        return DatasetSampling(src_domain).sample(
            helper_fn(src_domain, data_augment, split=split)
        )

    def _get_target_domain_helper(self) -> Callable[[TestDomain, bool], PyTorchDataset]:
        if "cifar" in self.base_data_name:
            # CIFAR datasets support [no_shift, natural, synthetic].
            helper_fn = self._get_cifar_test_domain_datasets_helper
        elif "imagenet" in self.base_data_name:
            # ImageNet datasets support [no_shift, natural, synthetic].
            helper_fn = self._get_imagenet_test_domain_datasets_helper
        elif "officehome" in self.base_data_name:
            # OfficeHome dataset only supports [natural].
            helper_fn = self._get_officehome_test_domain_datasets_helper
        elif "pacs" in self.base_data_name:
            # OfficeHome dataset only supports [natural].
            helper_fn = self._get_pacs_test_domain_datasets_helper
        elif "mnist" in self.base_data_name:
            # This benchmark only supports ColoredMNIST now, which is synthetic shift.
            helper_fn = self._get_mnist_test_domain_datasets_helper
        elif "waterbirds" in self.base_data_name:
            helper_fn = self._get_waterbirds_test_domain_datasets_helper
        elif "yearbook" in self.base_data_name:
            helper_fn = self._get_yearbook_test_domain_datasets_helper
        else:
            raise NotImplementedError(
                f"invalid base_data_name={self.base_data_name} for test domain."
            )
        return helper_fn

    def _get_cifar_test_domain_datasets_helper(
        self, test_domain: TestDomain, data_augment: bool = False, split: str = "test"
    ) -> PyTorchDataset:
        # get data_shift_class
        if test_domain.shift_type == "no_shift":
            data_shift_class = functools.partial(
                NoShiftedData, data_name=test_domain.base_data_name
            )
        elif test_domain.shift_type == "natural":
            if "shiftedlabel" in test_domain.data_name:
                data_shift_class = functools.partial(
                    NaturalShiftedData,
                    data_name=test_domain.data_name,
                    new_data=LabelShiftedCIFAR(
                        root=os.path.join(self.data_path, test_domain.base_data_name),
                        data_name=test_domain.data_name,
                        shift_type=test_domain.shift_property.version,
                        train=False,
                        param=self.meta_conf.label_shift_param,
                        data_size=self.meta_conf.data_size,
                        random_seed=self.seed,
                    ),
                )
            elif "cifar10_1" in test_domain.data_name:
                data_shift_class = functools.partial(
                    NaturalShiftedData,
                    data_name=test_domain.data_name,
                    new_data=CIFAR10_1(
                        root=os.path.join(self.data_path, test_domain.data_name),
                        data_name=test_domain.data_name,
                        version=test_domain.shift_property.version,
                    ),
                )
            else:
                raise NotImplementedError(f"invalid data_name={test_domain.data_name}.")
        elif test_domain.shift_type == "synthetic":
            assert 1 <= test_domain.shift_property.shift_degree <= 5

            data_shift_class = functools.partial(
                SyntheticShiftedData,
                data_name=test_domain.data_name,
                seed=self.seed,
                synthetic_class=functools.partial(
                    CIFARSyntheticShift,
                    corruption_type=test_domain.shift_property.shift_name,
                ),
                version=test_domain.shift_property.version,
                severity=test_domain.shift_property.shift_degree,
            )
        else:
            raise NotImplementedError(
                f"invalid shift type={test_domain.shift_type} for cifar datasets"
            )

        # create dataset.
        dataset = CIFARDataset(
            root=os.path.join(self.data_path, self.base_data_name),
            data_name=test_domain.data_name,
            split=split,
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
            input_size=self.input_size,
            data_size=self.meta_conf.data_size,
        )
        return dataset

    def _get_imagenet_test_domain_datasets_helper(
        self, test_domain: TestDomain, data_augment: bool = False, split: str = "test"
    ) -> PyTorchDataset:
        if test_domain.shift_type == "no_shift":
            data_shift_class = functools.partial(
                NoShiftedData, data_name=test_domain.base_data_name
            )
        elif test_domain.shift_type == "synthetic":
            assert 1 <= test_domain.shift_property.shift_degree <= 5

            data_shift_class = functools.partial(
                SyntheticShiftedData,
                data_name=test_domain.data_name,
                seed=self.seed,
                synthetic_class=functools.partial(
                    ImageNetSyntheticShift,
                    corruption_type=test_domain.shift_property.shift_name,
                ),
                version=test_domain.shift_property.version,
                severity=test_domain.shift_property.shift_degree,
            )
        elif test_domain.shift_type == "natural":
            assert test_domain.data_name in [
                "imagenet_a",
                "imagenet_r",
                "imagenet_v2_matched-frequency",
                "imagenet_v2_threshold0.7",
                "imagenet_v2_topimages",
            ]
            data_shift_class = functools.partial(
                NaturalShiftedData,
                data_name=test_domain.data_name,
                new_data=ImageNetValNaturalShift(
                    root=os.path.join(self.data_path, test_domain.base_data_name),
                    data_name=test_domain.data_name,
                    version=test_domain.shift_property.version,
                ),
            )
        else:
            raise NotImplementedError(
                f"invalid shift type={test_domain.shift_type} for ImageNet dataset."
            )

        dataset = ImageNetDataset(
            root=os.path.join(self.data_path, "ILSVRC"),
            data_name=test_domain.data_name,
            split=split,
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
        )
        return dataset

    def _get_officehome_test_domain_datasets_helper(
        self, test_domain: TestDomain, data_augment: bool = False, split: str = "test"
    ) -> PyTorchDataset:
        if test_domain.shift_type != "natural":
            raise NotImplementedError(
                f"invalid shift type={test_domain.shift_type} for officehome dataset."
            )

        _data_names = test_domain.data_name.split("_")  # e.g., "officehome_art"
        domain_path = os.path.join(self.data_path, self.base_data_name, _data_names[1])
        data_shift_class = functools.partial(
            NaturalShiftedData,
            data_name=test_domain.data_name,
            new_data=ImageFolderDataset(root=domain_path),
        )

        src_domain_path = os.path.join(
            self.data_path,
            self.base_data_name,
            self.meta_conf.src_data_name.split("_")[1],
        )
        dataset = OfficeHomeDataset(
            root=src_domain_path,  # replace with target domain data later
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
            data_size=self.meta_conf.data_size,
        )
        return dataset

    def _get_pacs_test_domain_datasets_helper(
        self, test_domain: TestDomain, data_augment: bool = False, split: str = "test"
    ) -> PyTorchDataset:
        if test_domain.shift_type != "natural":
            raise NotImplementedError(
                f"invalid shift type={test_domain.shift_type} for pacs dataset."
            )

        _data_names = test_domain.data_name.split("_")  # e.g., "pacs_art"
        domain_path = os.path.join(self.data_path, self.base_data_name, _data_names[1])
        data_shift_class = functools.partial(
            NaturalShiftedData,
            data_name=test_domain.data_name,
            new_data=ImageFolderDataset(root=domain_path),
        )

        src_domain_path = os.path.join(
            self.data_path,
            self.base_data_name,
            self.meta_conf.src_data_name.split("_")[1],
        )
        dataset = PACSDataset(
            root=src_domain_path,
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
            data_size=self.meta_conf.data_size,
        )
        return dataset

    def _get_mnist_test_domain_datasets_helper(
        self, test_domain: TestDomain, data_augment: bool = False, split: str = "test"
    ) -> PyTorchDataset:
        # get data_shift_class
        if test_domain.shift_type == "no_shift":
            data_shift_class = functools.partial(
                NoShiftedData, data_name=test_domain.base_data_name
            )
        elif test_domain.shift_type == "synthetic":
            data_shift_class = functools.partial(
                SyntheticShiftedData,
                data_name=test_domain.data_name,
                seed=self.seed,
                synthetic_class=ColoredSyntheticShift(
                    data_name=test_domain.data_name, seed=self.seed
                ),
                version=test_domain.shift_property.version,
            )
        else:
            raise NotImplementedError(
                f"invalid shift type={test_domain.shift_type} for mnist dataset."
            )

        dataset = ColoredMNIST(
            root=os.path.join(self.data_path, "mnist"),
            data_name=test_domain.data_name,
            split=split,
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
        )

        return dataset

    def _get_waterbirds_test_domain_datasets_helper(
        self, test_domain: TestDomain, data_augment: bool = False, split: str = "test"
    ) -> PyTorchDataset:
        if test_domain.shift_type != "natural":
            raise NotImplementedError(
                f"invalid shift type={test_domain.shift_type} for waterbirds dataset."
            )

        domain_path = os.path.join(self.data_path, self.base_data_name)
        data_shift_class = functools.partial(
            NaturalShiftedData,
            data_name=test_domain.data_name,
            new_data=WBirdsDataset(
                root=domain_path,
                split=split,
                device=self.device,
            ).dataset,
        )

        dataset = WBirdsDataset(
            root=domain_path,
            split=split,
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
        )
        return dataset
    
    def _get_yearbook_test_domain_datasets_helper(
        self, test_domain: TestDomain, data_augment: bool = False, split: str = "test"
    ) -> PyTorchDataset:
        if test_domain.shift_type != "natural":
            raise NotImplementedError(
                f"invalid shift type={test_domain.shift_type} for yearbook dataset."
            )

        domain_path = os.path.join(self.data_path, self.base_data_name)
        data_shift_class = functools.partial(
            NaturalShiftedData,
            data_name=test_domain.data_name,
            new_data=YearBookDataset(
                root=domain_path,
                split=split,
                device=self.device,
            ).dataset,
        )

        dataset = YearBookDataset(
            root=domain_path,
            split=split,
            device=self.device,
            data_shift_class=data_shift_class,
        )
        return dataset

    def construct_test_dataset(
        self, scenario: Scenario, data_augment: bool = False
    ) -> PyTorchDataset:
        return MergeMultiTestDatasets().merge(
            test_case=scenario.test_case,
            test_datasets=self.get_test_datasets(scenario.test_domains, data_augment),
            src_dataset=self.get_src_dataset(scenario, data_augment),
            random_seed=self.seed,
        )

    def construct_test_loader(self, scenario: Scenario) -> loaders.BaseLoader:
        test_dataset = self.construct_test_dataset(scenario, data_augment=False)
        test_dataloader: loaders.BaseLoader = loaders.get_test_loader(
            dataset=test_dataset, device=self.device
        )
        return test_dataloader


class ConstructAuxiliaryDataset(ConstructTestDataset):
    def __init__(self, config):
        super(ConstructAuxiliaryDataset, self).__init__(config)

    def construct_auxiliary_loader(
        self, scenario: Scenario, data_augment: bool = False
    ) -> loaders.BaseLoader:
        auxiliary_dataset = self.construct_test_dataset(scenario, data_augment)
        auxiliary_dataloader: loaders.BaseLoader = loaders.get_auxiliary_loader(
            dataset=auxiliary_dataset, device=self.device
        )
        return auxiliary_dataloader

    def construct_src_dataset(
        self, scenario: Scenario, data_size: int, data_augment: bool = False
    ) -> Tuple[PyTorchDataset, loaders.BaseLoader]:
        """Load a dataset that has the same distribution as the source domain."""
        split = "test"
        if scenario.src_data_name in ["imagenet", "coloredmnist"]:
            split = "val"
        elif scenario.src_data_name == "waterbirds":
            # waterbirds has no availble ID test set
            split = "train"
        src_domain = self.get_src_domain(scenario)
        src_dataset = self._get_target_domain_helper()(
            src_domain, data_augment, split=split
        )
        data_ratio = min(data_size / len(src_dataset), 1.0)
        src_dataset = DatasetSampling.uniform_sample(src_dataset, data_ratio, self.seed)

        src_dataloader = loaders.BaseLoader = loaders.get_auxiliary_loader(
            dataset=src_dataset, device=self.device
        )
        return src_dataset, src_dataloader
