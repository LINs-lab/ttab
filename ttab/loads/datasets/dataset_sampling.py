# -*- coding: utf-8 -*-
import random

from ttab.api import PyTorchDataset
from ttab.scenarios import TestDomain


class DatasetSampling(object):
    def __init__(self, test_domain: TestDomain):
        self.domain_sampling_name = test_domain.domain_sampling_name
        self.domain_sampling_value = test_domain.domain_sampling_value
        self.domain_sampling_ratio = test_domain.domain_sampling_ratio

    def sample(
        self, dataset: PyTorchDataset, random_seed: int = None
    ) -> PyTorchDataset:
        if self.domain_sampling_name == "uniform":
            return self.uniform_sample(
                dataset=dataset,
                ratio=self.domain_sampling_ratio,
                random_seed=random_seed,
            )
        else:
            raise NotImplementedError

    @staticmethod
    def uniform_sample(
        dataset: PyTorchDataset, ratio: float, random_seed: int = None
    ) -> PyTorchDataset:
        """This function uniformly samples data from the original dataset without replacement."""
        random.seed(random_seed)
        indices = dataset.query_dataset_attr("indices")
        sampled_list = random.sample(
            indices,
            int(ratio * len(indices)),
        )
        sampled_list.sort()
        dataset.replace_indices(
            indices_pattern="new", new_indices=sampled_list, random_seed=random_seed
        )
        return dataset
