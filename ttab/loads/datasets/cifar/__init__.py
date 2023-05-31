# -*- coding: utf-8 -*-
import os

import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets.utils import download_url
from ttab.loads.datasets.cifar.synthetic import (
    brightness,
    contrast,
    defocus_blur,
    elastic_transform,
    fog,
    frost,
    gaussian_noise,
    glass_blur,
    impulse_noise,
    jpeg_compression,
    motion_blur,
    pixelate,
    shot_noise,
    snow,
    zoom_blur,
)


class CIFAR10_1(object):
    """
    Borrowed from https://github.com/modestyachts/CIFAR-10.1

    Naming convention:
        cifar10_1/cifar10_1_v4/cifar10_1_v6
    """

    stats = {
        "v4": {
            "data": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_data.npy",
            "labels": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_labels.npy",
        },
        "v6": {
            "data": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy",
            "labels": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy",
        },
    }

    def __init__(self, root: str, data_name: str, version: str):
        version = "v4" if version is None else version
        if version not in ["v4", "v6"]:
            raise ValueError(f"version must be in ['v4', 'v6'], but got {version}")

        self.data_name = data_name
        self.path_data = os.path.join(root, f"cifar10.1_{version}_data.npy")
        self.path_labels = os.path.join(root, f"cifar10.1_{version}_labels.npy")
        self._download(root, version)

        self.data = np.load(self.path_data)
        self.targets = np.load(self.path_labels).tolist()
        self.indices = list(range(len(self.data)))
        self.data_size = len(self.indices)

    def _download(self, root: str, version: str) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(url=self.stats[version]["data"], root=root)
        download_url(url=self.stats[version]["labels"], root=root)

    def _check_integrity(self) -> bool:
        if os.path.exists(self.path_data) and os.path.exists(self.path_labels):
            return True
        else:
            return False

    def __getitem__(self, index):
        data_idx = self.indices[index]
        img_array = self.data[data_idx]
        target = self.targets[data_idx]
        return img_array, target

    def __len__(self):
        return len(self.indices)


# def np_to_png(a, fmt="png", scale=1):
#     a = np.uint8(a)
#     f = io.BytesIO()
#     tmp_img = PILImage.fromarray(a)
#     tmp_img = tmp_img.resize((scale * 32, scale * 32), PILImage.NEAREST)
#     tmp_img.save(f, fmt)
#     return f.getvalue()


"""Some corruptions are referred to https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py"""


class CIFARSyntheticShift(object):
    """
    The class of synthetic corruptions/shifts introduced in CIFAR_C.

    It applys a type of corruption to the input clean image given a random state.
    Generated images are different from the publicly accessible CIFAR_C dataset for evaluation.
    We also provide the publicly accessible CIFAR_C dataset for evaluation in the paper.
    """

    def __init__(
        self, data_name: str, seed: int, severity: int = 5, corruption_type: str = None
    ):
        if "cifar10" not in data_name:
            raise ValueError(f"data_name must have 'cifar', but got {data_name}")

        self.data_name = data_name
        self.base_data_name = data_name.split("_")[0]
        self.seed = seed
        self.severity = severity
        self.corruption_type = corruption_type
        self.dict_corruption = {
            "gaussian_noise": gaussian_noise,
            "shot_noise": shot_noise,
            "impulse_noise": impulse_noise,
            "defocus_blur": defocus_blur,
            "glass_blur": glass_blur,
            "motion_blur": motion_blur,
            "zoom_blur": zoom_blur,
            "snow": snow,
            "frost": frost,
            "fog": fog,
            "brightness": brightness,
            "contrast": contrast,
            "elastic_transform": elastic_transform,
            "pixelate": pixelate,
            "jpeg_compression": jpeg_compression,
        }
        if corruption_type is not None:
            assert (
                corruption_type in self.dict_corruption.keys()
            ), f"{corruption_type} is out of range"
        self.random_state = np.random.RandomState(self.seed)

    def _apply_corruption(self, pil_img):
        if self.corruption_type is None:
            corruption = self.random_state.choice(self.dict_corruption.values())
        else:
            corruption = self.dict_corruption[self.corruption_type]

        return np.uint8(
            corruption(pil_img, random_state=self.random_state, severity=self.severity)
        )

    def apply(self, pil_img):
        return self._apply_corruption(pil_img)


"""Label shift implementations are referred to https://github.com/Angie-Liu/labelshift/blob/master/cifar10_for_labelshift.py"""


class LabelShiftedCIFAR(object):
    """
    CIFAR dataset with label shift.

    Type of shifts:
        1. uniform distribution.
        2. dirichlet shift.

    Naming conventions:
        cifar10_shiftedlabel_uniform_gaussian_noise_5,
        cifar10_shiftedlabel_constant-size-dirichlet_gaussian_noise_5,
    """

    def __init__(
        self,
        root: str,
        data_name: str,
        shift_type: str,
        train: bool = False,
        param: float = None,
        data_size: int = None,
        target_label: int = None,
        random_seed: int = None,
    ) -> None:
        self.data_name = data_name
        self.shift_type = shift_type
        self.target_label = target_label
        self.data_size = data_size
        self.seed = random_seed

        # init dataset.
        if "100" in data_name:
            dataset_fn = datasets.CIFAR100
            self.num_classes = 100
        elif "10" in data_name:
            dataset_fn = datasets.CIFAR10
            self.num_classes = 10
        else:
            raise NotImplementedError(f"invalid data_name={data_name}.")

        basic_conf = {
            "root": root,
            "train": train,
            "transform": None,
            "target_transform": None,
            "download": True,
        }
        dataset = dataset_fn(**basic_conf)
        raw_targets = dataset.targets

        _data_names = self.shift_type.split("_")
        if len(_data_names) == 1:
            raw_data = dataset.data  # no corruption
        else:
            _shift_name = "_".join(_data_names[1:-1])  # corruption type
            _shift_degree = _data_names[-1]
            raw_data = self._load_cifar_c(root, _shift_name, _shift_degree)

        # create label shift.
        rng = np.random.default_rng(self.seed)
        if "uniform" in self.shift_type:
            self.data, self.targets = self._apply_uniform_subset_shift(
                data=raw_data,
                targets=raw_targets,
                data_size=self.data_size
                if self.data_size is not None
                else len(raw_data),
                random_generator=rng,
            )
        else:
            label_shifter = self._get_label_shifter()
            self.data, self.targets = label_shifter(
                data=raw_data,
                targets=raw_targets,
                param=param,
                random_generator=rng,
            )

    def _get_label_shifter(self):
        if "constant-size-dirichlet" in self.shift_type:
            return self._apply_constant_size_dirichlet_shift
        elif "dirichlet" in self.shift_type:
            return self._apply_dirichlet_shift
        else:
            raise NotImplementedError(f"invalid shift_type={self.shift_type}.")

    def _apply_dirichlet_shift(self, data, targets, param, random_generator=None):
        """
        Simulate non-i.i.d. dataset using dirichlet distribution.

        The size of dataset is dependent on the random seed.
        Args:
            data: original data
            targets: labels of original data
            param: parameter of dirichlet distribution
            random_generator: to control the randomness
        Returns:
            shifted_data: simulated dataset with label shift
            shifted_targets: labels of simulated dataset
        """
        if param is None:
            param = 4

        alpha = np.ones(self.num_classes) * param
        prob = random_generator.dirichlet(alpha)
        params = prob
        # use the maximum prob to decide the total number of training samples
        target_label = np.argmax(params)

        indices_target = [i for i, x in enumerate(targets) if x == target_label]
        num_targets = len(indices_target)
        prob_max = np.amax(params)
        num_data = int(num_targets / prob_max)
        indices_data = []

        for i in range(self.num_classes):
            num_i = int(num_data * params[i])
            all_indices_i = [t for t, x in enumerate(targets) if x == i]
            shuffle_i = random_generator.permutation(len(all_indices_i))
            selected_indices_i = [all_indices_i[shuffle_i[i]] for i in range(num_i)]
            indices_data += selected_indices_i

        shifted_data = data[(indices_data,)]
        shifted_targets = [targets[ele] for ele in indices_data]
        return shifted_data, shifted_targets

    def _apply_uniform_subset_shift(
        self, data, targets, data_size, random_generator=None
    ):
        """
        Build a subset of the original dataset with uniform distribution.
        Args:
            data: original data
            targets: labels of original data
            data_size: size of the subset
            random_generator: to control the randomness
        Returns:
            shifted_data: simulated data
            shifted_targets: labels of simulated data
        """
        # uniform on all labels
        num_per_class = int(data_size / self.num_classes)
        indices_data = []

        for i in range(self.num_classes):
            all_indices_i = [t for t, x in enumerate(targets) if x == i]
            shuffle_i = random_generator.permutation(len(all_indices_i))
            selected_indices_i = [
                all_indices_i[shuffle_i[j]] for j in range(num_per_class)
            ]
            indices_data += selected_indices_i

        shifted_data = data[(indices_data,)]
        shifted_targets = [targets[ele] for ele in indices_data]
        return shifted_data, shifted_targets

    def _apply_constant_size_dirichlet_shift(
        self, data, targets, param, random_generator=None
    ):
        """
        Simulate non-i.i.d. dataset with a constant size using dirichlet distribution.

        Args:
            data: original data
            targets: labels of original data
            param: parameter of dirichlet distribution
            random_generator: to control the randomness
        Returns:
            shifted_data: simulated dataset with label shift
            shifted_targets: labels of simulated dataset
        """
        if param is None:
            param = 4

        alpha = np.ones(self.num_classes) * param
        prob = random_generator.dirichlet(alpha)
        params = prob
        # use the maximum prob to decide the total number of training samples
        target_label = np.argmax(params)

        indices_target = [i for i, x in enumerate(targets) if x == target_label]
        # constant size. Maximum number of samples for a single category.
        if "100" in self.data_name:
            num_targets = self.data_size
            maximum_num = 100
        elif "10" in self.data_name:
            num_targets = len(indices_target)
            maximum_num = 1000
        prob_sum = np.sum(params)
        indices_data = []

        for i in range(self.num_classes):
            num_i = int(num_targets * params[i] / prob_sum)
            assert num_i <= maximum_num, "do not have enough data."
            all_indices_i = [t for t, x in enumerate(targets) if x == i]
            shuffle_i = random_generator.permutation(len(all_indices_i))
            selected_indices_i = [all_indices_i[shuffle_i[j]] for j in range(num_i)]
            indices_data += selected_indices_i

        # randomly add more samples if the total number of samples is not enough
        num_samples_to_add = num_targets - len(indices_data)
        if num_samples_to_add > 0:
            left_indices = [
                indice
                for indice in list(range(len(targets)))
                if indice not in indices_data
            ]
            samples = random_generator.choice(
                left_indices, num_samples_to_add, replace=False
            ).tolist()
            indices_data += samples

        shifted_data = data[(indices_data,)]
        shifted_targets = [targets[ele] for ele in indices_data]
        return shifted_data, shifted_targets

    def _load_cifar_c(self, root, shift_name, shift_degree):
        """Use publicly accessible cifar_c data"""
        domain_path = os.path.join(root + "_c", shift_name + ".npy")

        if not os.path.exists(domain_path):
            # download data from website: https://zenodo.org/record/2535967#.YxS6D-wzY-R
            raise ValueError("Please download cifar_c data from the web.")

        data_raw = np.load(domain_path)
        data_raw = data_raw[(int(shift_degree) - 1) * 10000 : int(shift_degree) * 10000]
        return data_raw
