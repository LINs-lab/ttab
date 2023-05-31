# -*- coding: utf-8 -*-
import os
import numpy as np
import tarfile

from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder


class ImageNetValNaturalShift(object):
    """Borrowed from
    (1) https://github.com/hendrycks/imagenet-r/,
    (2) https://github.com/hendrycks/natural-adv-examples,
    (3) https://github.com/modestyachts/ImageNetV2.
    """

    stats = {
        "imagenet_r": {
            "data_and_labels": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
            "folder_name": "imagenet-r",
        },
        "imagenet_a": {
            "data_and_labels": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar",
            "folder_name": "imagenet-a",
        },
        "imagenet_v2_matched-frequency": {
            "data_and_labels": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz",
            "folder_name": "imagenetv2-matched-frequency-format-val",
        },
        "imagenet_v2_threshold0.7": {
            "data_and_labels": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-threshold0.7.tar.gz",
            "folder_name": "imagenetv2-threshold0.7-format-val",
        },
        "imagenet_v2_topimages": {
            "data_and_labels": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz",
            "folder_name": "imagenetv2-topimages-format-val",
        },
    }

    def __init__(self, root, data_name, version=None):
        self.data_name = data_name
        self.path_data_and_labels_tar = os.path.join(
            root, self.stats[data_name]["data_and_labels"].split("/")[-1]
        )
        self.path_data_and_labels = os.path.join(
            root, self.stats[data_name]["folder_name"]
        )

        self._download(root)

        self.image_folder = ImageFolder(self.path_data_and_labels)
        self.data = self.image_folder.samples
        self.targets = self.image_folder.targets

    def _download(self, root):
        download_url(url=self.stats[self.data_name]["data_and_labels"], root=root)

        if self._check_integrity():
            print("Files already downloaded, verified, and uncompressed.")
            return
        self._uncompress(root)

    def _uncompress(self, root):
        with tarfile.open(self.path_data_and_labels_tar) as file:
            file.extractall(root)

    def _check_integrity(self) -> bool:
        if os.path.exists(self.path_data_and_labels):
            return True
        else:
            return False

    def __getitem__(self, index):
        path, target = self.data[index]
        img = self.image_folder.loader(path)
        return img, target

    def __len__(self):
        return len(self.data)


"""Some corruptions are referred to https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py"""


class ImageNetSyntheticShift(object):
    """The class of synthetic corruptions/shifts introduced in ImageNet_C."""

    def __init__(
        self, data_name, seed, severity=5, corruption_type=None, img_resolution=224
    ):
        assert "imagenet" in data_name

        if img_resolution == 224:
            from ttab.loads.datasets.imagenet.synthetic_224 import (
                gaussian_noise,
                shot_noise,
                impulse_noise,
                defocus_blur,
                glass_blur,
                motion_blur,
                zoom_blur,
                snow,
                frost,
                fog,
                brightness,
                contrast,
                elastic_transform,
                pixelate,
                jpeg_compression,
                # for validation.
                speckle_noise,
                gaussian_blur,
                spatter,
                saturate,
            )
        elif img_resolution == 64:
            from ttab.loads.datasets.imagenet.synthetic_64 import (
                gaussian_noise,
                shot_noise,
                impulse_noise,
                defocus_blur,
                glass_blur,
                motion_blur,
                zoom_blur,
                snow,
                frost,
                fog,
                brightness,
                contrast,
                elastic_transform,
                pixelate,
                jpeg_compression,
                # for validation.
                speckle_noise,
                gaussian_blur,
                spatter,
                saturate,
            )
        else:
            raise NotImplementedError(
                f"Invalid img_resolution for ImageNet: {img_resolution}"
            )

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
            "speckle_noise": speckle_noise,
            "gaussian_blur": gaussian_blur,
            "spatter": spatter,
            "saturate": saturate,
        }
        if corruption_type is not None:
            assert (
                corruption_type in self.dict_corruption.keys()
            ), f"{corruption_type} is out of range"
        self.random_state = np.random.RandomState(self.seed)

    def _apply_corruption(self, pil_img):
        if self.corruption_index is None or self.corruption_type == "all":
            corruption = self.random_state.choice(self.dict_corruption.values())
        else:
            corruption = self.dict_corruption[self.corruption_type]

        return np.uint8(
            corruption(pil_img, random_state=self.random_state, severity=self.severity)
        )

    def apply(self, pil_img):
        return self._apply_corruption(pil_img)
