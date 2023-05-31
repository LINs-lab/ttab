# -*- coding: utf-8 -*-
import numpy as np
from ttab.loads.datasets.datasets import ImageArrayDataset

"""ColoredMNIST is borrowed from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py"""


class ColoredSyntheticShift(object):
    """The class of synthetic colored shifts introduced in ColoredMNIST."""

    def __init__(self, data_name, seed, color_flip_prob: float = 0.25) -> None:
        self.data_name = data_name
        self.base_data_name = data_name.split("_")[0]
        self.seed = seed
        self.color_flip_prob = color_flip_prob
        assert (
            0 <= self.color_flip_prob <= 1
        ), f"{self.color_flip_prob} is out of range."

    def _color_grayscale_arr(self, arr, red=True):
        """Converts grayscale image to either red or green"""
        assert arr.ndim == 2
        dtype = arr.dtype
        h, w = arr.shape
        arr = np.reshape(arr, [h, w, 1])
        if red:
            arr = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
        else:
            arr = np.concatenate(
                [
                    np.zeros((h, w, 1), dtype=dtype),
                    arr,
                    np.zeros((h, w, 1), dtype=dtype),
                ],
                axis=2,
            )
        return arr

    def apply(self, dataset, transform, target_transform):
        return self._apply_color(
            dataset, self.color_flip_prob, transform, target_transform
        )

    def _apply_color(self, dataset, color_flip_prob, transform, target_transform):

        train_data = []
        train_targets = []
        val_data = []
        val_targets = []
        test_data = []
        test_targets = []
        for idx, (im, label) in enumerate(dataset):
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with probability of `color_flip_prob`
            if np.random.uniform() < color_flip_prob:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            # Flip the color with a probability e that depends on the domain
            if idx < 30000:
                # 10% in the training environment
                if np.random.uniform() < 0.1:
                    color_red = not color_red
            elif idx < 40000:
                # 10% in the in-distribution eval environment
                # val set should have the same distribution as the source domain
                if np.random.uniform() < 0.1:
                    color_red = not color_red
            else:
                # 90% in the ood test environment
                if np.random.uniform() < 0.9:
                    color_red = not color_red

            colored_arr = self._color_grayscale_arr(im_array, red=color_red)

            if idx < 30000:
                train_data.append(colored_arr)
                train_targets.append(binary_label)
            elif idx < 40000:
                val_data.append(colored_arr)
                val_targets.append(binary_label)
            else:
                test_data.append(colored_arr)
                test_targets.append(binary_label)

        classes = ["0-4", "5-9"]
        class_to_index = {"0-4": 0, "5-9": 1}
        train_dataset = ImageArrayDataset(
            data=train_data,
            targets=train_targets,
            classes=classes,
            class_to_index=class_to_index,
            transform=transform,
            target_transform=target_transform,
        )
        val_dataset = ImageArrayDataset(
            data=val_data,
            targets=val_targets,
            classes=classes,
            class_to_index=class_to_index,
            transform=transform,
            target_transform=target_transform,
        )
        test_dataset = ImageArrayDataset(
            data=test_data,
            targets=test_targets,
            classes=classes,
            class_to_index=class_to_index,
            transform=transform,
            target_transform=target_transform,
        )

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
