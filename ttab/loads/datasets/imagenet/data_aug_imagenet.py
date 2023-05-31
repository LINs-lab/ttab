from typing import Callable, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from ttab.configs.datasets import dataset_defaults

## https://github.com/google-research/augmix


def _augmix_aug(x_orig: torch.Tensor, data_name: str) -> torch.Tensor:
    tensor_to_image, preprocess = get_ops(data_name)
    x_orig = tensor_to_image(x_orig.squeeze(0))
    preaugment = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
    )
    x_orig = preaugment(x_orig)
    x_processed = preprocess(x_orig)
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(augmentations)(x_aug)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


aug_imagenet = _augmix_aug


def autocontrast(pil_img, level=None):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, level=None):
    return ImageOps.equalize(pil_img)


def rotate(pil_img, level):
    degrees = int_parameter(rand_lvl(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)


def solarize(pil_img, level):
    level = int_parameter(rand_lvl(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (224, 224),
        Image.AFFINE,
        (1, level, 0, 0, 1, 0),
        resample=Image.BILINEAR,
        fillcolor=128,
    )


def shear_y(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (224, 224),
        Image.AFFINE,
        (1, 0, 0, level, 1, 0),
        resample=Image.BILINEAR,
        fillcolor=128,
    )


def translate_x(pil_img, level):
    level = int_parameter(rand_lvl(level), 224 / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (224, 224),
        Image.AFFINE,
        (1, 0, level, 0, 1, 0),
        resample=Image.BILINEAR,
        fillcolor=128,
    )


def translate_y(pil_img, level):
    level = int_parameter(rand_lvl(level), 224 / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (224, 224),
        Image.AFFINE,
        (1, 0, 0, 0, 1, level),
        resample=Image.BILINEAR,
        fillcolor=128,
    )


def posterize(pil_img, level):
    level = int_parameter(rand_lvl(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.0


def rand_lvl(n):
    return np.random.uniform(low=0.1, high=n)


augmentations = [
    autocontrast,
    equalize,
    lambda x: rotate(x, 1),
    lambda x: solarize(x, 1),
    lambda x: shear_x(x, 1),
    lambda x: shear_y(x, 1),
    lambda x: translate_x(x, 1),
    lambda x: translate_y(x, 1),
    lambda x: posterize(x, 1),
]

def get_ops(data_name: str) -> Tuple[Callable, Callable]:
    """Get the operations to be applied when defining transforms."""
    unnormalize = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0 / v for v in dataset_defaults[data_name]["statistics"]["std"]],
            ),
            transforms.Normalize(
                mean=[-v for v in dataset_defaults[data_name]["statistics"]["mean"]],
                std=[1.0, 1.0, 1.0],
            ),
        ]
    )

    tensor_to_image = transforms.Compose([unnormalize, transforms.ToPILImage()])
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                dataset_defaults[data_name]["statistics"]["mean"], dataset_defaults[data_name]["statistics"]["std"]
            ),
        ]
    )
    return tensor_to_image, preprocess


def tr_transforms_imagenet(image: torch.Tensor, data_name: str) -> torch.Tensor:
    """
    Data augmentation for input images.
    args:
    inputs:
        image: tensor [n_channel, H, W]
    outputs:
        augment_image: tensor [1, n_channel, H, W]
    """
    tensor_to_image, preprocess = get_ops(data_name)
    image = tensor_to_image(image)

    preaugment = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
    )
    augment_image = preaugment(image)
    augment_image = preprocess(augment_image)

    return augment_image
