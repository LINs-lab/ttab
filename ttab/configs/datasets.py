# -*- coding: utf-8 -*-

# 1. This file collects hyperparameters significant for pretraining and test-time adaptation.
# 2. We are only concerned about dataset-related hyperparameters here, e.g., lr, dataset statistics, and type of corruptions.
# 3. We provide default hyperparameters if users have no idea how to set up reasonable values.

dataset_defaults = {
    "cifar10": {
        "statistics": {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
            "n_classes": 10,
        },
        "version": "deterministic",
        "img_shape": (32, 32, 3),
    },
    "cifar100": {
        "statistics": {
            "mean": (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            "std": (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
            "n_classes": 100,
        },
        "version": "deterministic",
        "img_shape": (32, 32, 3),
    },
    "officehome": {
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 65,
        },
        "img_shape": (224, 224, 3),
    },
    "pacs": {
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 7,
        },
        "img_shape": (224, 224, 3),
    },
    "coloredmnist": {
        "statistics": {
            "mean": (0.1307, 0.1307, 0.0),
            "std": (0.3081, 0.3081, 0.3081),
            "n_classes": 2,
        },
        "img_shape": (28, 28, 3),
    },
    "waterbirds": {
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 2,
        },
        "group_counts": [3498, 184, 56, 1057],  # used to compute group ratio.
        "img_shape": (224, 224, 3),
    },
    "imagenet": {
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 1000,
        },
        "img_shape": (224, 224, 3),
    },
    "yearbook": {
        "statistics": {"n_classes": 2,},
        "img_shape": (32, 32, 3),
    }
}
