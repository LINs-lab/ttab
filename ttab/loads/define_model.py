# -*- coding: utf-8 -*-
import os

import timm
import torch
import ttab.model_adaptation.utils as adaptation_utils
from torch import nn
from ttab.loads.models import WideResNet, cct_7_3x1_32, resnet
from ttab.loads.models.resnet import ResNetCifar, ResNetImagenet, ResNetMNIST


class SelfSupervisedModel(nn.Module):
    """
    This class is built for TTT.

    It adds an auxiliary head to the original model architecture.
    """

    def __init__(self, model, config):
        super(SelfSupervisedModel, self).__init__()
        self._config = config
        self.main_model = model
        self.ext, self.head = self._define_head()
        self.ssh = adaptation_utils.ExtractorHead(self.ext, self.head)

    def _define_resnet_head(self):
        assert hasattr(
            self._config, "entry_of_shared_layers"
        ), "Need to set up the number of shared layers as feature extractor."

        if isinstance(self.main_model, ResNetImagenet):
            if (
                self._config.entry_of_shared_layers == "layer4"
                or self._config.entry_of_shared_layers == None
            ):
                ext = adaptation_utils.shared_ext_from_layer4(self.main_model)
                head = adaptation_utils.head_from_classifier(
                    self.main_model, self._config.dim_out
                )
            elif self._config.entry_of_shared_layers == "layer3":
                ext = adaptation_utils.shared_ext_from_layer3(self.main_model)
                head = adaptation_utils.head_from_last_layer1(
                    self.main_model, self._config.dim_out
                )
            else:
                raise ValueError(
                    f"invalid configuration: entry_of_shared_layers={self._config.entry_of_shared_layers} for dataset={self._config.base_data_name}."
                )
        elif isinstance(self.main_model, (ResNetCifar, WideResNet)):
            if (
                self._config.entry_of_shared_layers == "layer3"
                or self._config.entry_of_shared_layers == None
            ):
                ext = adaptation_utils.shared_ext_from_layer3(self.main_model)
                head = adaptation_utils.head_from_classifier(
                    self.main_model, self._config.dim_out
                )
            elif self._config.entry_of_shared_layers == "layer2":
                ext = adaptation_utils.shared_ext_from_layer2(self.main_model)
                head = adaptation_utils.head_from_last_layer1(
                    self.main_model, self._config.dim_out
                )
            else:
                raise ValueError(
                    f"invalid configuration: entry_of_shared_layers={self._config.entry_of_shared_layers} for dataset={self._config.base_data_name}."
                )
        elif isinstance(self.main_model, ResNetMNIST):
            if (
                self._config.entry_of_shared_layers == "layer4"
                or self._config.entry_of_shared_layers == None
            ):
                ext = adaptation_utils.shared_ext_from_layer4(self.main_model)
                head = adaptation_utils.head_from_classifier(
                    self.main_model, self._config.dim_out
                )
            elif self._config.entry_of_shared_layers == "layer3":
                ext = adaptation_utils.shared_ext_from_layer3(self.main_model)
                head = adaptation_utils.head_from_last_layer1(
                    self.main_model, self._config.dim_out
                )
            else:
                raise ValueError(
                    f"invalid configuration: entry_of_shared_layers={self._config.entry_of_shared_layers} for dataset={self._config.base_data_name}."
                )
        return ext, head

    def _define_vit_head(self):
        ext = adaptation_utils.VitExtractor(self.main_model)
        head = nn.Linear(
            in_features=self.main_model.head.in_features,
            out_features=self._config.dim_out,
            bias=True,
        )
        return ext, head

    def _define_head(self):
        if "resnet" in self._config.model_name:
            return self._define_resnet_head()
        elif "vit" in self._config.model_name:
            return self._define_vit_head()

    def load_pretrained_parameters(self, ckpt_path):
        """This function helps to load pretrained parameters given the checkpoint path."""
        ckpt = torch.load(ckpt_path, map_location=self._config.device)
        self.main_model.load_state_dict(ckpt["model"])
        self.head.load_state_dict(ckpt["head"])


def define_model(config):
    # use public models and checkpoints and not adjust the model arch.
    if "imagenet" in config.data_names:
        if config.group_norm_num_groups is not None:
            assert config.model_name == "resnet50"
            return timm.create_model(config.model_name + "_gn", pretrained=True)
        return timm.create_model(config.model_name, pretrained=True)

    # use built-in models and local checkpoints.
    if "wideresnet" in config.model_name:
        components = config.model_name.split("_")
        depth = int(components[0].replace("wideresnet", ""))
        widen_factor = int(components[1])
        init_model = WideResNet(
            depth,
            widen_factor,
            config.statistics["n_classes"],
            split_point=config.entry_of_shared_layers,
            dropout_rate=0.0,
        )
    elif "resnet" in config.model_name:
        depth = int(config.model_name.replace("resnet", ""))
        init_model = resnet(
            config.base_data_name,
            depth,
            split_point=config.entry_of_shared_layers,
            group_norm_num_groups=config.group_norm_num_groups,
            grad_checkpoint=config.grad_checkpoint,
        )
    elif "vit" in config.model_name:
        init_model = timm.create_model(config.model_name, pretrained=False)
        init_model.head = nn.Linear(
            init_model.head.in_features, config.statistics["n_classes"]
        )
        if config.grad_checkpoint:
            init_model.set_grad_checkpointing()
    elif "cct" in config.model_name:
        return cct_7_3x1_32(pretrained=False)  # not support TTT yet.
    else:
        raise NotImplementedError(f"invalid model_name={config.model_name}.")

    if config.model_adaptation_method == "ttt":
        return SelfSupervisedModel(init_model, config)
    return init_model


def load_pretrained_model(config, model):
    # safety check
    assert os.path.exists(
        config.ckpt_path
    ), "The user-provided path for the checkpoint does not exist."

    # check IABN layers.
    # If not having IABN layers, skip the loading.
    if hasattr(config, "iabn") and config.iabn:
        iabn_flag = False
        for _, module in model.named_modules():
            if isinstance(
                module,
                (
                    adaptation_utils.InstanceAwareBatchNorm2d,
                    adaptation_utils.InstanceAwareBatchNorm1d,
                ),
            ):
                iabn_flag = True
        if not iabn_flag:
            return

    if "imagenet" in config.data_names:
        return

    # load parameters
    if isinstance(model, SelfSupervisedModel):
        model.load_pretrained_parameters(config.ckpt_path)
    else:
        ckpt = torch.load(config.ckpt_path, map_location=config.device)
        model.load_state_dict(ckpt["model"])  # ignore the auxiliary branch.
