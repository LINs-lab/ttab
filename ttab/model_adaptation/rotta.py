# -*- coding: utf-8 -*-
import copy
import functools
import math
import random
from copy import deepcopy
from typing import List

import PIL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, Compose, Lambda

from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_adaptation.utils import RobustBN1d, RobustBN2d
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class Rotta(BaseAdaptation):
    """Robust Test-time Adaptation,
    http://arxiv.org/abs/2303.13899
    https://github.com/BIT-DA/RoTTA
    """

    def __init__(self, meta_conf, model: nn.Module):
        super().__init__(meta_conf, model)
        self.mem = CSTU(
            capacity=self._meta_conf.memory_size,
            num_class=self._meta_conf.statistics["n_classes"],
            lambda_t=self._meta_conf.lambda_t,
            lambda_u=self._meta_conf.lambda_u,
        )
        self.model_ema = self.build_ema(self._model)
        self.transform = self.get_tta_transforms(self._meta_conf)
        self.nu = self._meta_conf.nu
        self.update_frequency = self._meta_conf.update_frequency
        self.current_instance = 0

    def _initialize_model(self, model: nn.Module):
        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(
                sub_module, nn.BatchNorm2d
            ):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = self.get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer, self._meta_conf.alpha)
            momentum_bn.requires_grad_(True)
            self.set_named_submodule(model, name, momentum_bn)
        return model.to(self._meta_conf.device)

    def one_adapt_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        timer: Timer,
        batch: Batch,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        # batch data
        with timer("forward"):
            with torch.no_grad():
                model.eval()
                self.model_ema.eval()
                ema_out = self.model_ema(batch._x)
                predict = torch.softmax(ema_out, dim=1)
                pseudo_label = torch.argmax(predict, dim=1)
                entropy = torch.sum(-predict * torch.log(predict + 1e-6), dim=1)

        loss, grads = None, None
        # add into memory
        for i, data in enumerate(batch._x):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                loss, grads = self.update_model(model, optimizer)

        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss,
            "grads": grads,
            "yhat": ema_out,
        }

    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        current_batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                timer,
                current_batch,
                random_seed,
            )
            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(model).state_dict(),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **adaptation_result,
                },
                current_batch=current_batch,
            )

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()
        # get memory data
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            strong_sup_aug = self.transform(sup_data)
            ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = model(strong_sup_aug)
            instance_weight = self.timeliness_reweighting(ages)
            l_sup = (self.softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        l = l_sup
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            optimizer.step()

        self.update_ema_variables(self.model_ema, self._model, self.nu)

        return l, grads

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        # evaluate the per batch pre-adapted performance. Different with no adaptation.
        if self._meta_conf.record_preadapted_perf:
            with timer("evaluate_preadapted_performance"):
                self._model.eval()
                with torch.no_grad():
                    yhat = self._model(current_batch._x)
                self._model.train()
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )

        # adaptation.
        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=self._optimizer,
                current_batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=self._meta_conf.seed,
            )

        # select the optimal checkpoint, and return the corresponding prediction.
        with timer("select_optimal_checkpoint"):
            optimal_state = model_selection_method.select_state()
            log(
                f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
            )

            self._model.load_state_dict(optimal_state["model"])
            model_selection_method.clean_up()

            if self._oracle_model_selection:
                # oracle model selection needs to save steps
                self.oracle_adaptation_steps.append(optimal_state["step"])
                # update optimizer.
                self._optimizer.load_state_dict(optimal_state["optimizer"])

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, optimal_state["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    optimal_state["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )

    @property
    def name(self):
        return "rotta"

    @staticmethod
    def build_ema(model):
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model
    
    @staticmethod
    def timeliness_reweighting(ages):
        if isinstance(ages, list):
            ages = torch.tensor(ages).float().cuda()
        return torch.exp(-ages) / (1 + torch.exp(-ages))
    
    @staticmethod
    @torch.jit.script
    def softmax_entropy(x, x_ema):
        return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

    def _initialize_trainable_parameters(self):
        names = []
        params = []

        for n, p in self._model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names
    
    @staticmethod
    def get_tta_transforms(cfg, gaussian_std: float = 0.005, soft=False):
        img_shape = (*cfg.img_shape, 3)
        n_pixels = img_shape[0]

        clip_min, clip_max = 0.0, 1.0

        p_hflip = 0.5

        tta_transforms = transforms.Compose(
            [
                Clip(0.0, 1.0),
                ColorJitterPro(
                    brightness=[0.8, 1.2] if soft else [0.6, 1.4],
                    contrast=[0.85, 1.15] if soft else [0.7, 1.3],
                    saturation=[0.75, 1.25] if soft else [0.5, 1.5],
                    hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
                    gamma=[0.85, 1.15] if soft else [0.7, 1.3],
                ),
                transforms.Pad(padding=int(n_pixels / 2), padding_mode="edge"),
                transforms.RandomAffine(
                    degrees=[-8, 8] if soft else [-15, 15],
                    translate=(1 / 16, 1 / 16),
                    scale=(0.95, 1.05) if soft else (0.9, 1.1),
                    shear=None,
                    interpolation=PIL.Image.BILINEAR,
                    fill=None,
                ),
                transforms.GaussianBlur(
                    kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]
                ),
                transforms.CenterCrop(size=n_pixels),
                transforms.RandomHorizontalFlip(p=p_hflip),
                GaussianNoise(0, gaussian_std),
                Clip(clip_min, clip_max),
            ]
        )
        return tta_transforms
    
    @staticmethod
    def get_named_submodule(model, sub_name: str):
        names = sub_name.split(".")
        module = model
        for name in names:
            module = getattr(module, name)

        return module

    @staticmethod
    def set_named_submodule(model, sub_name, value):
        names = sub_name.split(".")
        module = model
        for i in range(len(names)):
            if i != len(names) - 1:
                module = getattr(module, names[i])

            else:
                setattr(module, names[i], value)

# ========== Memory ==========
class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"


class CSTU:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u

        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls)
        return occupancy

    def per_class_dist(self):
        per_class_occupied = [0] * self.num_class
        for cls, class_list in enumerate(self.data):
            per_class_occupied[cls] = len(class_list)

        return per_class_occupied

    def add_instance(self, instance):
        assert len(instance) == 3
        x, prediction, uncertainty = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls, score):
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                majority_classes = self.get_majority_classes()
                return self.remove_from_classes(majority_classes, score)
        else:
            return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes: List[int], score_base):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                uncertainty = item.uncertainty
                age = item.age
                score = self.heuristic_score(age=age, uncertainty=uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls

        if max_class is not None:
            if max_score > score_base:
                self.data[max_class].pop(max_index)
                return True
            else:
                return False
        else:
            return True

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if occupied == max_occupied:
                classes.append(i)

        return classes

    def heuristic_score(self, age, uncertainty):
        return self.lambda_t * 1 / (
            1 + math.exp(-age / self.capacity)
        ) + self.lambda_u * uncertainty / math.log(self.num_class)

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return

    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age)

        tmp_age = [x / self.capacity for x in tmp_age]

        return tmp_data, tmp_age

# ========== custom_transforms ==========
class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Clip(torch.nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + "(min_val={0}, max_val={1})".format(
            self.min_val, self.max_val
        )


class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, "gamma")

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(lambda img: F.adjust_brightness(img, brightness_factor))
            )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor))
            )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(lambda img: F.adjust_saturation(img, saturation_factor))
            )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = (
                    torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                )
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = (
                    torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                )
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = (
                    torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                )
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(
                    1e-8, 1.0
                )  # to fix Nan values in gradients, which happens when applying gamma
                # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        format_string += ", gamma={0})".format(self.gamma)
        return format_string
