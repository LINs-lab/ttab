# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import PIL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import ttab.loads.define_dataset as define_dataset
import ttab.model_adaptation.utils as adaptation_utils
from numpy import random
from torchvision.transforms import ColorJitter, Compose, Lambda
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class CoTTA(BaseAdaptation):
    """Continual Test-Time Domain Adaptation,
    https://arxiv.org/abs/2203.13591,
    https://github.com/qinenergy/cotta
    """

    def __init__(self, meta_conf, model: nn.Module):
        super(CoTTA, self).__init__(meta_conf, model)

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        model.train()
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # bn module always uses batch statistics, in both training and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else:
                m.requires_grad_(True)
        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """select target params for adaptation methods."""
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_modules():
            self._adapt_module_names.append(name_module)
            for name_parameter, parameter in module.named_parameters():
                if name_parameter in ["weight", "bias"]:
                    adapt_params.append(parameter)
                    adapt_param_names.append(f"{name_module}.{name_parameter}")

        assert (
            len(self._adapt_module_names) > 0
        ), "CoTTA needs some adaptable model parameters."
        return adapt_params, adapt_param_names

    def _post_safety_check(self):
        is_training = self._model.training
        assert is_training, "adaptation needs train mode: call model.train()."

        param_grads = [p.requires_grad for p in self._model.parameters()]
        has_any_params = any(param_grads)
        assert has_any_params, "adaptation needs some trainable params."

    def initialize(self, seed: int):
        """Initialize the algorithm."""
        if self._meta_conf.model_selection_method == "oracle_model_selection":
            self._oracle_model_selection = True
            self.oracle_adaptation_steps = []
        else:
            self._oracle_model_selection = False

        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
        self._base_model = copy.deepcopy(self._model)  # update base model
        params, names = self._initialize_trainable_parameters()
        self._optimizer = self._initialize_optimizer(params)
        self._base_optimizer = copy.deepcopy(self._optimizer)
        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(
            config=self._meta_conf
        )
        self.transform = self.get_aug_transforms(img_shape=self._meta_conf.img_shape)
        # compute fisher regularizer
        self.fishers = None
        self.ewc_optimizer = torch.optim.SGD(params, 0.001)

        # base model state.
        (
            self.model_state_dict,
            self.model_ema,
            self.model_anchor,
        ) = self.copy_model_states(self._model)

    @staticmethod
    def get_aug_transforms(
        img_shape: tuple, gaussian_std: float = 0.005, soft: bool = False
    ):
        """Get augmentation transforms used at test time."""
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
                    resample=PIL.Image.BILINEAR,
                    fillcolor=None,
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
    def update_ema_variables(ema_model, model, alpha_teacher):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (
                alpha_teacher * ema_param[:].data[:]
                + (1 - alpha_teacher) * param[:].data[:]
            )
        return ema_model

    @staticmethod
    def copy_model_states(model):
        """Copy the model states for resetting after adaptation."""
        model_state = copy.deepcopy(model.state_dict())
        model_anchor = copy.deepcopy(model)
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return model_state, ema_model, model_anchor

    @staticmethod
    def load_model_and_optimizer(
        model, optimizer, model_state, target_optimzizer
    ):
        """Restore the model and optimizer states from copies."""
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(target_optimzizer.state_dict())

    def reset(self):
        """recover model and optimizer to their initial states."""
        self.load_model_and_optimizer(
            self._model, self._optimizer, self.model_state_dict, self._base_optimizer
        )
        # restore the teacher model.
        (
            self.model_state_dict,
            self.model_ema,
            self.model_anchor,
        ) = self.copy_model_states(self._model)

    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                outputs = model(batch._x)
            # Teacher prediction
            anchor_prob = torch.nn.functional.softmax(
                self.model_anchor(batch._x), dim=1
            ).max(1)[0]
            standard_ema = self.model_ema(batch._x)
            # Augmentation-averaged Prediction
            outputs_emas = []
            for i in range(self._meta_conf.aug_size):
                outputs_ = self.model_ema(self.transform(batch._x)).detach()
                outputs_emas.append(outputs_)
            # Threshold choice discussed in supplementary
            if anchor_prob.mean(0) < self._meta_conf.threshold_cotta:
                outputs_ema = torch.stack(outputs_emas).mean(0)
            else:
                outputs_ema = standard_ema
            # Student update
            loss = adaptation_utils.teacher_student_softmax_entropy(
                outputs, outputs_ema
            ).mean(0)

            # apply fisher regularization when enabled
            if self.fishers is not None:
                ewc_loss = 0
                for name, param in model.named_parameters():
                    if name in self.fishers:
                        ewc_loss += (
                            self._meta_conf.fisher_alpha
                            * (
                                self.fishers[name][0]
                                * (param - self.fishers[name][1]) ** 2
                            ).sum()
                        )
                loss += ewc_loss

        with timer("backward"):
            loss.backward()
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            optimizer.step()
            optimizer.zero_grad()
            # Update the teacher model
            self.model_ema = self.update_ema_variables(
                ema_model=self.model_ema,
                model=self._model,
                alpha_teacher=self._meta_conf.alpha_teacher,
            )
            # Stochastic restore
            self.stochastic_restore()
        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
            "yhat": outputs_ema,
        }

    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                batch,
                timer,
                random_seed=random_seed,
            )

            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(model).state_dict(),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **adaptation_result,
                },
                current_batch=batch,
            )

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch: Batch,
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
                batch=current_batch,
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
        return "cotta"


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
