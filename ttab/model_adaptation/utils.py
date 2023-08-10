# -*- coding: utf-8 -*-
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ttab.loads.models.resnet import (
    ResNetCifar,
    ResNetImagenet,
    ResNetMNIST,
    ViewFlatten,
)
from ttab.loads.models.wideresnet import WideResNet

"""optimization dynamics"""


def define_optimizer(meta_conf, params, lr=1e-3):
    """Set up optimizer for adaptation."""
    weight_decay = meta_conf.weight_decay if hasattr(meta_conf, "weight_decay") else 0

    if not hasattr(meta_conf, "optimizer") or meta_conf.optimizer == "SGD":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=meta_conf.momentum if hasattr(meta_conf, "momentum") else 0.9,
            dampening=meta_conf.dampening if hasattr(meta_conf, "dampening") else 0,
            weight_decay=weight_decay,
            nesterov=meta_conf.nesterov if hasattr(meta_conf, "nesterov") else True,
        )
    elif meta_conf.optimizer == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=(meta_conf.beta if hasattr(meta_conf, "beta") else 0.9, 0.999),
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError


def lr_scheduler(optimizer, iter_ratio, gamma=10, power=0.75):
    decay = (1 + gamma * iter_ratio) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
    return optimizer


class SAM(torch.optim.Optimizer):
    """
    SAM is an optimizer proposed to seek parameters that lie in neighborhoods having uniformly low loss.

    Sharpness-Aware Minimization for Efficiently Improving Generalization
    https://arxiv.org/abs/2010.01412
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


"""method-wise modification on model structure"""

# for bn_adapt
def modified_bn_forward(self, input):
    """
    Leverage the statistics already computed on the seen data as a prior and infer the test statistics for each test batch as a weighted sum of
    prior statistics and estimated statistics on the current batch.

    Improving robustness against common corruptions by covariate shift adaptation
    https://arxiv.org/abs/2006.16971
    """
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(
        input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps
    )


class shared_ext_from_layer4(nn.Module):
    """
    Select all layers before layer4 and layer4 as the shared feature extractor for the main and auxiliary branches.

    Only used for ResNets.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        return self.model.forward_features(x)

    def _select_layers(self):
        if isinstance(self.model, ResNetImagenet):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "layer4": self.model.layer4,
                "avgpool": self.model.avgpool,
                "ViewFlatten": ViewFlatten(),
            }
        elif isinstance(self.model, ResNetMNIST):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "layer4": self.model.layer4,
                "ViewFlatten": ViewFlatten(),
            }
        else:
            raise NotImplementedError

    def make_train(self):
        for _, layer_module in self.layers.items():
            layer_module.train()

    def make_eval(self):
        for _, layer_module in self.layers.items():
            layer_module.eval()


class shared_ext_from_layer3(nn.Module):
    """
    Select all layers before layer3 and layer3 as the shared feature extractor for the main and auxiliary branches.

    Only used for ResNets.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        return self.model.forward_features(x)

    def _select_layers(self):
        if isinstance(self.model, ResNetCifar):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "avgpool": self.model.avgpool,
                "ViewFlatten": ViewFlatten(),
            }
        elif isinstance(self.model, (ResNetImagenet, ResNetMNIST)):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
            }
        elif isinstance(self.model, WideResNet):
            return {
                "conv1": self.model.conv1,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
                "layer3": self.model.layer3,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "avgpool": self.model.avgpool,
                "ViewFlatten": ViewFlatten(),
            }
        else:
            raise NotImplementedError

    def make_train(self):
        for _, layer_module in self.layers.items():
            layer_module.train()

    def make_eval(self):
        for _, layer_module in self.layers.items():
            layer_module.eval()


class shared_ext_from_layer2(nn.Module):
    """
    Select all layers before layer2 and layer2 as the shared feature extractor for the main and auxiliary branches.

    Only used for ResNets.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        return self.model.forward_features(x)

    def _select_layers(self):
        if isinstance(self.model, ResNetCifar):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
            }
        elif isinstance(self.model, (ResNetImagenet, ResNetMNIST)):
            return {
                "conv1": self.model.conv1,
                "bn1": self.model.bn1,
                "relu": self.model.relu,
                "maxpool": self.model.maxpool,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
            }
        elif isinstance(self.model, WideResNet):
            return {
                "conv1": self.model.conv1,
                "layer1": self.model.layer1,
                "layer2": self.model.layer2,
            }
        else:
            raise NotImplementedError

    def make_train(self):
        for _, layer_module in self.layers.items():
            layer_module.train()

    def make_eval(self):
        for _, layer_module in self.layers.items():
            layer_module.eval()


def head_from_classifier(model, dim_out):
    """Select the last classifier layer in ResNets as head."""
    # Self-supervised task used in TTT is rotation prediction. Thus the out_features = 4.
    head = nn.Linear(
        in_features=model.classifier.in_features, out_features=dim_out, bias=True
    )
    return head


def head_from_last_layer1(model, dim_out):
    """
    Select the layer 3 or 4 and the following classifier layer as head.

    Only used for ResNets.
    """
    if isinstance(model, ResNetCifar):
        head = copy.deepcopy([model.layer3, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, ResNetImagenet):
        head = copy.deepcopy([model.layer4, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, ResNetMNIST):
        head = copy.deepcopy([model.layer4])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, WideResNet):
        head = copy.deepcopy([model.layer3, model.bn1, model.relu, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.classifier.in_features, dim_out, bias=False))
    elif isinstance(model, models.ResNet):
        # for torchvision.models.resnet50
        head = copy.deepcopy([model.layer4, model.avgpool])
        head.append(ViewFlatten())
        head.append(nn.Linear(model.fc.in_features, dim_out, bias=False))
    else:
        raise NotImplementedError

    return nn.Sequential(*head)


class ExtractorHead(nn.Module):
    """
    Combine the extractor and the head together in ResNets.
    """

    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x):
        return self.head(self.ext(x))

    def make_train(self):
        self.ext.make_train()
        self.head.train()

    def make_eval(self):
        self.ext.make_eval()
        self.head.eval()


class VitExtractor(nn.Module):
    """
    Combine the extractor and the head together in ViTs.
    """

    def __init__(self, model):
        super(VitExtractor, self).__init__()
        self.model = model
        self.layers = self._select_layers()

    def forward(self, x):
        x = self.model.forward_features(x)
        if self.model.global_pool:
            x = (
                x[:, self.model.num_prefix_tokens :].mean(dim=1)
                if self.model.global_pool == "avg"
                else x[:, 0]
            )
        x = self.model.fc_norm(x)
        return x

    def _select_layers(self):
        layers = []
        for named_module, module in self.model.named_children():
            if not module == self.model.get_classifier():
                layers.append(module)
        return layers

    def make_train(self):
        for layer in self.layers:
            layer.train()

    def make_eval(self):
        for layer in self.layers:
            layer.eval()


# for ttt++
class FeatureQueue:
    def __init__(self, dim, length):
        self.length = length
        self.queue = torch.zeros(length, dim)
        self.ptr = 0

    @torch.no_grad()
    def update(self, feat):

        batch_size = feat.shape[0]
        assert self.length % batch_size == 0  # for simplicity

        # replace the features at ptr (dequeue and enqueue)
        self.queue[self.ptr : self.ptr + batch_size] = feat
        self.ptr = (self.ptr + batch_size) % self.length  # move pointer

    def get(self):
        cnt = (self.queue[-1] != 0).sum()
        if cnt.item():
            return self.queue
        else:
            return None


# for note
class InstanceAwareBatchNorm2d(nn.Module):
    def __init__(
        self, num_channels, k=3.0, eps=1e-5, momentum=0.1, threshold=1, affine=True
    ):
        super(InstanceAwareBatchNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.k = k
        self.threshold = threshold
        self.affine = affine
        self._bn = nn.BatchNorm2d(
            num_channels, eps=eps, momentum=momentum, affine=affine
        )

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, h, w = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True)  # IN

        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(
                x, dim=[0, 2, 3], keepdim=True, unbiased=True
            )
        else:
            if (
                self._bn.track_running_stats == False
                and self._bn.running_mean is None
                and self._bn.running_var is None
            ):  # use batch stats
                sigma2_b, mu_b = torch.var_mean(
                    x, dim=[0, 2, 3], keepdim=True, unbiased=True
                )
            else:
                mu_b = self._bn.running_mean.view(1, c, 1, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1, 1)

        if h * w <= self.threshold:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / (h * w))
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (h * w - 1))

            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)

            sigma2_adj = sigma2_b + self._softshrink(
                sigma2 - sigma2_b, self.k * s_sigma2
            )

            sigma2_adj = F.relu(sigma2_adj)  # non negative

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
        if self.affine:
            weight = self._bn.weight.view(c, 1, 1)
            bias = self._bn.bias.view(c, 1, 1)
            x_n = x_n * weight + bias
        return x_n


class InstanceAwareBatchNorm1d(nn.Module):
    def __init__(
        self, num_channels, k=3.0, eps=1e-5, momentum=0.1, threshold=1, affine=True
    ):
        super(InstanceAwareBatchNorm1d, self).__init__()
        self.num_channels = num_channels
        self.k = k
        self.eps = eps
        self.threshold = threshold
        self.affine = affine
        self._bn = nn.BatchNorm1d(
            num_channels, eps=eps, momentum=momentum, affine=affine
        )

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, l = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2], keepdim=True, unbiased=True)
        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
        else:
            if (
                self._bn.track_running_stats == False
                and self._bn.running_mean is None
                and self._bn.running_var is None
            ):  # use batch stats
                sigma2_b, mu_b = torch.var_mean(
                    x, dim=[0, 2], keepdim=True, unbiased=True
                )
            else:
                mu_b = self._bn.running_mean.view(1, c, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1)

        if l <= self.threshold:
            mu_adj = mu_b
            sigma2_adj = sigma2_b

        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / l)  ##
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (l - 1))

            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)
            sigma2_adj = sigma2_b + self._softshrink(
                sigma2 - sigma2_b, self.k * s_sigma2
            )
            sigma2_adj = F.relu(sigma2_adj)

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        if self.affine:
            weight = self._bn.weight.view(c, 1)
            bias = self._bn.bias.view(c, 1)
            x_n = x_n * weight + bias

        return x_n


"""Auxiliary tasks"""

# rotation prediction task
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def tensor_rot_180(x):
    return x.flip(2).flip(1)


def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        if label == 1:
            img = tensor_rot_90(img)
        elif label == 2:
            img = tensor_rot_180(img)
        elif label == 3:
            img = tensor_rot_270(img)
        images.append(img.unsqueeze(0))
    return torch.cat(images)


def rotate_batch(batch, label, device, generator=None):
    if label == "rand":
        labels = torch.randint(
            4, (len(batch),), generator=generator, dtype=torch.long
        ).to(device)
    elif label == "expand":
        labels = torch.cat(
            [
                torch.zeros(len(batch), dtype=torch.long),
                torch.zeros(len(batch), dtype=torch.long) + 1,
                torch.zeros(len(batch), dtype=torch.long) + 2,
                torch.zeros(len(batch), dtype=torch.long) + 3,
            ]
        ).to(device)
        batch = batch.repeat((4, 1, 1, 1))

    return rotate_batch_with_labels(batch, labels), labels


"""loss-related functions."""


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def teacher_student_softmax_entropy(
    x: torch.Tensor, x_ema: torch.Tensor
) -> torch.Tensor:
    """Cross entropy between the teacher and student predictions."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def entropy(input):
    bs = input.size(0)
    ent = -input * torch.log(input + 1e-5)
    ent = torch.sum(ent, dim=1)
    return ent


def covariance(features):
    assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)
    return cov


def coral(cs, ct):
    d = cs.shape[0]
    loss = (cs - ct).pow(2).sum() / (4.0 * d**2)
    return loss


def linear_mmd(ms, mt):
    loss = (ms - mt).pow(2).mean()
    return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, device, epsilon=0.1, reduction=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).cpu(), 1
        )
        targets = targets.to(self.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super().__init__()
        self.temp_factor = temp_factor

    def forward(self, x):

        softmax = F.softmax(x / self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax + 1e-6)
        b = entropy.mean()

        return b
