"""
This file is modified from the official implementation of Test-time training: https://github.com/yueatsprograms/ttt_cifar_release.
"""
from csv import writer
import os
import time
import copy
from datetime import datetime
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

from ttab.loads.datasets.datasets import CIFARDataset
import ttab.model_adaptation.utils as adaptation_utils
from ttab.loads.datasets.dataset_shifts import NoShiftedData
from ttab.loads.datasets.loaders import NormalLoader
from ttab.utils.auxiliary import dict2obj
from ttab.loads.models import resnet, WideResNet
from ttab.loads.define_model import SelfSupervisedModel

from third_party.utils import LabelSmoothingCrossEntropy

config = {
    # general
    "seed": 2022,
    "ckpt_path": "./pretrain/checkpoint",
    "device": "cuda:0",
    "log_dir": "./data/runs",
    # data
    "data_path": "./data/datasets",
    "data_name": "cifar10",
    "num_classes": 10,
    # model
    "model_name": "resnet26",
    "task_name": "classification",
    "resume": False,
    # hyperparams
    "entry_of_shared_layers": "layer2",
    "dim_out": 4,
    "use_iabn": False,
    "iabn_k": 4,
    "use_ls": True,  # label smoothing: https://arxiv.org/abs/1906.02629
    "threshold_note": 1,
    "rotation_type": "expand",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "maxEpochs": 150,
    "milestones": [75, 125],
    "batch_size": 128,
    "save_epoch": 10,  # save weights file per SAVE_EPOCH epoch.
}

DATE_FORMAT = "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
TIME_NOW = datetime.now().strftime(DATE_FORMAT)


def convert_iabn(module, **kwargs):
    module_output = module
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        IABN = (
            adaptation_utils.InstanceAwareBatchNorm2d
            if isinstance(module, nn.BatchNorm2d)
            else adaptation_utils.InstanceAwareBatchNorm1d
        )
        module_output = IABN(
            num_channels=module.num_features,
            k=config.iabn_k,
            eps=module.eps,
            momentum=module.momentum,
            threshold=config.threshold_note,
            affine=module.affine,
        )

        module_output._bn = copy.deepcopy(module)

    for name, child in module.named_children():
        module_output.add_module(name, convert_iabn(child, **kwargs))
    del module
    return module_output


def train(epoch):

    start = time.time()
    model.main_model.train()
    model.ssh.train()
    for batch_index, epoch_fractional, batch in train_loader.iterator(
        batch_size=config.batch_size,
        shuffle=True,
        repeat=False,
        ref_num_data=None,
        num_workers=config.num_workers if hasattr(config, "num_workers") else 2,
        pin_memory=True,
        drop_last=False,
    ):

        optimizer.zero_grad()
        inputs_cls, targets_cls = batch._x, batch._y
        targets_cls_hat = model.main_model(inputs_cls)
        loss = loss_function(targets_cls_hat, targets_cls)

        inputs_ssh, targets_ssh = adaptation_utils.rotate_batch(
            batch._x, config.rotation_type, config.device
        )
        targets_ssh_hat = model.ssh(inputs_ssh)
        loss_ssh = loss_function(targets_ssh_hat, targets_ssh)
        loss += loss_ssh

        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader.dataset) + batch_index + 1

        last_layer = list(model.main_model.children())[-1]
        for name, para in last_layer.named_parameters():
            if "weight" in name:
                writer.add_scalar(
                    "LastLayerGradients/grad_norm2_weights", para.grad.norm(), n_iter
                )
            if "bias" in name:
                writer.add_scalar(
                    "LastLayerGradients/grad_norm2_bias", para.grad.norm(), n_iter
                )

        # update training loss for each iteration
        writer.add_scalar("Train/loss", loss.item(), n_iter)

    for name, param in model.main_model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    # print("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    model.main_model.eval()
    model.ssh.eval()

    test_loss = 0.0  # cost function error
    ssh_loss = 0.0  # ssh task error
    correct = 0.0
    correct_ssh = 0.0

    for batch_index, epoch_fractional, batch in test_loader.iterator(
        batch_size=config.batch_size,
        shuffle=True,
        repeat=False,
        ref_num_data=None,
        num_workers=config.num_workers if hasattr(config, "num_workers") else 2,
        pin_memory=True,
        drop_last=False,
    ):
        # main task.
        with torch.no_grad():
            outputs = model.main_model(batch._x)
            loss = loss_function(outputs, batch._y)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(batch._y).sum()

        # auxiliary task
        with torch.no_grad():
            inputs_ssh, targets_ssh = adaptation_utils.rotate_batch(
                batch._x, config.rotation_type, config.device
            )
            outputs_ssh = model.ssh(inputs_ssh)
            loss_ssh = loss_function(outputs_ssh, targets_ssh)

        ssh_loss += loss_ssh.item()
        _, preds_ssh = outputs_ssh.max(1)
        correct_ssh += preds_ssh.eq(targets_ssh).sum()

    print(
        ("Epoch %d/%d:" % (epoch, config.maxEpochs)).ljust(24)
        + "%.2f\t\t%.2f"
        % (
            correct.float() / len(test_loader.dataset) * 100,
            correct_ssh.float() / (4 * len(test_loader.dataset)) * 100,
        )
    )

    # add informations to tensorboard
    if tb:
        writer.add_scalar(
            "Test/Average loss", test_loss / len(test_loader.dataset), epoch
        )
        writer.add_scalar(
            "Test/Accuracy", correct.float() / len(test_loader.dataset), epoch
        )

    return correct.float() / len(test_loader.dataset)


if __name__ == "__main__":
    config = dict2obj(config)
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    # configure model.
    if "wideresnet" in config.model_name:
        components = config.model_name.split("_")
        depth = int(components[0].replace("wideresnet", ""))
        widen_factor = int(components[1])

        init_model = WideResNet(
            depth,
            widen_factor,
            config.num_classes,
            split_point=config.entry_of_shared_layers,
            dropout_rate=0.3,
        )
    elif "resnet" in config.model_name:
        depth = int(config.model_name.replace("resnet", ""))
        if hasattr(config, "group_norm"):
            init_model = resnet(
                config.data_name,
                depth,
                split_point=config.entry_of_shared_layers,
                group_norm_num_groups=config.group_norm,
            ).to(config.device)
        else:
            init_model = resnet(
                config.data_name, depth, split_point=config.entry_of_shared_layers
            ).to(config.device)
            if config.use_iabn:
                init_model = convert_iabn(init_model)
    model = SelfSupervisedModel(init_model, config)
    model.main_model.to(config.device)
    model.ssh.to(config.device)
    params = list(model.main_model.parameters()) + list(model.ssh.head.parameters())

    # configure dataset.
    data_shift_class = functools.partial(NoShiftedData, data_name=config.data_name)
    train_dataset = CIFARDataset(
        root=os.path.join(config.data_path, config.data_name),
        data_name=config.data_name,
        split="train",
        device=config.device,
        data_augment=True,
        data_shift_class=data_shift_class,
    )
    train_loader = NormalLoader(train_dataset)

    test_dataset = CIFARDataset(
        root=os.path.join(config.data_path, config.data_name),
        data_name=config.data_name,
        split="test",
        device=config.device,
        data_augment=False,
        data_shift_class=data_shift_class,
    )
    test_loader = NormalLoader(test_dataset)

    # configure optimization things.
    if config.use_ls:
        loss_function = LabelSmoothingCrossEntropy()
    else:
        loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    train_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[config.milestones[0], config.milestones[1]],
        gamma=0.1,
        last_epoch=-1,
    )

    model.main_model.requires_grad_(True)
    model.ssh.requires_grad_(True)
    model.main_model.train()
    model.ssh.train()

    # use tensorboard
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    writer = SummaryWriter(
        log_dir=os.path.join(
            config.log_dir, config.model_name, config.data_name, TIME_NOW
        )
    )
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if not config.device == "cpu":
        input_tensor = input_tensor.to(config.device)
    writer.add_graph(model.main_model, input_tensor)

    # create checkpoint folder to save model
    checkpoint_path = os.path.join(
        config.ckpt_path,
        config.model_name + "_with_head",
        config.data_name,
        TIME_NOW,
    )

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if hasattr(config, "group_norm"):
        checkpoint_path = os.path.join(
            checkpoint_path, "{model_name}-{epoch}-{type}-gn-{acc}.pth"
        )
    else:
        checkpoint_path = os.path.join(
            checkpoint_path, "{model_name}-{epoch}-{type}-{acc}.pth"
        )

    best_acc = 0.0

    print("Running...")
    print("Error (%)\t\ttest\t\tself-supervised")
    for epoch in range(1, config.maxEpochs + 1):

        train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch < 0.2 * config.maxEpochs:
            weights_path = checkpoint_path.format(
                model_name=config.model_name + "_with_head",
                epoch=epoch,
                type="init",
                acc=acc,
            )
            # print("saving weights file to {}".format(weights_path))
            states = {
                "model": model.main_model.state_dict(),
                "head": model.head.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(states, weights_path)
            continue

        if epoch > config.milestones[1] and best_acc < acc:
            weights_path = checkpoint_path.format(
                model_name=config.model_name + "_with_head",
                epoch=epoch,
                type="best",
                acc=acc,
            )
            # print("saving weights file to {}".format(weights_path))
            states = {
                "model": model.main_model.state_dict(),
                "head": model.head.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(states, weights_path)
            best_acc = acc
            continue

        if not epoch % config.save_epoch:
            weights_path = checkpoint_path.format(
                model_name=config.model_name + "_with_head",
                epoch=epoch,
                type="regular",
                acc=acc,
            )
            # print("saving weights file to {}".format(weights_path))
            states = {
                "model": model.main_model.state_dict(),
                "head": model.head.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(states, weights_path)

    writer.close()
