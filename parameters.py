# -*- coding: utf-8 -*-
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # define test evaluation info.
    parser.add_argument("--root_path", default="./data/logs", type=str)
    parser.add_argument("--data_path", default="./datasets", type=str)
    parser.add_argument(
        "--ckpt_path",
        default="./pretrained_ckpts/classification/resnet26_with_head/cifar10/rn26_bn.pth",
        type=str,
    )
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_cpus", default=2, type=int)

    # define the task & model & adaptation & selection method.
    parser.add_argument("--model_name", default="resnet26", type=str)
    parser.add_argument("--group_norm_num_groups", default=None, type=int)
    parser.add_argument(
        "--model_adaptation_method",
        default="tent",
        choices=[
            "no_adaptation",
            "tent",
            "bn_adapt",
            "memo",
            "shot",
            "t3a",
            "ttt",
            "note",
            "sar",
            "conjugate_pl",
            "cotta",
            "eata",
        ],
        type=str,
    )
    parser.add_argument(
        "--model_selection_method",
        default="last_iterate",
        choices=["last_iterate", "oracle_model_selection"],
        type=str,
    )
    parser.add_argument("--task", default="classification", type=str)

    # define the test scenario.
    parser.add_argument("--test_scenario", default=None, type=str)
    parser.add_argument(
        "--base_data_name",
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
            "imagenet",
            "officehome",
            "pacs",
            "coloredmnist",
            "waterbirds",
        ],
        type=str,
    )
    parser.add_argument("--src_data_name", default="cifar10", type=str)
    parser.add_argument(
        "--data_names", default="cifar10_c_deterministic-gaussian_noise-5", type=str
    )
    parser.add_argument(
        "--data_wise",
        default="batch_wise",
        choices=["batch_wise", "sample_wise"],
        type=str,
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--n_train_steps", default=1, type=int)
    parser.add_argument("--offline_pre_adapt", default=False, type=str2bool)
    parser.add_argument("--episodic", default=False, type=str2bool)
    parser.add_argument("--intra_domain_shuffle", default=True, type=str2bool)
    parser.add_argument(
        "--inter_domain",
        default="HomogeneousNoMixture",
        choices=[
            "HomogeneousNoMixture",
            "HeterogeneousNoMixture",
            "InOutMixture",
            "CrossMixture",
        ],
        type=str,
    )
    # Test domain
    parser.add_argument("--domain_sampling_name", default="uniform", type=str)
    parser.add_argument("--domain_sampling_ratio", default=1.0, type=float)
    # HeterogeneousNoMixture
    parser.add_argument("--non_iid_pattern", default="class_wise_over_domain", type=str)
    parser.add_argument("--non_iid_ness", default=0.1, type=float)
    # for evaluation.
    # label shift
    parser.add_argument(
        "--label_shift_param",
        help="parameter to control the severity of label shift",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--data_size",
        help="parameter to control the size of dataset",
        default=None,
        type=int,
    )
    # optimal model selection
    parser.add_argument(
        "--step_ratios",
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.75],
        help="ratios used to control adaptation step length",
        type=float,
    )
    parser.add_argument("--step_ratio", default=None, type=float)
    # time-varying
    parser.add_argument("--stochastic_restore_model", default=False, type=str2bool)
    parser.add_argument("--restore_prob", default=0.01, type=float)
    parser.add_argument("--fishers", default=False, type=str2bool)
    parser.add_argument(
        "--fisher_size",
        default=5000,
        type=int,
        help="number of samples to compute fisher information matrix.",
    )
    parser.add_argument(
        "--fisher_alpha",
        type=float,
        default=1.5,
        help="the trade-off between entropy and regularization loss",
    )
    # method-wise hparams
    parser.add_argument(
        "--aug_size",
        default=32,
        help="number of per-image augmentation operations in memo and ttt",
        type=int,
    )
    parser.add_argument(
        "--entry_of_shared_layers",
        default=None,
        help="the split position of auxiliary head. Only used in TTT.",
    )
    # metrics
    parser.add_argument(
        "--record_preadapted_perf",
        default=False,
        help="record performance on the local batch prior to implementing test-time adaptation.",
        type=str2bool,
    )
    # misc
    parser.add_argument(
        "--grad_checkpoint",
        default=False,
        help="Trade computation for gpu space.",
        type=str2bool,
    )
    parser.add_argument("--debug", default=False, help="Display logs.", type=str2bool)

    # parse conf.
    conf = parser.parse_args()
    return conf


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


if __name__ == "__main__":
    args = get_args()
