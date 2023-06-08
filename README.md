# Test-Time Adaptation Benchmark (TTAB)

This repository is the official implementation of
<br>
**[On Pitfalls of Test-time Adaptation](https://arxiv.org/abs/2306.03536)**, ICML, 2023
<br>
<a href="https://people.epfl.ch/hao.zhao/?lang=en">Hao Zhao*</a>,
<a href="https://sites.google.com/view/yuejiangliu">Yuejiang Liu*</a>,
<a href="https://people.epfl.ch/alexandre.alahi/?lang=en">Alexandre Alahi</a>,
<a href="https://tlin-taolin.github.io">Tao Lin</a>

> TL;DR: We introduce a test-time adaptation benchmark that systematically examines a large array of recent methods under diverse conditions. Our results reveal three common pitfalls in prior efforts.
> - Model selection is exceedingly difficult for test-time adaptation due to online batch dependency.
> - The effectiveness of TTA methods varies greatly depending on the quality and properties of pre-trained models.
> - Even with oracle-based tuning, no existing methods can yet address all common classes of distribution shifts.


## Overview

The TTAB package contains:
1. Data loaders that automatically handle data processing and splitting to cover multiple significant evaluation settings considered in prior work.
2. Unified dataset evaluators that standardize model evaluation for each dataset and setting.
3. Multiple representative Test-time Adaptation (TTA) algorithms.

In addition, the example scripts contain default models, optimizers, and evaluation code.
New algorithms can be easily added and run on all of the TTAB datasets.

## Installation
To run a baseline test, please prepare the relevant pre-trained checkpoints for the base model and place them in `pretrain/ckpt/`.
### Requirements
The TTAB package depends on the following requirements:

- numpy>=1.21.5
- pandas>=1.1.5
- pillow>=9.0.1
- pytz>=2021.3
- torch>=1.7.1
- torchvision>=0.8.2
- timm>=0.6.11
- scikit-learn>=1.0.3
- scipy>=1.7.3
- tqdm>=4.56.2

## Datasets
Distribution shift occurs when the test distribution differs from the training distribution, and it can considerably degrade performance of machine learning models deployed in the real world. The form of distribution shifts differs greatly across varying applications in practice. In TTAB, we collect 10 datasets and systematically sort them into 5 types of distribution shifts:
- Covariate Shift
- Natural Shift
- Domain Generalization
- Label Shift
- Spurious Correlation Shift

![TTAB -- Dataset Description](./figs/overview%20of%20datasets.jpg)
<!-- | Dataset     | Types of distribution shift  | Access to the dataset                                                        |
| ----------- | ---------------------------- | ---------------------------------------------------------------------------- |
| CIFAR10-C   | Covariate shift              | [link](https://zenodo.org/record/2535967#.Y_F1DXbMI2w)                    |
| CIFAR10.1   | Natural shift                | [link](https://github.com/modestyachts/CIFAR-10.1/tree/master/datasets)   |
| OfficeHome  | Domain Generalization        | [link](https://www.hemanthdv.org/officeHomeDataset.html)                  |
| PACS        | Domain Generalization        | [link](https://dali-dl.github.io/project_iccv2017.html)                   |
| Waterbirds  | Spurious correlation         | [link](https://github.com/kohpangwei/group_DRO)                           |
| ColoredMNIST| Spurious correlation         | torchvision or [link](http://yann.lecun.com/exdb/mnist/)                  | -->

## Using the TTAB package

The TTAB package provides a simple, standardized interface for all TTA algorithms and datasets in the benchmark. This short Python snippet covers all of the steps of getting started with a user-customizable configuration, including the choice of TTA algorithms, datasets, base models, model selection methods, experimental setups, evaluation scenarios (we will discuss evaluation scenarios in more detail in [Scenario](#scenario)) and protocols. 

```py
config, scenario = configs_utils.config_hparams(config=init_config)

# Dataset
test_data_cls = define_dataset.ConstructTestDataset(config=config)
test_loader = test_data_cls.construct_test_loader(scenario=scenario)

# Base model.
model = define_model(config=config)
load_pretrained_model(config=config, model=model)

# Algorithms.
model_adaptation_cls = get_model_adaptation_method(
    adaptation_name=scenario.model_adaptation_method
)(meta_conf=config, model=model)
model_selection_cls = get_model_selection_method(selection_name=scenario.model_selection_method)(
    meta_conf=config, model=model
)

# Evaluate.
benchmark = Benchmark(
    scenario=scenario,
    model_adaptation_cls=model_adaptation_cls,
    model_selection_cls=model_selection_cls,
    test_loader=test_loader,
    meta_conf=config,
)
benchmark.eval()
```

### Data loading
For evaluation, the TTAB package provides two types of dataset objects. The standard dataset object stores data, labels and indices as well as several APIs to support high-level manipulation, such as mixing the source and target domains. The standard dataset object serves common evaluation metrics like Top-1 accuracy and cross-entropy. 

To support other metrics, such as worst-group accuracy, for more robust evaluation, we provide a group-wise dataset object that records additional group information.

To provide a more seamless user experience, we have designed a unified data loader that supports all dataset objects. To load data in TTAB, simply run the following command with `config` and `scenario` as inputs.

```py
test_data_cls = define_dataset.ConstructTestDataset(config=config)
test_loader = test_data_cls.construct_test_loader(scenario=scenario)
```

### Scenario
In the scenario section, we outline all relevant parameters for defining a distribution shift problem in practice, such as `test_domain` and `test_case`. In the `test_domain`, we specify the implicit $\mathcal{P}(a^{1:K})$ and selected sampling strategy. `test_case` determines how we organize the existing dataset corresponding to `test_domain` into a data stream that will be fed to TTA methods. Besides, we also define the model architecture, TTA method, and model selection method that we will use for the defined distribution shift problem.

Here, we present an example of `scenario`. Please feel free to suggest a new `scenario` for your research.

```py
"S1": Scenario(
        task="classification",
        model_name="resnet26",
        model_adaptation_method="tent",
        model_selection_method="last_iterate",
        base_data_name="cifar10",
        test_domains=[
            TestDomain(
                base_data_name="cifar10",
                data_name="cifar10_c_deterministic-gaussian_noise-5",
                shift_type="synthetic",
                shift_property=SyntheticShiftProperty(
                    shift_degree=5,
                    shift_name="gaussian_noise",
                    version="deterministic",
                    has_shift=True,
                ),
                domain_sampling_name="uniform",
                domain_sampling_value=None,
                domain_sampling_ratio=1.0,
            )
        ],
        test_case=TestCase(
            inter_domain=HomogeneousNoMixture(has_mixture=False),
            batch_size=64,
            data_wise="batch_wise",
            offline_pre_adapt=False,
            episodic=False,
            intra_domain_shuffle=True,
        ),
    ),
```


## Using the example scripts
We provide an example script that can be used to adapt distribution shifts on the TTAB datasets. 

```bash
python run_exp.py
```

Currently, before using the example script, you need to manually set up the `args` object in the `parameters.py`. This script is configured to use the default base model, dataset, evaluation protocol and reasonable hyperparameters.  

<!-- ## Algorithms

In the `ttab/model_adaptation` folder, we provide implementations of the TTA algorithms benchmarked in our paper. We use unified setups for the base model, datasets, hyperparameters, and evaluators, so new algorithms can be easily added and run on all of the TTAB datasets.

In addition to shared hyperparameters such as `lr`, `weight_decay`, `batch_size`, and `optimizer`, the scripts also take in command line arguments for algorithm-specific hyperparameters.

|                 Algorithm                |     Venue    | Adjust pretraining | Access to source domain | Reuse test data | Coupled w/ BatchNorm | Resetting model |  Optimizer  |
|:----------------------------------------:|:------------:|:------------------:|:-----------------------:|:---------------:|:--------------------:|:---------------:|:-----------:|
| [SHOT](https://arxiv.org/abs/2002.08546) | ICML 2020    |       &cross;      |          &cross;        |      &check;    |        &cross;       |      &cross;    |     SGD     |
| [TTT](https://arxiv.org/abs/1909.13231)  | ICML 2020    |       &check;      |          &cross;        |      &cross;    |        &cross;       |      &cross;    |     SGD     |
| [BN_Adapt](https://arxiv.org/abs/2006.16971) | NeurIPS 2020 |       &cross;      |          &cross;        |      &cross;    |        &check;       |      &cross;    |     -     |
| [TENT](https://arxiv.org/abs/2006.10726) | ICLR 2021    |       &cross;      |          &cross;        |      &cross;    |        &check;       |      &cross;    | Adam & SGDm |
| [T3A](https://openreview.net/forum?id=e_yvNqkJKAW)  | NeurIPS 2021    |       &cross;      |          &cross;        |      &cross;    |        &cross;       |      &cross;    |     -     |
| [Conjugate PL](http://arxiv.org/abs/2207.09640)  | NeurIPS 2022    |       &cross;      |          &cross;        |      &cross;    |        &check;       |      &cross;    |     Adam     |
| [MEMO](https://arxiv.org/abs/2110.09506) | NeurIPS 2022 |       &cross;      |          &cross;        |      &cross;    |        &cross;       |      &check;    |     SGD     |
| [NOTE](https://arxiv.org/abs/2208.05117) | NeurIPS 2022 |       &check;      |          &cross;        |      &cross;    |        &check;       |      &cross;    |     Adam    |
| [SAR](https://openreview.net/pdf?id=g2YraF75Tj)  | ICLR 2023    |       &cross;      |          &cross;        |      &cross;    |        &check;       |      &cross;    |     SAM     |

In order to make a fair comparison across different TTA algorithms, we make reasonable modifications to these algorithms, which may induce inconsistency with their official implementation. -->

## Pretraining
In this [link](https://drive.google.com/drive/folders/1ALNIYnnTJwqP80n9pEjSWtb_UdbcrsVi?usp=sharing), we provide a set of scripts that can be used to pre-train models on the in-distribution TTAB datasets. These pre-trained models were used to benchmark baselines in our paper. Note that we adopt self-supervised learning with a rotation prediction task to train the baseline model in our paper for a fair comparison. In practice, please feel free to choose whatever pre-training methods you prefer, but please pay attention to the setup of TTA methods.
<!-- ## Citing TTAB -->

## Bibliography
If you find this repository helpful for your project, please consider citing:
```
@inproceedings{zhao2023ttab,
  title     = {On Pitfalls of Test-time Adaptation},
  author    = {Zhao, Hao and Liu, Yuejiang and Alahi, Alexandre and Lin, Tao},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023},
}
```
