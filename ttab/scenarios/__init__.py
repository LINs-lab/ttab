# -*- coding: utf-8 -*-
from typing import List, NamedTuple, Union

from ttab.loads.datasets.dataset_shifts import (
    NaturalShiftProperty,
    NoShiftProperty,
    SyntheticShiftProperty,
)


class TestDomain(NamedTuple):
    """
    The definition of TestDomain: shift property (i.e., P(a^{1:K})) for each domain and its sampling strategy.

    Each data_name follow the pattern of either 1) <a>, or 2), <a>_<b>, or 3) <a>_<b>_<c>,
    where <a> is the base_data_name, <b> is the shift_name, and <c> is the shift version.
    """

    base_data_name: str
    data_name: str
    shift_type: str  # shift between data_name and base_data_name
    shift_property: Union[SyntheticShiftProperty, NaturalShiftProperty, NoShiftProperty]

    domain_sampling_name: str = "uniform"  # ['uniform', 'label_skew']
    domain_sampling_value: float = None  # hyper-parameter for domain_sampling_name
    domain_sampling_ratio: float = 1.0


class HomogeneousNoMixture(NamedTuple):
    """
    Only consider the shift on P(a^{1:K}) in Figure 6 of the paper, but no label shift.
    """

    # no mixture
    has_mixture: bool = False


class HeterogeneousNoMixture(NamedTuple):
    """
    Only consider the shift on P(a^{1:K}) with label shift.

    We use this setting to evaluate TTA methods under the continual distribution shift setting in Table 4 of the paper.
    """

    # no mixture
    has_mixture: bool = False
    non_iid_pattern: str = "class_wise_over_domain"
    non_iid_ness: float = 100


class InOutMixture(NamedTuple):
    """
    Mix the source domain (left) with the target domain (right).
    """

    # mix one in-domain (left) with one out out-domain (right)
    has_mixture: bool = True
    ratio: float = 0.5  # for left domain


class CrossMixture(NamedTuple):
    """
    Mix multiple target domains (right). Consider shuffle data across domains.
    """

    # cross-shuffle test domains.
    has_mixture: bool = True


class TestCase(NamedTuple):
    """
    Defines the interaction across domains and some necessary setups in the test-time.
    """

    inter_domain: Union[
        HomogeneousNoMixture, HeterogeneousNoMixture, InOutMixture, CrossMixture
    ]
    batch_size: int = 32
    data_wise: str = "sample_wise"
    offline_pre_adapt: bool = False
    episodic: bool = True
    intra_domain_shuffle: bool = False


class Scenario(NamedTuple):
    """
    Defines a distribution shift scenario in practice. More details can be found in Setion 4 of the paper.
    """

    task: str
    model_name: str
    model_adaptation_method: str
    model_selection_method: str

    base_data_name: str  # test dataset (base type).
    src_data_name: str  # name of source domain
    test_domains: List[TestDomain]  # a list of domain (will be evaluated in order)
    test_case: TestCase
