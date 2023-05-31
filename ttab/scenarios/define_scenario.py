# -*- coding: utf-8 -*-

import ttab.configs.utils as config_utils
from ttab.loads.datasets.dataset_shifts import (
    NaturalShiftProperty,
    NoShiftProperty,
    SyntheticShiftProperty,
    data2shift,
)
from ttab.scenarios import (
    CrossMixture,
    HeterogeneousNoMixture,
    HomogeneousNoMixture,
    InOutMixture,
    Scenario,
    TestCase,
    TestDomain,
)
from ttab.scenarios.default_scenarios import default_scenarios


def get_inter_domain(config):
    """
    Defines the type of inter-domain shift.

    Retrive config values for particular arguments which are necessary to determine an inter-domain shift.

    Args:
        config: config object.
    Returns:
        NamedTuple.
    """
    assert "inter_domain" in vars(config)
    inter_domain_name = getattr(config, "inter_domain")
    inter_domain_fn = {
        "HomogeneousNoMixture": HomogeneousNoMixture,
        "HeterogeneousNoMixture": HeterogeneousNoMixture,
        "InOutMixture": InOutMixture,
        "CrossMixture": CrossMixture,
    }.get(inter_domain_name, HomogeneousNoMixture)

    if "InOutMixture" == inter_domain_name:
        arg_names = ["ratio"]
        arg_values = config_utils.build_dict_from_config(arg_names, config)
        return inter_domain_fn(**arg_values)
    elif "HeterogeneousNoMixture" == inter_domain_name:
        arg_names = ["non_iid_pattern", "non_iid_ness"]
        arg_values = config_utils.build_dict_from_config(arg_names, config)
        return inter_domain_fn(**arg_values)
    else:
        return inter_domain_fn()


def get_test_case(config):
    """
    Config-compliant definition of the scenario instance.

    Args:
        config: config object.
    Returns:
        test_case: NamedTuple.
    """
    # get intra_domain/inter_domain for each test domain.
    inter_domain = get_inter_domain(config)

    # get other arguments for test_case.
    extra_arg_names = list(arg for arg in TestCase._fields if arg != "inter_domain")
    extra_args = config_utils.build_dict_from_config(extra_arg_names, config)

    return TestCase(inter_domain=inter_domain, **extra_args)


def _is_defined_name_tuple(in_object):
    return any(
        [
            isinstance(in_object, defined_named_tuple)
            for defined_named_tuple in [
                HomogeneousNoMixture,
                HeterogeneousNoMixture,
                InOutMixture,
                CrossMixture,
                TestCase,
                TestDomain,
                SyntheticShiftProperty,
                NaturalShiftProperty,
                NoShiftProperty,
            ]
        ]
    )


def _registry_named_tuple(input):
    """
    Iteratively convert a NamedTuple into a dictionary.
    Args:
        input: NamedTuple or list of NamedTuple
    Returns:
        new_dict: dictionary
    """
    if _is_defined_name_tuple(input):
        new_dict = dict()
        for key, val in dict(input._asdict()).items():
            new_dict[key] = dict(val._asdict()) if _is_defined_name_tuple(val) else val
        return new_dict
    elif isinstance(input, list) and all(
        [_is_defined_name_tuple(val) for val in input]
    ):
        return [_registry_named_tuple(val) for val in input]
    else:
        return input


def scenario_registry(config, scenario):
    """
    Inherit arguments from the `scenario` and add them into `config`.

    Args:
        config: namespace
        scenario: NamedTuple (its value may also be a NamedTuple)
    Returns:
        config: namespace
    """
    # retrive name of arguments.
    field_names = list(scenario._fields)

    dict_config = vars(config)
    dict_scenario = scenario._asdict()
    for field_name in field_names:
        dict_config[field_name] = _registry_named_tuple(dict_scenario[field_name])
    return config


def extract_synthetic_info(data_name):
    """
    Reads input `data_name` and defines the value of arguments necessary for generating synthetic shift dataset.

    Args:
        data_name: string
    Returns:
        shift_state: string
        shift_name: string
        shift_degree: int
    """

    # corruption data.
    if any(
        [
            base_data_name in data_name
            for base_data_name in ["cifar10", "cifar100", "imagenet"]
        ]
    ):
        _new_data_names = data_name.split(
            "_", 2
        )  # support string like "cifar10_c_stochastic-gaussian_noise-5", "imagenet_c_deterministic-gaussian_noise-5"
        assert len(_new_data_names) == 3
        _patterns = _new_data_names[-1].split(
            "-"
        )  # where the corruption data comes from, corruption name and severity
        assert len(_patterns) == 3, "<shift_state>-<shift_name>-<shift_degree>"
        return _patterns[0], _patterns[1], int(_patterns[2])
    # synthetic spurious correlation data.
    elif data_name == "coloredmnist":
        shift_state = "stochastic"
        shift_name = "color"
        shift_degree = 0
        return shift_state, shift_name, shift_degree


def _get_shift(config, data_name):
    """
    Defines an instance of P(a^{1:K}) as shown in Figure 6 of the paper.

    Args:
        config: namespace
        data_name: string
    Returns:
        shift_property: NamedTuple
    """

    # split data_name and make sure of using a correct format.
    # please check the description of TestDomain for more details.
    _data_names = data_name.split("_")

    # get the shift type
    base_data_name = _data_names[0]
    _data_name = "_".join(_data_names[:2]) if len(_data_names) >= 2 else _data_names[0]
    shift_type = data2shift[_data_name]

    # define shift property.
    if shift_type == "no_shift":
        shift_property = NoShiftProperty(has_shift=False)
    elif shift_type == "natural":
        version = (
            "_".join(_data_names[2:]) if len(_data_names) > 2 else None
        )  # e.g., cifar10_shiftedlabel_constant-size-dirichlet_gaussian_noise_5
        shift_property = NaturalShiftProperty(version=version, has_shift=True)
    elif shift_type == "synthetic":
        shift_state, shift_name, shift_degree = extract_synthetic_info(data_name)
        shift_property = SyntheticShiftProperty(
            has_shift=True,
            shift_degree=shift_degree,
            shift_name=shift_name,
            version=shift_state,  # either 'stochastic' or 'deterministic'
        )

    # extract domain data sampling scheme.
    extra_arg_names = [
        "domain_sampling_name",
        "domain_sampling_value",
        "domain_sampling_ratio",
    ]
    extra_args = config_utils.build_dict_from_config(extra_arg_names, config)

    return TestDomain(
        base_data_name=base_data_name,
        data_name=data_name,
        shift_type=shift_type,
        shift_property=shift_property,
        **extra_args,
    )


def get_scenario(config):
    # Check whether there is a specified scenario or not.
    scenario = default_scenarios.get(config.test_scenario, None)
    if scenario is not None:
        return scenario

    # Use a new scenario determined by user rather than defaults.
    # In default, we assume the shift property and sampling strategy for each data domain is the same.
    data_names = config.data_names.split(";")
    test_domains = [_get_shift(config, data_name) for data_name in data_names]

    # setup of test_case
    test_case = get_test_case(config)

    # init the scenario
    scenario = Scenario(
        base_data_name=config.base_data_name,
        src_data_name=config.src_data_name,
        test_domains=test_domains,
        test_case=test_case,
        task=config.task,
        model_name=config.model_name,
        model_adaptation_method=config.model_adaptation_method,
        model_selection_method=config.model_selection_method,
    )
    return scenario
