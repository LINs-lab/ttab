class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        seed=[2022, 2023, 2024],
        main_file=[
            "run_exp.py",
            ],
        job_name=[
            "cifar100c_episodic_oracle_model_selection",
            # "cifar100c_online_last_iterate",
        ],
        base_data_name=[
            "cifar100",
        ],
        data_names=[
            "cifar100_c_deterministic-gaussian_noise-5",
            "cifar100_c_deterministic-shot_noise-5",
            "cifar100_c_deterministic-impulse_noise-5",
            "cifar100_c_deterministic-defocus_blur-5",
            "cifar100_c_deterministic-glass_blur-5",
            "cifar100_c_deterministic-motion_blur-5",
            "cifar100_c_deterministic-zoom_blur-5",
            "cifar100_c_deterministic-snow-5",
            "cifar100_c_deterministic-frost-5",
            "cifar100_c_deterministic-fog-5",
            "cifar100_c_deterministic-brightness-5",
            "cifar100_c_deterministic-contrast-5",
            "cifar100_c_deterministic-elastic_transform-5",
            "cifar100_c_deterministic-pixelate-5",
            "cifar100_c_deterministic-jpeg_compression-5",
        ],
        model_name=[
            "resnet26",
        ],
        model_adaptation_method=[
            # "no_adaptation",
            "tent",
            # "bn_adapt",
            # "t3a",
            # "memo",
            # "shot",
            # "ttt",
            # "note",
            # "sar",
            # "conjugate_pl",
            # "cotta",
            # "eata",
        ],
        model_selection_method=[
            "oracle_model_selection", 
            # "last_iterate",
        ],
        offline_pre_adapt=[
            "false",
        ],
        data_wise=["batch_wise"],
        batch_size=[64],
        episodic=[
            # "false", 
            "true",
        ],
        inter_domain=["HomogeneousNoMixture"],
        non_iid_ness=[0.1],
        non_iid_pattern=["class_wise_over_domain"],
        python_path=["/opt/conda/bin/python"],
        data_path=["/run/determined/workdir/data/"],
        ckpt_path=[
            "./data/pretrained_ckpts/classification/resnet26_with_head/cifar100/rn26_bn.pth",
        ],
        # oracle_model_selection
        lr_grid=[
            [1e-3], 
            [5e-4], 
            [1e-4],
        ],
        n_train_steps=[50],
        # last_iterate
        # lr=[
        #     5e-3,
        #     1e-3,
        #     5e-4,
        # ],
        # n_train_steps=[
        #     1, 
        #     2,
        #     3,
        # ],
        intra_domain_shuffle=["true"],
        record_preadapted_perf=["true"],
        device=[
            "cuda:0",
        ],
        gradient_checkpoint=["false"],
    )
