class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        seed=[2022, 2023, 2024],
        main_file=[
            "run_exp.py",
            ],
        job_name=[
            "imagenet_c_episodic_oracle_model_selection",
            # "imagenet_c_online_last_iterate",
        ],
        base_data_name=[
            "imagenet",
        ],
        data_names=[
            "imagenet_c_deterministic-gaussian_noise-5",
            "imagenet_c_deterministic-shot_noise-5",
            "imagenet_c_deterministic-impulse_noise-5",
            "imagenet_c_deterministic-defocus_blur-5",
            "imagenet_c_deterministic-glass_blur-5",
            "imagenet_c_deterministic-motion_blur-5",
            "imagenet_c_deterministic-zoom_blur-5",
            "imagenet_c_deterministic-snow-5",
            "imagenet_c_deterministic-frost-5",
            "imagenet_c_deterministic-fog-5",
            "imagenet_c_deterministic-brightness-5",
            "imagenet_c_deterministic-contrast-5",
            "imagenet_c_deterministic-elastic_transform-5",
            "imagenet_c_deterministic-pixelate-5",
            "imagenet_c_deterministic-jpeg_compression-5",
        ],
        model_name=[
            "resnet50",
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
            "./data/pretrained_ckpts/classification/resnet26_with_head/cifar10/rn26_bn.pth", # Since ttab will automatically download the pretrained model from torchvision or huggingface, what ckpt_path is here does not matter.
        ],
        # oracle_model_selection
        lr_grid=[
            [1e-3], 
            [5e-4], 
            [1e-4],
        ],
        n_train_steps=[10],
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
            "cuda:1",
            "cuda:2",
            "cuda:3",
            "cuda:4",
            "cuda:5",
            "cuda:6",
            "cuda:7",
        ],
        gradient_checkpoint=["false"],
    )
