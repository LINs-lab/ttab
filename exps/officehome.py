class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        seed=[2022, 2023, 2024],
        main_file=[
            "run_exp.py",
            ],
        job_name=[
            "officehome_episodic_oracle_model_selection",
            # "officehome_online_last_iterate",
        ],
        base_data_name=[
            "officehome",
        ],
        data_names=[
            "officehome_clipart",
            "officehome_product",
            "officehome_realworld",
            "officehome_art",
            "officehome_product",
            "officehome_realworld",
            "officehome_art",
            "officehome_clipart",
            "officehome_realworld",
            "officehome_art",
            "officehome_clipart",
            "officehome_product",
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
            # "sar",
            # "conjugate_pl",
            # "note",
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
        python_path=["/opt/conda/bin/python"],
        data_path=["/run/determined/workdir/data/"],
        ckpt_path=[
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_art.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_art.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_art.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_clipart.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_clipart.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_clipart.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_product.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_product.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_product.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_realworld.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_realworld.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/officehome/rn50_bn_realworld.pth",
        ],
        # oracle_model_selection
        lr_grid=[
            [1e-3], 
            [5e-4], 
            [1e-4],
        ],
        n_train_steps=[25],
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
        entry_of_shared_layers=["layer3"],
        intra_domain_shuffle=["true"],
        record_preadapted_perf=["true"],
        device=[
            "cuda:0",
        ],
        grad_checkpoint=["false"],
        coupled=[
            "data_names",
            "ckpt_path",
        ],
    )
