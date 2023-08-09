class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        seed=[2022, 2023, 2024],
        main_file=[
            "run_exp.py",
            ],
        job_name=[
            # "pacs_online_last_iterate",
            "pacs_episodic_oracle_model_selection"
        ],
        base_data_name=[
            "pacs",
        ],
        data_names=[
            "pacs_cartoon",
            "pacs_photo",
            "pacs_sketch",
            "pacs_art",
            "pacs_photo",
            "pacs_sketch",
            "pacs_art",
            "pacs_cartoon",
            "pacs_sketch",
            "pacs_art",
            "pacs_cartoon",
            "pacs_photo",
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
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_art.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_art.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_art.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_cartoon.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_cartoon.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_cartoon.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_photo.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_photo.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_photo.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_sketch.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_sketch.pth",
            "./data/pretrained_ckpts/classification/resnet50_with_head/pacs/rn50_bn_sketch.pth",
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
