class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        seed=[2022, 2023, 2024],
        main_file=[
            "run_exp.py",
            ],
        job_name=[
            "yearbook_episodic_oracle_model_selection",
            # "yearbook_online_last_iterate",
        ],
        base_data_name=[
            "yearbook",
        ],
        data_names=[
            "yearbook",
        ],
        model_name=[
            "resnet18",
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
        python_path=["/home/ttab/anaconda3/envs/test_algo/bin/python"],
        data_path=["./datasets"],
        ckpt_path=[
            "./pretrain/checkpoint/resnet18_with_head/yearbook/resnet18_bn.pth",
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
        entry_of_shared_layers=["layer3"],
        intra_domain_shuffle=["false"],
        record_preadapted_perf=["true"],
        device=[
            "cuda:0",
        ],
        grad_checkpoint=["false"],
    )
