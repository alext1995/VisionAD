msflow_default_model_params = {"mode":"train",
                                "lr": 1e-4,
                                "lr_decay_milestones": [75,90],
                                "lr_decay_gamma": 0.33,
                                "lr_warmup": True,
                                "lr_warmup_from": 0.3,
                                "lr_warmup_epochs": 3,
                                "workers": 4,
                                "epochs": 100,
                                "seed": 2023,
                                "extractor": "wide_resnet50_2",
                                "pool_type": "avg",
                                "parallel_blocks": [2, 5, 8],      
                                "c_conds": [64, 64, 64],
                                "clamp_alpha": 1.9,
                                "warmup_scheduler": True,
                                "top_k": 0.1,
                                "verbose": 2,
}