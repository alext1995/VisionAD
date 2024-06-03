cflow_default_model_params =  {"enc_arch": "wide_resnet50_2", # default='wide_resnet50_2', type=str, feature extractor
                                "dec_arch": 'freia-cflow', # default='freia-cflow', type=str, normalizing flow model (default: freia-cflow)
                                "pool_layers":  3, # default=3, type=int, number of layers used in NF model (default: 3)
                                "coupling_blocks":  8, # default=8, type=int, number of layers used in NF model (default: 8)
                                "batch-size":  32, # default=32, type=int, metavar='B', help='train batch size (default: 32)')
                                "lr": 2e-4, # default=2e-4, type=float, learning rate (default: 2e-4)
                                "meta_epochs": 25, # default=25, type=int, number of meta epochs to train (default: 25)
                                "sub_epochs": 8, # default=8, type=int, number of sub epochs to train (default: 8)
                                "condition_vec": 128,
                                "clamp_alpha": 1.9,
                                "num_sub_epochs_per_meta_epoch": 8,
                                "device": "cuda:0",
                                "N": 256, #default=256, hyperparameter that increases batch size for the decoder model by N
                                "lr_cosine": True,
                                "lr_decay_rate": 0.1,
                                "lr_warm_epochs": 2,
                                "lr_warm": True,
                                "verbose": True, 
                                }