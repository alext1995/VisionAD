pfm_default_model_params = {"seed": 888,
              "gpu_id": "0",
              "data_trans": 'imagenet', # choices=['navie', 'imagenet']
              "loss_type": 'l2norm+l2', # ['l2norm+l2', 'l2', 'l1', 'consine', 'l2+consine']
              "agent_S": 'resnet34',
              "agent_T": 'resnet50',
              "lr2": 3e-3,
              "lr3": 3e-4,
              "lr4": 3e-4,
              "norm_training_features": False,
              "weight_decay": 1e-5,
              "latent_dim": 200,
              "resize": 256,
              "post_smooth": 0,}

pefm_default_model_params = {"seed": 888,
               "gpu_id": "0",
               "data_trans": 'imagenet', # choices=['navie', 'imagenet']
               "loss_type": 'l2norm+l2', # ['l2norm+l2', 'l2', 'l1', 'consine', 'l2+consine']
               "agent_S": 'resnet34',
               "agent_T": 'resnet50',
               "lr2": 3e-3,
               "lr3": 3e-4,
               "lr4": 3e-4,
               "norm_training_features": False,
               "weight_decay": 1e-5,
               "latent_dim": 200,
               "resize": 256,
               "post_smooth": 0,
               "fea_norm": True,
               "dual_type": "small",
                }