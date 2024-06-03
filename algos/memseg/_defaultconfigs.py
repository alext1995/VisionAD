import os
if os.name=='nt':
    texture_source_dir = r"D:\Documents\PhD_work\supporting_data\dtd"
else:
    texture_source_dir = r"/mnt/faster0/adjt20/support_datasets/dtd"


memseg_default_model_params = {"texture_source_dir" : texture_source_dir,
                                "structure_grid_size": 8,
                                "transparency_range": (0.15, 1),
                                "perlin_scale": 6,
                                "min_perlin_scale": 0,
                                "perlin_noise_threshold": 0.5,
                                "nb_memory_sample": 30,
                                "feature_extractor_name": 'resnet18',
                                "l1_weight": 0.6,
                                "num_training_steps": 5000,
                                "focal_weight": 0.4,
                                "focal_alpha": None,
                                "focal_gamma": 4,
                                "lr": 0.003,
                                "weight_decay": 0.0005,
                                "min_lr": 0.0001,
                                "warmup_ratio": 0.1,
                                "use_scheduler": True,
                                "always_add_anomaly": True,
                                }