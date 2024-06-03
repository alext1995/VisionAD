from torchvision import transforms
from PIL import Image
from algos.altub._defaultconfigs import ff_altub_default_model_params
image_size = 256

ff_altub_default_model_params["zero_grads"] = True
ff_altub_default_model_params["mode"]       = "multi_channel_mean_var"

model_list = [{"algo_class_name": "fastflow2d_altub", 
                "model_params": ff_altub_default_model_params,
                "epochs": 500,
                "training_transformations":  transforms.Compose([]),
                "training_transformation_string": "transforms.Compose([])",
                "identical_transforms": transforms.Compose([transforms.Resize((image_size, image_size), 
                                                            interpolation=Image.NEAREST),
                                                            ]),
                "batch_size": 32,
                "save_model_every_n_epochs": None,
                "save_heatmaps_n_epochs": 1,
                "evaluate_n_epochs": 1,
                "test_batch_size": 8,
                "device": "cuda:0",
                "input_size": 256,
                "model_description": "fastflow altub model", # description saved to each run from the dictionary - DIFFERENT to run_description in the cmd, which is the description attached to all the runs in a given config file - use no more than 20 characters - used in the results directory name
                "model_long_description": "", 
                "save_metric_meta_data": False, # whether the metric data is saved alongside the metric results, setting as True will results in a lot of physical memory consumption
                "wandb": True,
                }]

model_list[0]["train_time_limit"] = 2*3600

dataset_list = ['mvtec_bottle',
                 'mvtec_cable',
                 'mvtec_capsule',
                 'mvtec_carpet',
                 'mvtec_grid',
                 'mvtec_hazelnut',
                 'mvtec_leather',
                 'mvtec_metal_nut',
                 'mvtec_pill',
                 'mvtec_screw',
                 'mvtec_tile',
                 'mvtec_toothbrush',
                 'mvtec_transistor',
                 'mvtec_wood',
                 'mvtec_zipper'
]

metrics = ["all"]
