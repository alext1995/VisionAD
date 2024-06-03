from torchvision import transforms
from PIL import Image
from algos.cfa._defaultconfigs import cfa_default_model_params
image_size = 256

model_list = []

model_list.append({"algo_class_name": "cfa",
                    "model_params": cfa_default_model_params,
                    "epochs": cfa_default_model_params["epochs"], # or enter the number
                    "training_transformations":  transforms.Compose([]),
                    "training_transformation_string": "transforms.Compose([])",
                    "identical_transforms": transforms.Compose([transforms.Resize((image_size, image_size), 
                                                                interpolation=Image.NEAREST),
                                                                ]),
                    "batch_size": 4,
                    "save_model_every_n_epochs": None,
                    "save_heatmaps_n_epochs": 1,
                    "evaluate_n_epochs": 1,
                    "test_batch_size": 8,
                    "device": "cuda:0",
                    "input_size": 256,
                    "model_description": "cfa model", # description saved to each run from the dictionary - DIFFERENT to run_description in the cmd, which is the description attached to all the runs in a given config file - use no more than 20 characters - used in the results directory name
                    "model_long_description": "", 
                    "save_metric_meta_data": False, # whether the metric data is saved alongside the metric results, setting as True will results in a lot of physical memory consumption
                    "wandb": True,})

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
