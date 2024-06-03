from torchvision import transforms
from PIL import Image
from algos.fastflow2d._defaultconfigs import fastflow_default_model_params
from algos.patchcore._defaultconfigs import patchcore_default_model_params
from algos.pefm._defaultconfigs import pefm_default_model_params, pfm_default_model_params
from algos.cflow._defaultconfigs import cflow_default_model_params

image_size = 256
specific_parameters = [{
                        "algo_class_name": "PFM",
                        "model_params": pfm_default_model_params,
                        "model_description": "PFM model", # description saved to each run from the dictionary - DIFFERENT to run_description in the cmd, which is the description attached to all the runs in a given config file - use no more than 20 characters - used in the results directory name
                       }
]

model_list = []
for item in specific_parameters:
    final_parameter_set  = {"algo_class_name": None, 
                            "model_params": None,
                            "evaluate_n_epochs": 8,
                            "epochs": 200,
                            "training_transformations":  transforms.Compose([]),
                            "training_transformation_string": "transforms.Compose([])",
                            "identical_transforms": transforms.Compose([transforms.Resize((image_size, image_size), 
                                                                        interpolation=Image.NEAREST),
                                                                        ]),
                            "batch_size": 8,
                            "save_model_every_n_epochs": None,
                            "save_heatmaps_n_epochs": 1,
                            "evaluate_n_epochs": 1,
                            "test_batch_size": 8,
                            "device": "cuda:0",
                            "input_size": 256,
                            "model_description": "patchcore model", # description saved to each run from the dictionary - DIFFERENT to run_description in the cmd, which is the description attached to all the runs in a given config file - use no more than 20 characters - used in the results directory name
                            "model_long_description": "", 
                            "save_metric_meta_data": False, # whether the metric data is saved alongside the metric results, setting as True will results in a lot of physical memory consumption
                            "wandb": True,
                            }
    for key, value in item.items():
        final_parameter_set[key] = value

    model_list.append(final_parameter_set)

for model in model_list:
    model["train_time_limit"] = 2*3600


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
