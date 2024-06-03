from torchvision import transforms
from PIL import Image
import numpy as np
from algos.ast._defaultconfigs import ast_default_model_params

image_size = ast_default_model_params["input_size"]
model_list = [{"algo_class_name": "ast", 
                "model_params": ast_default_model_params,
                "epochs": ast_default_model_params["switch_teacher_student_training_epoch"]*2,
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
                "model_description": "ast run", # description saved to each run from the dictionary - DIFFERENT to run_description in the cmd, which is the description attached to all the runs in a given config file - use no more than 20 characters - used in the results directory name
                "model_long_description": "", 
                "save_metric_meta_data": False, # whether the metric data is saved alongside the metric results, setting as True will results in a lot of physical memory consumption
                "wandb": True,
                "specific_evaluation_regime": {"epochs": list(np.arange(ast_default_model_params["switch_teacher_student_training_epoch"]+1, 
                                                                    ast_default_model_params["switch_teacher_student_training_epoch"]*2, 
                                                                    5)),},      
                }]

model_list[0]["train_time_limit"] = 2*3600

dataset_list = [
            'visa_pipe_fryum',
            'visa_pcb4',
            'visa_pcb3',
 'visa_pcb2',
 'visa_pcb1',
 'visa_macaroni2',
 'visa_macaroni1',
 'visa_fryum',
 'visa_chewinggum',
 'visa_cashew',
 'visa_capsules',
 'visa_candle'] 

metrics = ["all"]
