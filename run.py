'''
Copyright 2024 - authors of TMLR submission 
'''
import os 
import sys
sys.path.append(os.getcwd())
import argparse
import traceback
import importlib
from torchvision import transforms
from wrapper.wrapper import train_algo
from algos.algo_class_list import get_algo
from data.configure_dataset import load_dataset, RESULTS_DIRECTORY
from datetime import datetime


default_run_arguments = {
    "model_description": "", # description attached to the path of the model
    "model_long_description": "",
    "save_metric_meta_data": False,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    parser.add_argument("--run_description", default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", type=str, default="not entered")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    config = importlib.import_module(".".join(list(args.config_file.split(os.sep)))[:-3])

    run_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # we loop through each model in the config.model_list and each dataset in the config.dataset_list
    for ind, model_arguments in enumerate(config.model_list):
        model_dataset_paths = {}
        print(f"Parent results directory: {RESULTS_DIRECTORY}\n")
        print(f"Start time: {run_time}")
        print(f"Algo: {model_arguments['algo_class_name']}\n")

        for key, value in default_run_arguments.items():
            if key not in model_arguments:
                model_arguments[key] = value
            
        model_path = os.path.join(RESULTS_DIRECTORY, 
                                  run_time+"_"+args.run_description[:20].replace(" ", "_"), 
                                  model_arguments["algo_class_name"]+"_"+model_arguments["model_description"] )
        print("\n")
        if "dataset_list_override" in model_arguments:
            dataset_list = model_arguments["dataset_list_override"]
            print("Got dataset override keys:")
            print(dataset_list)
        else:
            dataset_list = config.dataset_list
            print("Got dataset keys:")
            print(dataset_list)
            print("\n")
        for dataset_key in config.dataset_list:
            print(f'Running model: {model_arguments["algo_class_name"].lower()} {model_arguments["model_description"]}, dataset: {dataset_key}')
            try:            
                dataset = load_dataset(dataset_key)

                if args.device:
                    model_arguments["device"]                 = str(args.device)
                    model_arguments["model_params"]["device"] = str(args.device)
                else:
                    model_arguments["model_params"]["device"] = model_arguments["device"]
                
                if not args.wandb=="not entered":
                    model_arguments["wandb"]                 = bool(int(args.wandb))
                    model_arguments["model_params"]["wandb"] = bool(int(args.wandb))
                elif "wandb" not in model_arguments:
                    model_arguments["wandb"]                 = False
                    model_arguments["model_params"]["wandb"] = False
                else:
                    model_arguments["model_params"]["wandb"] = model_arguments["wandb"] 

                model_arguments["model_params"]["wandb"] = model_arguments["wandb"]
                algo_class = get_algo(model_arguments["algo_class_name"].lower())(**model_arguments["model_params"])
                model_arguments["pre_pro_transforms"] = transforms.Compose([transforms.Normalize(mean=dataset['mean'],std=dataset['std'])])
                if model_arguments["epochs"]==0:
                    model_arguments["epochs"]=1
                
                datetime_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

                model_dataset_path = os.path.join(model_path, dataset_key)
                print(f"Results path for model+dataset experiment:\n{model_dataset_path}\n")
                train_algo(algo_class, 
                            dataset_key,
                            config.metrics,
                            run_time = run_time,
                            model_dataset_path = model_dataset_path,
                            run_description = args.run_description,
                            model_arguments = model_arguments,
                            )
                model_dataset_paths[dataset_key] = model_dataset_path
            except:
                print(f'Error on run {model_arguments["algo_class_name"]}, dataset: {dataset_key}')
                print(traceback.format_exc())
        
