'''
Copyright © 2024 Alexander Taylor
'''
import os
import datetime
import json
import numpy as np
from datetime import datetime
from data.configure_dataset import load_dataset, RESULTS_DIRECTORY
from data.load_image_data import (create_dataloaders, make_unnormalise,)
from wrapper.metrics import metric_list, reset_all
import torch
import time
import numpy as np
import os
from collections import defaultdict
import pickle
import wandb
from multiprocessing.pool import Pool
from wrapper.metrics import metric_key_list_async
from pprint import pprint

def train_algo(algo_class, 
                dataset_key,
                metrics,
                run_time,
                model_dataset_path,
                run_description, 
                model_arguments,
                ):
         
    os.makedirs(model_dataset_path, exist_ok=True)
    dataset = load_dataset(dataset_key) 
    dataloader_train, dataloader_regular_test, dataloader_novel_test = create_dataloaders(                                                 
                                                                            pre_pro_transforms=model_arguments["pre_pro_transforms"], 
                                                                            training_transformations=model_arguments["training_transformations"],
                                                                            identical_transforms=model_arguments["identical_transforms"],
                                                                            batch_size = model_arguments["batch_size"],
                                                                            shuffle = True,
                                                                            device = model_arguments["model_params"]["device"], 
                                                                            unnormalise = make_unnormalise(mean=dataset['mean'], 
                                                                                                           std=dataset['std']),
                                                                            test_batch_size = model_arguments["test_batch_size"],
                                                                            **dataset)
    print("Device: ", model_arguments["model_params"]["device"])
                                    
    algo_class.enter_dataloaders(dataloader_train, 
                                 dataloader_regular_test, 
                                 dataloader_novel_test)
    
    
    if not "save_best_preds" in model_arguments:
        model_arguments["save_best_preds"] = True
    if not "save_best_preds_metric" in model_arguments:
        model_arguments["save_best_preds_metric"] = "PL"
     
    print("Run and model paramteres:")
    pprint(model_arguments)
    print("\n")


    if model_arguments["wandb"]:
        if run_description != "":
            description = run_description+"_"
        else:
            description = ""
        name = f"{description}_{model_arguments['algo_class_name']}_{dataset_key}"
        print("Creating project with name: ", f"{description}_{model_arguments['algo_class_name']}_{dataset_key}")
        run = wandb.init(project=f"{description}_{model_arguments['algo_class_name']}_{dataset_key}",
                            name=name,
                            config={**{"algo_class": algo_class,
                                "dataset_key": dataset_key,}, **model_arguments},
                            reinit=True)

    total_train_time = 0
    start_time = time.time()

    if 'specific_evaluation_regime' in model_arguments:
        decipher_specific_evaluation_regime = RunIrregularEvaluationRegime(model_arguments['specific_evaluation_regime'],
                                                                                )
    
    end_run = False
    best_score = -1
    for epoch in range(model_arguments["epochs"]):
        print(f'\n\nEpoch: {epoch} - Running model: {model_arguments["algo_class_name"]}, dataset: {dataset_key}\n')
        epoch_save_path = os.path.join(model_dataset_path, f"{epoch}")
        
        if model_arguments["wandb"]:
            wandb.log({"epoch":epoch})

        start_training = time.time()
        algo_class.train_one_epoch()
        training_time = time.time() - start_training
        total_train_time += training_time

        save_model = model_arguments["save_model_every_n_epochs"] and (epoch+1)%model_arguments["save_model_every_n_epochs"]==0
        evaluate_model = model_arguments["evaluate_n_epochs"] and (epoch+1)%model_arguments["evaluate_n_epochs"]==0
        if model_arguments["save_heatmaps_n_epochs"]==0:
            save_heatmaps = True
        else:
            save_heatmaps = model_arguments["save_heatmaps_n_epochs"] and (epoch+1)%model_arguments["save_heatmaps_n_epochs"]==0

        evaluate_model = evaluate_model or (epoch+1)>=model_arguments["epochs"]
        
        if 'specific_evaluation_regime' in model_arguments:
            evaluate_model = decipher_specific_evaluation_regime(total_train_time, 
                                                                    epoch,)
        if 'hard_time_limit' in model_arguments:
            if (time.time()-start_time)>model_arguments['hard_time_limit']:
                end_run=True
                evaluate_model=True
        if 'train_time_limit' in model_arguments:
            if total_train_time>model_arguments['train_time_limit']:
                end_run=True
                evaluate_model=True
            

        if save_model or evaluate_model or save_heatmaps:
            start_pre_eval_processing = time.time()
            algo_class.pre_eval_processing()
            pre_eval_processing_time = time.time() - start_pre_eval_processing

        if save_model:
            model_save_path = os.path.join(epoch_save_path, "model")
            os.makedirs(os.path.join(model_save_path, f".{model_arguments['algo_class_name']}"), exist_ok=True)
            algo_class.save_model(model_save_path)
            print(f"Saved model to:\n{model_save_path}")


        if evaluate_model or save_heatmaps:
            evaluate_time_start = time.time()
            heatmap_set, image_score_set, targets_set, paths_set = algo_class.test()
            evaluate_time = time.time() - evaluate_time_start

            os.makedirs(epoch_save_path, exist_ok=True)

            train_time = total_train_time + pre_eval_processing_time
            save_path = os.path.join(epoch_save_path, "metrics.pk")

            metric_time_start = time.time()
            metric_results = run_metrics(metrics, 
                                         heatmap_set, 
                                         image_score_set, 
                                         targets_set, 
                                         paths_set,
                                         save_path,
                                         model_arguments["save_metric_meta_data"],
                                         )
            
            metric_time = time.time() - metric_time_start
            if model_arguments["wandb"]:
                to_log = {}
                for metric, metric_item in metric_results.items():
                    for method, method_item in metric_item[0].items():
                        to_log[f"{metric}_{method}"] = method_item
                wandb.log(to_log)
                wandb.log({"evaluation_epoch":epoch})
                wandb.log({"evaluation_time":evaluate_time})
                wandb.log({"metric_time":metric_time})
                
            reset_all()

            heatmap_save_path = os.path.join(epoch_save_path, "heatmaps.pk")

            description = {}
        
            description["dataset"] = dataset
            description["run_description"] = run_description
            description["model_description"] = model_arguments["model_description"]
            description["model_long_description"] = model_arguments["model_long_description"]

            #with open(os.path.join(epoch_save_path, f"dump_{epoch}.pk"), "wb") as f:
            #    pickle.dump((heatmap_set, image_score_set, targets_set, paths_set),f)
            #    print(f"Saved .pk dump to\n{os.path.join(epoch_save_path, f'dump_{epoch}.pk')}")
            if save_heatmaps:
                save_heatmaps_func(heatmap_set, 
                                   image_score_set, 
                                   targets_set, 
                                   paths_set, 
                                   heatmap_save_path, 
                                   description)

            data = {"time": run_time, 
                    "run_description": run_description, 
                    "model_description": model_arguments["model_description"],
                    "model_long_description": model_arguments["model_long_description"], 
                    "epoch": epoch, 
                    "training_transforms": model_arguments["training_transformation_string"],
                    "model": model_arguments["algo_class_name"],
                    "train_time": train_time,
                    "train_time_per_image": train_time/len(dataloader_train.dataset),
                    "inference_time": evaluate_time,
                    "inference_time_per_image": evaluate_time/(len(dataloader_regular_test.dataset)+len(dataloader_novel_test.dataset)),
                    "model_params": model_arguments["algo_class_name"]+str(model_arguments["model_params"]),
                    "dataset": dataset_key,
                    "metrics": {key: item["metric_results"] for key, item in metric_results.items()},
                    "batch_size": model_arguments["batch_size"]} 
            
            if model_arguments["wandb"] or model_arguments["save_best_preds"]:
                to_log = {}
                for metric, metric_item in metric_results.items():
                    for method, method_item in metric_item["metric_results"].items():
                        
                        to_log[f"{metric}_{method}"] = method_item

                        if metric==model_arguments["save_best_preds_metric"]:
                            if method_item>best_score:
                                best_score = method_item
                                best_results = metric_results
                                best_epoch   = epoch
                                if model_arguments["save_best_preds"]:
                                    print("Saving best heatmaps")
                                    save_heatmaps_func(heatmap_set, 
                                                        image_score_set,
                                                        targets_set, 
                                                        paths_set, 
                                                        os.path.join(model_dataset_path, 
                                                                     "best.pk"), 
                                                        description)



                if model_arguments["wandb"]:
                    wandb.log(to_log)
                    wandb.log({"train_time": train_time,
                                "train_time_per_image": train_time/len(dataloader_train.dataset),
                                "inference_time": evaluate_time,
                                "inference_time_per_image": evaluate_time/(len(dataloader_regular_test.dataset)+len(dataloader_novel_test.dataset)),
                                })
            
            print(data)
            with open(os.path.join(epoch_save_path, "data.json"), "w") as f:
                json.dump(data, f) 
        if end_run:
            if "hard_time_limit" in model_arguments:
                print(f"Run Times out:\nRun time:{time.time()-start_time} Time limit: {model_arguments['hard_time_limit']}")
            elif "train_time_limit":
                print(f"Run Times out:\nRun time:{time.time()-start_time} Time limit: {model_arguments['train_time_limit']}")

            break
    if model_arguments["wandb"]:
        run.finish()
    
    print(f"Best results @ epoch {best_epoch}")
    for key, item in best_results.items():
        print(f"{key}:")
        for heatmap_method, result in item["metric_results"].items():
            print(f"\t{heatmap_method}: {result}")

class RunIrregularEvaluationRegime:
    '''
    This allows a model to be evaluated at irregular and specific epoch and times after training
    e.g. 5 min, 15 min, 30 min, 1hr 2hr 4hr
    e.g. epoch 0, epoch 1, epoch 5, epoch 10
    e.g. add the following to the config dictionary
    'specific_evaluation_regime': {"epochs":[0, 1, 5, 10],
                                   "times": [5*60, 15*60, 30*60, 60*60, 60*60*2, 60*60*4,]}
    '''
    def __init__(self, specific_evaluation_regime): 
        self.specific_evaluation_regime = specific_evaluation_regime
        self.times_tested = []
    
    def __call__(self, train_time, epoch):

        if 'epochs' in self.specific_evaluation_regime:
            if epoch in self.specific_evaluation_regime["epochs"]:
                print(f"Logging due to epoch {epoch} in specific_evaluation_regime['epochs']")
                return True
        if 'times' in self.specific_evaluation_regime:
            for time_0, time_1 in zip(self.specific_evaluation_regime["times"][:-1], self.specific_evaluation_regime["times"][1:]):
                if train_time>time_0 and train_time<time_1 and time_0 not in self.times_tested:
                    print(f"Logging due to time {time_0} in specific_evaluation_regime['times']")
                    self.times_tested.append(time_0)
                    return True
        return False

def evaluate_algo(algo_class, 
                    algo_name,
                    dataset_key,
                    metrics,
                    run_time,
                    epoch,
                    model_dataset_path,
                    run_description = "",
                    **model_arguments,
                    ):
         
    os.makedirs(model_dataset_path, exist_ok=True)
    dataset = load_dataset(dataset_key)
    dataloader_train, dataloader_regular_test, dataloader_novel_test = create_dataloaders(                                                 
                                                                            pre_pro_transforms=model_arguments["pre_pro_transforms"], 
                                                                            training_transformations=model_arguments["training_transformations"],
                                                                            identical_transforms=model_arguments["identical_transforms"],
                                                                            batch_size = model_arguments["batch_size"],
                                                                            shuffle = True,
                                                                            device = model_arguments["device"], 
                                                                            unnormalise = make_unnormalise(mean=dataset['mean'], 
                                                                                                           std=dataset['std']),
                                                                            test_batch_size = model_arguments["test_batch_size"],
                                                                            **dataset)

    algo_class.enter_dataloaders(dataloader_train, 
                                 dataloader_regular_test, 
                                 dataloader_novel_test)
    
    heatmap_set, image_score_set, targets_set, paths_set = algo_class.test()

    epoch_save_path = os.path.join(model_dataset_path, f"{epoch}")

    os.makedirs(epoch_save_path, exist_ok=True)

    save_path = os.path.join(epoch_save_path, "metrics.pk")
    metric_results = run_metrics(metrics, 
                                    heatmap_set, 
                                    image_score_set, 
                                    targets_set, 
                                    paths_set,
                                    save_path,
                                    )

    heatmap_save_path = os.path.join(epoch_save_path, "heatmaps")

    description = {}

    description["dataset"] = dataset
    description["description"] = run_description

    if model_arguments["save_heatmaps"]:
        save_heatmaps_func(heatmap_set, image_score_set, targets_set, paths_set, heatmap_save_path, description)

    data = {"time": run_time, 
            "run_description": run_description, 
            "model_description": model_arguments["model_description"],
            "model_long_description": model_arguments["model_long_description"], 
            "epoch": epoch, 
            "training_transforms": model_arguments["training_transformation_string"],
            "model": algo_name,
            "model_params": model_arguments,
            "dataset": dataset_key,
            "metrics": metric_results,
            "batch_size": model_arguments["batch_size"]} 

    with open(os.path.join(epoch_save_path, "data.json"), "w") as f:
        json.dump(data, f) 

def get_metrics(keys, heatmap_set, image_score_set, targets_set, paths_set, save_metric_meta_data):
    to_save = {}
    metric_time_taken = {}
    for metric in keys:
        start = time.time()
        scores, meta = metric_list[metric](heatmap_set, image_score_set, targets_set, paths_set,)
        if save_metric_meta_data:
            to_save[metric] = {"metric_results": scores, "metric_data": meta}
        else:
            to_save[metric] = {"metric_results": scores}
            
        metric_time_taken[metric] = time.time() - start
    return to_save, metric_time_taken

def run_metrics(metrics, 
                heatmap_set, image_score_set, targets_set, paths_set,
                save_path=None, 
                save_metric_meta_data=True, 
                print_=True):
    
    #the below code can be used for debugging
    #with open(os.path.join(os.path.split(save_path)[0], "raw_dump_new.pk"), "wb") as f:
    #   pickle.dump((heatmap_set, image_score_set, targets_set, paths_set),f)
    #print(f"Saved file to:\n{os.path.join(os.path.split(save_path)[0], 'raw_dump_new.pk')}")
    #return {}

    if metrics[0]=="all":
        metrics = list(metric_list.keys())
    
    reset_all()
    import time
    
    # this allows the metrics to be ran in parallel, 
    # however on some systems this can cause issues
    # change as you see fit
    if os.name=="nt":
        metric_async = True
    else:
        metric_async = False

    try:
        if metric_async:
            metric_start = time.time()
            
            from functools import partial
            func = partial(get_metrics, heatmap_set=heatmap_set, 
                            image_score_set=image_score_set, 
                            targets_set=targets_set, 
                            paths_set=paths_set,
                            save_metric_meta_data=save_metric_meta_data)

            # multiprocess_split has the list of keys already split
            with Pool(3) as pool:
                out = pool.map(func, metric_key_list_async)

            to_save = {**out[0][0], **out[1][0], **out[2][0]}
            metric_time_taken = {**out[0][1], **out[1][1], **out[2][1]}
            async_metric_time = time.time() - metric_start
            print(f"\nAsync metric time taken: {async_metric_time}\n")
        async_failed = False
    except:
        async_failed = True


    if not metric_async or async_failed:
        # to_save = {}
        # metric_time_taken = {}
        
        to_save, metric_time_taken = get_metrics(metrics, 
                                                 heatmap_set, 
                                                 image_score_set, 
                                                 targets_set, 
                                                 paths_set, 
                                                 save_metric_meta_data)
        # for metric in metrics:
        #     start = time.time()
        #     scores, meta = metric_list[metric](heatmap_set, image_score_set, targets_set, paths_set,)
        #     if save_metric_meta_data:
        #         to_save[metric] = {"scores": scores, "meta": meta}
        #     else:
        #         to_save[metric] = {"scores": scores}

        #     metric_time_taken[metric] = time.time() - start
    
    if print_:
        for key, item in to_save.items():
            print(f"{key}:")
            for heatmap_method, result in item["metric_results"].items():
                print(f"\t{heatmap_method}: {result}")

        print("\nMetric time taken:\n")
        for metric, time_taken in metric_time_taken.items():
            print(f"{metric}: {time_taken:.2g}s")

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(to_save, f)
            print(f"Saved metrics to path:\n{save_path}")
        else:
            print("Not saved the metrics results")
    return to_save

def save_heatmaps_func(heatmap_set, 
                       image_score_set, 
                       targets_set, 
                       paths_set, 
                       heatmap_save_path, 
                       description):

    with open(heatmap_save_path, "wb") as f:
        pickle.dump((heatmap_set, image_score_set, targets_set, paths_set), f)

    print(f"Saved predictions in {heatmap_save_path}")
    with open(os.path.join(os.path.split(heatmap_save_path)[0], "description.json"), "w") as f:
        json.dump(description, f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

