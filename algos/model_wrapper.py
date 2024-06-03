from abc import abstractmethod
import os 
import json
import numpy as np
import pickle
import torch

class ModelWrapper:
    def __init__(self):
        pass

    def enter_dataloaders(self, dataloader_train, 
                                dataloader_regular_test, 
                                dataloader_novel_test,
                                ):
        self.dataloader_train = dataloader_train 
        self.dataloader_regular_test = dataloader_regular_test 
        self.dataloader_novel_test = dataloader_novel_test

        self.len_dataloader_train = len(dataloader_train) 
        self.len_dataloader_regular_test = len(dataloader_regular_test) 
        self.len_dataloader_novel_test = len(dataloader_novel_test)

        paths_regular = []
        for _, path_batch, _ in self.dataloader_regular_test:
            paths_regular += [path for path in path_batch]
        paths_novel = []
        for _, _, path_batch, _ in self.dataloader_novel_test:
            paths_novel += [path for path in path_batch]    
        self.paths_required = {"regular_test": paths_regular, "novel_test": paths_novel}

    def run_error_test(self, heatmap_set, image_score_set, targets_set, paths_set):
        '''
        Error checking to make sure no algorithms cheat by combining batches into one image etc
        This makes sure all the paths from the dataloader are present in the outputted paths
        And the number of scores and heatmaps match
        '''
        ## checking paths here
        for path_key, path_set_key in zip(['regular_test', 'novel_test'], 
                                           ['paths_regular', 'paths_novel']):
            found_paths = paths_set[path_set_key] 
            desired_paths = self.paths_required[path_key]
            
            string = f"{path_key}: Paths required and paths outputted should be the same size"
            string += f" len(desired_paths)={len(desired_paths)}, len(outputted_paths)={len(found_paths)}"
            assert len(desired_paths)==len(found_paths), string
            string = f"{path_key}: The set of desired_paths and outputted paths are different."
            string += f" This means the class is doing something very weird via mixing up the paths somehow"
            string += f" This means there is a different between the paths entered into the class and those that come out"
            assert set(found_paths)==set(desired_paths), string
            
        ## checking the targets here
        for path_set_key, target_key in zip(['novel_test'], 
                                            ['targets_novel']):    
            targets = targets_set[target_key]
            path_set = self.paths_required[path_set_key]
            if targets.shape[0]!=len(path_set):
                print(f"Method: {path_set_key} has not outputed the right number of target images.")
                print(f"method.shape[0]: {targets.shape[0]}, num of images required: {len(path_set)}.")
                print(f"This means there is a bug in the code which outputs this method")
                print(f"The most likely cause of this bug is averaging over the batch to create a combined score over a batch")
                print(f"as opposed to a ")
                print(f"Removing data from {path_set_key} as it will cause invalid results")
                string = f"Targets of {target_key}.shape: {targets.shape[0]}. There are {len(path_set)} paths"
                string +=f" in the {target_key} test set"
                raise AssertionError("Incorrect implementation of taking targets from eval_dataloader class\n"+string)
                
        from collections import defaultdict
        image_score_set_out = {}
        for method, _ in image_score_set["image_score_set_regular"].items():

            passed = True
            for path_set_key, loss_key in zip(['regular_test', 'novel_test'], 
                                            ['image_score_set_regular', 'image_score_set_novel'],
                                            ):    
                path_set = self.paths_required[path_set_key]
                method_loss = image_score_set[loss_key][method]  
                
                if method_loss.shape[0]!=len(path_set):
                    print(f"\nMethod: {method} has not outputed the right number of loss[{loss_key}] images.")
                    print(f"loss[{loss_key}] .shape[0]: {method_loss.shape[0]}, num of images required: {len(path_set)}.")
                    print(f"This means there is a bug in the code which outputs this method")
                    print(f"The most likely cause of this bug is averaging over the batch to create a combined score over a batch")
                    print(f"as opposed to a creating a score per image")
                    print(f"Removing data from {method} as it will cause invalid results\n\n")
                    passed = False
            if passed:
                for loss_key in ['image_score_set_regular','image_score_set_novel']:    
                    if loss_key not in image_score_set_out:
                        image_score_set_out[loss_key] = {}
                    image_score_set_out[loss_key][method] = image_score_set[loss_key][method]  

        heatmap_set_out = {}
        for method, _ in heatmap_set["heatmap_set_regular"].items():
            passed = True
            for path_set_key, heatmap_key in zip(['regular_test', 'novel_test'], 
                                            ['heatmap_set_regular', 'heatmap_set_novel'],
                                            ):    
                path_set = self.paths_required[path_set_key]
                method_heatmaps = heatmap_set[heatmap_key][method]  
                
                if method_heatmaps.shape[0]!=len(path_set):
                    print(f"\nMethod: {method} has not outputed the right number of heatmap[{heatmap_key}] images.")
                    print(f"heatmap[{heatmap_key}].shape[0]: {method_heatmaps.shape[0]}, num of images required: {len(path_set)}.")
                    print(f"This means there is a bug in the code which outputs this method")
                    print(f"The most likely cause of this bug is averaging over the batch to create a combined score over a batch")
                    print(f"as opposed to a creating a score per image")
                    print(f"Removing {method} from data as it will cause invalid results\n\n")
                    passed = False
            if passed:
                for heatmap_key in ['heatmap_set_regular','heatmap_set_novel']:    
                    if heatmap_key not in heatmap_set_out:
                        heatmap_set_out[heatmap_key] = {}
                    heatmap_set_out[heatmap_key][method] = heatmap_set[heatmap_key][method]

        return heatmap_set_out, image_score_set_out, targets_set, paths_set

    def create_no_target_dataloader(self, dataloader, data_holder, regular):
        '''
        This occuldes the targets and paths from dataloaders for the algorithms.
        This is for a few reasons:
            1. Each algorithm should not see the targets/answers or the paths
            2. The code to summate the targets/paths should not need to be repeated across each algorithm
        '''
        
        for ind, items in enumerate(dataloader):
            if regular:
                image_batch, paths_batch, callback_info_batch = items
            else:
                image_batch, targets_batch, paths_batch, callback_info_batch = items

            if ind==0:
                if not regular:
                    data_holder["targets"] = [targets_batch.cpu().detach()]
                data_holder["paths"] = [path for path in paths_batch]
            else:
                if not regular:
                    data_holder["targets"].append(targets_batch.cpu().detach())
                data_holder["paths"] += [path for path in paths_batch]
            yield image_batch, callback_info_batch

    def test(self,):  
        self.data_holder_regular_test = {}
        no_target_dataloader_regular_test = self.create_no_target_dataloader(self.dataloader_regular_test, 
                                                          self.data_holder_regular_test, 
                                                          regular=True)
        heatmap_set_regular, image_score_set_regular = self.eval_outputs_dataloader(no_target_dataloader_regular_test, 
                                                                             self.len_dataloader_regular_test)
        paths_regular = self.data_holder_regular_test["paths"] 

        self.data_holder_novel_test = {}
        no_target_dataloader_novel_test = self.create_no_target_dataloader(self.dataloader_novel_test, 
                                                                    self.data_holder_novel_test, 
                                                                    regular=False)
        heatmap_set_novel, image_score_set_novel = self.eval_outputs_dataloader(no_target_dataloader_novel_test, 
                                                                         self.len_dataloader_novel_test)
        targets_novel, paths_novel = torch.concatenate(self.data_holder_novel_test["targets"]), self.data_holder_novel_test["paths"] 

        image_score_set = {"image_score_set_regular": image_score_set_regular, "image_score_set_novel": image_score_set_novel}
        heatmap_set = {"heatmap_set_regular": heatmap_set_regular, "heatmap_set_novel": heatmap_set_novel}

        for heatmap_group in [heatmap_set_regular, heatmap_set_novel]:
            for segmentation_method, result in heatmap_group.items():
                if len(result.shape)==3:
                    heatmap_group[segmentation_method] = result[:,None,:,:]

        # reshape the targets to be the standard 256, 256 shape
        if not targets_novel.shape[-2:]==(256, 256):
            targets_novel = torch.nn.functional.interpolate(targets_novel, 
                                                            size=256, 
                                                            mode='bilinear', 
                                                            align_corners=True)

        targets_set = {"targets_novel": targets_novel}
        paths_set = {"paths_regular": paths_regular, "paths_novel": paths_novel}

        heatmap_set, image_score_set, targets_set, paths_set = self.run_error_test(heatmap_set, 
                                                                            image_score_set, 
                                                                            targets_set, 
                                                                            paths_set)
        # with open(os.path.join(os.path.split(save_path)[0], "raw_dump_after.pk"), "wb") as f:
        #     pickle.dump((heatmap_set, image_score_set, targets_set, paths_set),f)
        return heatmap_set, image_score_set, targets_set, paths_set

    def save_params(self, location):
        '''
        Saving the params to create the model is identical each time
        '''
        os.makedirs(location, exist_ok=True)
        params_path = os.path.join(location, "model_params.json")
        with open(params_path, "w") as f:
            json.dump(self.params, f)

    def load_params(self, location):
        '''
        Loading the params to create the model is identical each time
        '''
        params_path = os.path.join(location, "model_params.json")
        with open(params_path, "r") as f:
            return json.load(f)
        
    @abstractmethod
    def pre_eval_processing(self):
        '''
        This is for any processing which needs to be done using the training data,
        but is not required every epoch, and is required for testing.
        Examples of this could be calculating the norm of the training features, 
        saving encoded features of the train set for comparison with test features, etc.
        This must be done here, as the wrapper class does not have access to the train dataloader in evaluation 
        Sometimes this opeartion can be expensive, therefore it is a waste to do it every epoch.
        Pre_eval_processing is called before any evaluation or save.
        If this is not needed, create the function in the algo_class with a pass'''
        pass
        
    @abstractmethod
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        '''
        This method needs the dataloader and the len of the dataloader.
        The len is passed because the dataloader is a generator, and a len(dataloader)
        will throw an error 
        '''
        pass

    @abstractmethod
    def train_one_epoch(self,):
        pass    

    @abstractmethod
    def save_model(self, location):
        pass
    
    @abstractmethod
    def load_model(self, location, **kwargs):
        pass
