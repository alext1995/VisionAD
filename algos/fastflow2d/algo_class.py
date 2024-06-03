'''
Based on code found here: https://github.com/gathierry/FastFlow
And based on the the paper: 
https://arxiv.org/abs/2111.07677
@misc{https://doi.org/10.48550/arxiv.2111.07677,
  doi = {10.48550/ARXIV.2111.07677},
  url = {https://arxiv.org/abs/2111.07677},
  author = {Yu, Jiawei and Zheng, Ye and Wang, Xiang and Li, Wei and Wu, Yushuang and Zhao, Rui and Wu, Liwei},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
'''
from algos.fastflow2d.fastflow import FastFlow
from algos.model_wrapper import ModelWrapper
import torch
import numpy as np
import os
import json
from tqdm import tqdm

class WrapperFastFlow2d(ModelWrapper):
    def  __init__(self, load_model_params=True, **params):
        super(ModelWrapper, self).__init__()
        if load_model_params:
            self.load_model_params(**params)
        self.default_segmentation = "anomaly_map_channel_mean"
        self.default_score = "loss_channel_mean_pixel_mean"
        
    def load_model_params(self, **params):
        self.__name__ = "fastflow"
        self.params = params
        self.model = FastFlow(**params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=1e-3, 
                                          weight_decay=1e-5
                                          )
       
        self.device = params["device"]
        self.model.to(self.device)
        print("params['device']: ", params["device"])

    def train_one_epoch(self):
        self.model.train()
        self.model.training = True
        for step, (data, _, _) in tqdm(enumerate(self.dataloader_train), total=self.len_dataloader_train):
            # forward
            data = data.to(self.device)
            ret = self.model(data)
            loss = ret["loss"]
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if "break_every_epoch" in self.params and self.params["break_every_epoch"]:
                break

    def pre_eval_processing(self):
        pass
    
    def save_model(self, location):
        self.save_params(location)

        for ind, model_item in enumerate(self.model.nf_flows):
            torch.save(model_item.state_dict(), os.path.join(location, f'nf_weights_{ind}.pth'))
        torch.save(self.model.norms.state_dict(), os.path.join(location, f'norms_weights.pth'))
        
    def load_model(self, location, **kwargs):
        
        params = self.load_params(location)
        self.load_model_params(**params)

        for ind, model_item in enumerate(self.model.nf_flows):
            model_item.load_state_dict(torch.load(os.path.join(location, f'nf_weights_{ind}.pth')))    
        self.model.norms.load_state_dict(torch.load(os.path.join(location, f'norms_weights.pth')))    

    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        self.model.eval()
        self.model.training = False
        with torch.no_grad():
            for ind, (data, _) in tqdm(enumerate(dataloader), total=len_dataloader):
                data = data.to(self.device)
                
                ret = self.model(data)
             
                if ind==0:
                  
                    segmentations_sets = {}
                    for key, item in ret['segmentations'].items():
                        segmentations_sets[key] = item.cpu().detach()

                    scores_sets = {}
                    for key, item in ret['scores'].items():
                        scores_sets[key] = item.cpu().detach()

                else:
                    for key, item in ret['segmentations'].items():
                        segmentations_sets[key] = torch.vstack([segmentations_sets[key], item.cpu().detach()])
                    
                    for key, item in ret['scores'].items():
                        scores_sets[key] = torch.concat([scores_sets[key], item.cpu().detach()])
            
        heatmap_set = segmentations_sets

        image_score_set = scores_sets
        
        return heatmap_set, image_score_set