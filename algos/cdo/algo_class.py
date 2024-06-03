'''
Based on code found here: https://github.com/caoyunkang/CDO
And based on the the paper: 
https://arxiv.org/ftp/arxiv/papers/2302/2302.08769.pdf
@ARTICLE{10034849,
  author={Cao, Yunkang and Xu, Xiaohao and Liu, Zhaoge and Shen, Weiming},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization}, 
  year={2023},
  volume={},
  number={},
  pages={1-10},
  doi={10.1109/TII.2023.3241579}}
'''
from algos.cdo.synthetic_augmentation import add_synthetic_anomaly_pixels
from algos.model_wrapper import ModelWrapper
import torch
from algos.cdo.cdo_model import CDOModel
from tqdm import tqdm
from functools import partial
import numpy as np
import os

class WrapperCDO(ModelWrapper):
    def  __init__(self, load_model_params=True, **params):
        super(ModelWrapper, self).__init__()
        if load_model_params:
            self.load_model_params(**params)
        self.default_segmentation = "normal"
        self.default_score = "max"
        
    def load_model_params(self, **params):
        self.__name__ = "cdo"
        self.params = params
        self.model = CDOModel(**params)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, 
                                                  self.model.parameters()), 
                                            lr=params["lr"],
                                            weight_decay=params["weight_decay"])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        self.device = params["device"]
        self.model.to(self.device)
        print("params['device']: ", params["device"])
        self.epoch = 0
        
    def train_one_epoch_prev_method(self, original_dataloader):
        self.model.train_mode()

        for (data, gt, _, _) in original_dataloader:
        
            # forward
            data = data.to(self.device)
            outputs = self.model(data)
            loss = self.model.cal_loss(outputs['FE'], outputs['FA'], mask=gt)
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.lr_scheduler.step()

    def train_one_epoch(self):
        self.dataloader_train.dataset.pre_normalisation_transform = partial(add_synthetic_anomaly_pixels,
                                                                            input_size=self.params["input_size"],
                                                                            skip_bkg=self.params["skip_bkg"])

        ii = 0
        for (data, _, gt) in tqdm(self.dataloader_train):
            ii+=1
            if self.epoch==0:
                break
            data = data.to(self.device)
            outputs = self.model(data)
            loss = self.model.cal_loss(outputs['FE'], outputs['FA'], mask=gt)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if "break_every_epoch" in self.params and self.params["break_every_epoch"] and ii>5:
                print("breaking")
                break

        self.lr_scheduler.step()
        self.epoch+=1
    def pre_eval_processing(self):
        pass
    
    def save_model(self, location):
        self.save_params(location)
        self.model.save(os.path.join(location, "model.pt"))
        
    def load_model(self, location, **kwargs):
        params = self.load_params(location)
        self.load_model_params(**params)
        self.model.load(os.path.join(location, "model.pt"))  

    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        self.model.eval_mode()
        with torch.no_grad():
            for ind, (data, _) in tqdm(enumerate(dataloader), total=len_dataloader):
                data = data.to(self.device)
                
                outputs = self.model(data)
                amaps   = self.model.cal_am(**outputs)
                amaps   = np.array(amaps)

                ret = {}   
                ret['segmentations'] = {}
                ret['scores']        = {}
                ret['segmentations']["normal"] = amaps
                ret['scores']["mean"]          = amaps.mean(-1).mean(-1)
                ret['scores']["max"]           = amaps.max(-1).max(-1)

                if ind==0:
                    segmentations_sets = {}
                    for key, item in ret['segmentations'].items():
                        segmentations_sets[key] = item

                    scores_sets = {}
                    for key, item in ret['scores'].items():
                        scores_sets[key] = item

                else:
                    for key, item in ret['segmentations'].items():
                        segmentations_sets[key] = np.vstack([segmentations_sets[key], item])
                    
                    for key, item in ret['scores'].items():
                        scores_sets[key] = np.concatenate([scores_sets[key], item])
                        
        heatmap_set = segmentations_sets
        image_score_set = scores_sets 
        return heatmap_set, image_score_set

