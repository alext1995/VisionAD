'''
Based on code found here: https://github.com/gudovskiy/cflow-ad
And based on the the paper: 
https://arxiv.org/pdf/2206.04325.pdf
@inproceedings{Gudovskiy_2022_WACV,
    author    = {Gudovskiy, Denis and Ishizaka, Shun and Kozuka, Kazuki},
    title     = {{CFLOW-AD}: Real-Time Unsupervised Anomaly Detection With Localization via Conditional Normalizing Flows},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {98-107}
}
'''
from algos.cflow.cflow import CFLOW
from algos.model_wrapper import ModelWrapper
import numpy as np

class WrapperCFLOW(ModelWrapper):
    def __init__(self, load_model_params=True, **kwargs):
        super(ModelWrapper, self).__init__()
        if load_model_params:
            self.load_model_params(**kwargs)
        self.default_segmentation = "score_map_standard"
        self.default_score = "max"
        
    def load_model_params(self, **params):
        self.__name__ = "CFLOW"
        self.params = params
        self.model = CFLOW(**params)
    
    def train_one_epoch(self):
        self.model.train_one_epoch(self.dataloader_train)
    
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        return self.model.eval_outputs_dataloader(dataloader, len_dataloader)
    
    def pre_eval_processing(self):
        pass
    
    def save_model(self, location):
        self.save_params(location)
        self.model.save_model(location) 
    
    def load_model(self, location, **kwargs):
        params = self.load_params(location)
        self.load_model_params(**kwargs)
        self.model.load_model(location) 
    