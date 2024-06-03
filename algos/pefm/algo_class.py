'''
Based on code found here: https://github.com/smiler96/PFM-and-PEFM-for-Image-Anomaly-Detection-and-Segmentation
And based on the the papers: 
@ARTICLE{PFM,
    author={Wan, Qian and Gao, Liang and Li, Xinyu and Wen, Long},
    journal={IEEE Transactions on Industrial Informatics},
    title={Unsupervised Image Anomaly Detection and Segmentation Based on Pre-trained Feature Mapping},
    year={2022},
    volume={},
    number={},
    pages={},
    doi={10.1109/TII.2022.3182385}
} 
@INPROCEEDINGS{PEFM,
    author={Wan, Qian and Cao YunKang and Gao, Liang and Shen Weiming and Li, Xinyu},
    booktitle={2022 IEEE 18th International Conference on Automation Science and Engineering (CASE)}, 
    title={Position Encoding Enhanced Feature Mapping for Image Anomaly Detection}, 
    year={2022},
    volume={},
    number={},
    pages={},
    doi={}
  }
'''
import os
from algos.model_wrapper import ModelWrapper
from algos.pefm.pefm import PEFM, PFM
import json

class WrapperPFM(ModelWrapper):
    def  __init__(self, load_model_params=True, **params):
        super(ModelWrapper, self).__init__()
        if load_model_params:
            self.load_model_params(**params)
        self.default_segmentation = "Original_method"
        self.default_score = "Original_method_score_max"

        
    def load_model_params(self, **params):
        self.__name__ = "PFM"
        self.params = params
        self.model = PFM(**params)
        
    def train_one_epoch(self):
        self.model.train_one_epoch(self.dataloader_train)
    
    def eval_outputs_dataloader(self, dataloader, len_dataloader, verbose=False, limit=None):
        return self.model.eval_outputs_dataloader(dataloader)
    
    def pre_eval_processing(self):
        self.model.pre_eval_processing(self.dataloader_train)

    def save_model(self, location):
        self.save_params(location)
        self.model.save_model(location)

    def load_model(self, location, **kwargs):
        params = self.load_params(location)
        self.load_model_params(**params)
        self.model.load_model(location)
    
class WrapperPEFM(ModelWrapper):
    def  __init__(self, load_model_params=True, **params):
        super(ModelWrapper, self).__init__()
        if load_model_params:
            self.load_params(**params)
        self.default_segmentation = "Original_method"
        self.default_score = "Original_method_score_max"

        
    def load_params(self, **params):
        self.__name__ = "PEFM"
        self.params = params
        self.model = PEFM(**params)
        
    def train_one_epoch(self):
        self.model.train_one_epoch(self.dataloader_train)
    
    def eval_outputs_dataloader(self, dataloader, verbose=False, limit=None):
        return self.model.eval_outputs_dataloader(dataloader)
    
    def pre_eval_processing(self):
        self.model.pre_eval_processing(self.dataloader_train)

    def save_model(self, location):
        os.makedirs(location, exist_ok=True)
        params_path = os.path.join(location, "model_params.json")
        with open(params_path, "w") as f:
            json.dump(self.params, f)
            
        self.model.save_model(location)

    def load_model(self, location, **kwargs):
        params_path = os.path.join(location, "model_params.json")
        with open(params_path, "r") as f:
            params = json.load(f)
        
        self.load_params(**params)
        self.model.load_model(location)