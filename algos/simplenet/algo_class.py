from algos.model_wrapper import ModelWrapper
import numpy as np
from torchvision import transforms
from torch import nn
from torchvision.datasets import ImageFolder
import os
import itertools
from torch.utils.data import Dataset, DataLoader
import torch
from algos.simplenet.simplenet import SimpleNet
import algos.simplenet.backbones as backbones

class WrapperSimpleNet(ModelWrapper):
    def __init__(self, load_params=True, **kwargs):
        super(ModelWrapper, self).__init__()
        if load_params:
            self.load_model_params(**kwargs)
        self.default_segmentation = "method_1"
        self.default_score = "method_1"
        
    def load_model_params(self, **params):
        self.__name__ = "SimpleNet"
        self.params = params
        self.device = self.params["device"]
        self.epoch = 0
        self.simplenet = SimpleNet(self.device)
        
        backbone = backbones.load(self.params["backbone"])
            
        self.simplenet.load(backbone = backbone,
                            layers_to_extract_from = self.params["layers_to_extract_from"],
                            device = self.params["device"],
                            input_shape = (3, self.params["input_size"], self.params["input_size"]),
                            pretrain_embed_dimension = self.params["pretrain_embed_dimension"], # 1536
                            target_embed_dimension = self.params["target_embed_dimension"], # 1536
                            patchsize = self.params["patchsize"], # 3
                            patchstride = self.params["patchstride"], 
                            embedding_size = self.params["embedding_size"], # 256
                            meta_epochs = self.params["epochs"], # 40
                            aed_meta_epochs = self.params["aed_meta_epochs"],
                            gan_epochs = self.params["gan_epochs"], # 4
                            noise_std = self.params["noise_std"],
                            mix_noise = self.params["mix_noise"],
                            noise_type = self.params["noise_type"],
                            dsc_layers = self.params["dsc_layers"], # 2
                            dsc_hidden = self.params["dsc_hidden"], # 1024
                            dsc_margin = self.params["dsc_margin"], # .5
                            dsc_lr = self.params["dsc_lr"],
                            train_backbone = self.params["train_backbone"],
                            auto_noise = self.params["auto_noise"],
                            cos_lr = self.params["cos_lr"],
                            lr = self.params["lr"],
                            pre_proj = self.params["pre_proj"], # 1
                            proj_layer_type = self.params["proj_layer_type"])
        
    def train_one_epoch(self,):
        if not self.dataloader_train.batch_size==1:
            raise NotImplementedError("The batchsize is not set to 1. Simplenet requires the batchsize is set to 1 in the config file.")

        self.simplenet.forward_modules.train()
        self.simplenet._train_one_epoch(self.dataloader_train)    
        self.epoch+=1
        
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        self.simplenet.forward_modules.eval()
        all_scores = []
        all_maps   = []
        
        for (image, _) in dataloader:
            for im in image:
                scores, pred_maps, features = self.simplenet.predict(im[None])
                all_scores.append(scores)
                all_maps.append(pred_maps)

        a_pixel_ret = {"method_1":np.array(all_maps),}
        a_image_ret = {"method_1":np.array(all_scores),}
        return a_pixel_ret, a_image_ret
    
    def pre_eval_processing(self):
        pass
        
    def save_model(self, location):
        self.save_params(location)
        filepath = os.path.join(location, "model.pt")
        torch.save(self.simplenet.state_dict(), filepath)
        
    def load_model(self, location):
        params = self.load_params(location)
        self.load_model_params(**params)
        filepath = os.path.join(location, "model.pt")
        self.simplenet.load_state_dict(torch.load(filepath))
        
  
