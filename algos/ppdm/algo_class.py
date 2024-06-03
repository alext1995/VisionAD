from algos.model_wrapper import ModelWrapper
import numpy as np
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import os
import itertools
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import torch
import numpy as np
import random
import os
import cv2
from algos.ppdm.resnet import wide_resnet50_2
from algos.ppdm.de_resnet import de_wide_resnet50_2
from torch.nn import functional as F
from algos.ppdm.evaluation import cal_anomaly_map, gaussian_filter
   
def loss_function(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

class WrapperPPDM(ModelWrapper):
    def __init__(self, load_params=True, **kwargs):
        super(ModelWrapper, self).__init__()
        if load_params:
            self.load_model_params(**kwargs)
        self.default_segmentation = "method_1"
        self.default_score = "method_1"
        
    def load_model_params(self, **params):
        self.__name__ = "PPDM"
        self.params = params
        self.device = self.params["device"]
        
        encoder, bn, offset = wide_resnet50_2(self.device, 
                                              pretrained=True, 
                                              vq=self.params["vq"], 
                                              gamma=self.params["gamma"])
        encoder = encoder.to(self.device)
        bn = bn.to(self.device)
        offset = offset.to(self.device)
        decoder = de_wide_resnet50_2(pretrained=False)
        decoder = decoder.to(self.device)
        encoder.eval()
            
        optimizer = torch.optim.AdamW(list(offset.parameters())+list(decoder.parameters())+list(bn.parameters()), 
                                      lr=params["learning_rate"], betas=(0.5,0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params["epochs"])
        
        self.encoder = encoder
        self.bn = bn
        self.offset = offset
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_one_epoch(self):
        self.offset.train()
        self.bn.train()
        self.decoder.train()
            
        for image, _, _ in self.dataloader_train:
            img = image.to(self.device)
            _, img_, offset_loss = self.offset(img)
            
            inputs = self.encoder(img_)
            vq, vq_loss = self.bn(inputs)
            outputs = self.decoder(vq)
            main_loss = loss_function(inputs, outputs)
            loss = main_loss + offset_loss + vq_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
          
        self.scheduler.step()        
        
    @torch.no_grad()
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        self.encoder.eval()
        self.bn.eval()
        self.decoder.eval()
        
        anomaly_maps = []
        anomaly_os1s = []
        anomaly_os2s = []
        
        for (image, _) in dataloader:
            for img in image:
                img = img[None].to(self.device)
                img_, offset1, offset2, offset1_, offset2_, grid1_, grid2_, grid2 = self.offset(img, test=True)
                inputs = self.encoder(img_)
                vq, _ = self.bn(inputs)
                outputs = self.decoder(vq)

                anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
                anomaly_map = F.grid_sample(F.grid_sample(torch.from_numpy(anomaly_map).float().to(self.device)[None, None], 
                                                        grid2_, 
                                                        align_corners=True), 
                                            grid1_, 
                                            align_corners=True).cpu().numpy()
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                #anomaly_os1 = F.grid_sample(offset1[None].detach(), grid2, align_corners=True).to('cpu').numpy()
                #anomaly_os1 = gaussian_filter(anomaly_os1, sigma=4)
                #anomaly_os2 = offset2[None].detach().to('cpu').numpy()
                #anomaly_os2 = gaussian_filter(anomaly_os2, sigma=4)
                anomaly_os1 = (F.grid_sample(offset1[None].detach(), grid2, align_corners=True)+offset2[None].detach()).to('cpu').numpy()
                anomaly_os1 = gaussian_filter(anomaly_os1, sigma=4)
                anomaly_os2 = (F.grid_sample(offset2_[None].detach(), grid1_, align_corners=True)+offset1_[None].detach()).to('cpu').numpy()
                anomaly_os2 = gaussian_filter(anomaly_os2, sigma=4)

                anomaly_maps.append(anomaly_map)
                anomaly_os1s.append(anomaly_os1)
                anomaly_os2s.append(anomaly_os2)
         
        anomaly_maps = np.concatenate(anomaly_maps)
        anomaly_os1s = np.concatenate(anomaly_os1s)
        anomaly_os2s = np.concatenate(anomaly_os2s)
        
        score_maps = anomaly_maps.max(axis=-1).max(axis=-1)
        score_os1s = anomaly_os1s.max(axis=-1).max(axis=-1)
        score_os2s = anomaly_os2s.max(axis=-1).max(axis=-1)
        
        a_pixel_ret  = {"method_1":anomaly_maps, 
                        "method_2":anomaly_os1s, 
                        "method_3":anomaly_os2s}
        a_image_ret  = {"method_1":score_maps, 
                        "method_2":score_os1s, 
                        "method_3":score_os2s}
        
        return a_pixel_ret, a_image_ret
    
    def pre_eval_processing(self):
        pass
        
    def save_model(self, location):
        self.save_params(location)
        torch.save(self.offset.state_dict(), os.path.join(location, "offset.pk"))
        torch.save(self.bn.state_dict(), os.path.join(location, "bn.pk"))
        torch.save(self.decoder.state_dict(), os.path.join(location, "decoder.pk"))

    def load_model(self, location):
        params = self.load_params(location)
        self.load_model_params(**params)
        self.offset.load_state_dict(torch.load(os.path.join(location, "offset.pk")))
        self.bn.load_state_dict(torch.load(os.path.join(location, "bn.pk")))
        self.decoder.load_state_dict(torch.load(os.path.join(location, "decoder.pk")))
        

        
