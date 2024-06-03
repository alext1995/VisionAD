'''
Based on code found here: https://github.com/hq-deng/RD4AD
And based on the the papers: 
@InProceedings{Deng_2022_CVPR,
author    = {Deng, Hanqiu and Li, Xingyu},
title     = {Anomaly Detection via Reverse Distillation From One-Class Embedding},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month     = {June},
year      = {2022},
pages     = {9737-9746}}
'''
import torch
import numpy as np
import os
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from algos.reverse_distillation.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from algos.reverse_distillation.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from algos.model_wrapper import ModelWrapper

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

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

class WrapperReverseDistillation(ModelWrapper):
    def  __init__(self, load_model_params=True, **params):
        super(ModelWrapper, self).__init__()
        if load_model_params:
            self.load_model_params(**params)
        self.default_segmentation = "original"
        self.default_score = "original"
        
    def load_model_params(self, **params):
        self.__name__ = "reverse_distillation"
        self.params = params
        
        self.device = self.params["device"]
        print("algo device: ", params["device"])
        
        self.encoder, self.bn = wide_resnet50_2(pretrained=True)
        self.encoder = self.encoder.to(self.device)
        self.bn = self.bn.to(self.device)
        self.encoder.eval()
        self.decoder = de_wide_resnet50_2(pretrained=False)
        self.decoder = self.decoder.to(self.device)

        self.optimizer = torch.optim.Adam(list(self.decoder.parameters())+list(self.bn.parameters()), 
                                          lr=params["learning_rate"], 
                                          betas=(0.5,0.999))
                
    def train_one_epoch(self):
        self.bn.train()
        self.decoder.train()
        for img, _, _ in self.dataloader_train:
            img = img.to(self.device)
            inputs = self.encoder(img)
            outputs = self.decoder(self.bn(inputs))
            loss = loss_function(inputs, outputs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def pre_eval_processing(self):
        pass
    
    def save_model(self, location):
        self.save_params(location)
        torch.save({'bn': self.bn.state_dict(),
                    'decoder': self.decoder.state_dict()}, os.path.join(location, "models.pt"))

    def load_model(self, location, **kwargs):
        params = self.load_params(location)
        self.load_model_params(**params)
        loaded_weights = torch.load(os.path.join(location, "models.pt"))
        self.decoder.load_state_dict(loaded_weights['decoder'])
        self.bn.load_state_dict(loaded_weights['bn'])
        
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        with torch.no_grad():
            for ind, (img, _) in enumerate(dataloader):
                img = img.to(self.device)
                
                inputs_batch = self.encoder(img)
                outputs_batch = self.decoder(self.bn(inputs_batch))
                
                anomaly_map_batch = []
                for ii in range(inputs_batch[0].shape[0]):
                    anomaly_map, _ = cal_anomaly_map([inputs_batch[jj][ii][None, ...] for jj in range(len(inputs_batch))], 
                                                     [outputs_batch[jj][ii][None, ...] for jj in range(len(outputs_batch))], 
                                                     img.shape[-1], amap_mode='a')
                    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                    anomaly_map_batch.append(anomaly_map)
                anomaly_map_batch = np.array(anomaly_map_batch)
             
                anomaly_maps = anomaly_map_batch
                image_scores = anomaly_map_batch.max(-1).max(-1)
   
                ret = {}   
                ret['segmentations'] = {}
                ret['scores']        = {}
                ret['segmentations']["original"] = anomaly_maps
                ret['scores']["original"]        = image_scores
                
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
