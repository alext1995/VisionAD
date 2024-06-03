'''
Based on code found here: https://github.com/sungwool/cfa_for_anomaly_localization
And based on the the paper: 
https://arxiv.org/pdf/2206.04325.pdf
@article{lee2022cfa,
  title={CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization},
  author={Lee, Sungwook and Lee, Seunghyun and Song, Byung Cheol},
  journal={arXiv preprint arXiv:2206.04325},
  year={2022}
}
'''
from algos.model_wrapper import ModelWrapper
from torch import optim
import torch
import numpy as np
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from algos.cfa.dsvdd import DSVDD
import torch.nn.functional as F
from algos.cfa.cnn.resnet import wide_resnet50_2 as wrn50_2
from algos.cfa.cnn.resnet import resnet18 as res18
from algos.cfa.cnn.efficientnet import EfficientNet as effnet
from algos.cfa.cnn.vgg import vgg19_bn as vgg19

def gaussian_smooth(x, sigma=4):
    bs = x.shape[0]
    for i in range(0, bs):
        x[i] = gaussian_filter(x[i], sigma=sigma)

    return x

class WrapperCFA(ModelWrapper):
    def  __init__(self, load_model_params=True, **params):
        super(ModelWrapper, self).__init__()
        if load_model_params:
            self.load_model_params(**params)
        self.default_segmentation = "normal"
        self.default_score = "max_smoothed"
        
    def load_model_params(self, **params):
        self.__name__ = "cfa"
        self.params = params
        if params["cnn"]== 'wrn50_2':
            self.model = wrn50_2(pretrained=True, progress=True)
        elif params["cnn"] == 'res18':
            self.model = res18(pretrained=True,  progress=True)
        elif params["cnn"] == 'effnet-b5':
            self.model = effnet.from_pretrained('efficientnet-b5')
        elif params["cnn"] == 'vgg19':
            self.model = vgg19(pretrained=True, progress=True)
        self.device = params["device"]
        print("self.device: ", self.device)
        self.loss_fn = DSVDD(params["cnn"], params["gamma_c"], params["gamma_d"], params["device"])
        self.loss_fn = self.loss_fn.to(params["device"])

        self.optimizer     = optim.AdamW(params        = [{'params' : self.loss_fn.parameters()},], 
                                         lr            = params["lr"],
                                         weight_decay  = 5e-4,
                                         amsgrad       = True )
        
        self.model.to(self.device)

    def train_one_epoch(self):

        if not self.loss_fn.centroids_initialised:
            self.loss_fn.init_centroid(self.model, self.dataloader_train)

        self.loss_fn.train()
        for (x, _, _) in self.dataloader_train:
            self.optimizer.zero_grad()
            p = self.model(x.to(self.device))
            loss, _ = self.loss_fn(p)
            loss.backward()
            self.optimizer.step()
        
    def pre_eval_processing(self):
        pass
    
    def save_model(self, location):
        self.save_params(location)
        #torch.save(self.model.state_dict(), os.path.join(location, "model.pt"))
        torch.save(self.loss_fn.state_dict(), os.path.join(location, "dsvdd.pt"))
        torch.save(self.loss_fn.C, os.path.join(location, "C.pt"))
        torch.save(self.loss_fn.scale, os.path.join(location, "scale.pt"))
        
    def load_model(self, location, **kwargs):
        params = self.load_params(location)
        self.load_model_params(**params)
        #self.model.load_state_dict(torch.load(os.path.join(location, "model.pt")), strict=False)
        self.loss_fn.load_state_dict(torch.load(os.path.join(location, "dsvdd.pt")), strict=False)
        self.loss_fn.C = torch.load(os.path.join(location, "C.pt"))
        self.loss_fn.scale = torch.load(os.path.join(location, "scale.pt"))
        
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        self.loss_fn.eval()
        with torch.no_grad():
            for ind, (x, _) in enumerate(tqdm(dataloader)):
                p = self.model(x.to(self.device))
                _, score = self.loss_fn(p)
                heatmap  = score.cpu().detach()
                heatmap  = torch.mean(heatmap, dim=1) 
                heatmaps_upsampled = F.interpolate(heatmap.unsqueeze(1), 
                                                   size=self.params["input_size"], 
                                                   mode='bilinear', align_corners=False)[:,0,...].numpy()
                heatmaps_upsampled = gaussian_smooth(heatmaps_upsampled, sigma=4)
                                
                ret = {}   
                ret['segmentations'] = {}
                ret['scores']        = {}
                ret['segmentations']["normal"] = heatmaps_upsampled
                ret['scores']["max"]           = np.array(heatmap.flatten(1).max(-1)[0])
                ret['scores']["max_smoothed"]  = heatmaps_upsampled.reshape(heatmaps_upsampled.shape[0], -1).max(1)
                
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
    