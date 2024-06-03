'''
Based on code found here: https://github.com/TooTouch/MemSeg
And based on the the paper: 
https://arxiv.org/ftp/arxiv/papers/2205/2205.00908.pdf
@article{DBLP:journals/corr/abs-2205-00908,
  author    = {Minghui Yang and
               Peng Wu and
               Jing Liu and
               Hui Feng},
  title     = {MemSeg: {A} semi-supervised method for image surface defect detection
               using differences and commonalities},
  journal   = {CoRR},
  volume    = {abs/2205.00908},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.00908},
  doi       = {10.48550/arXiv.2205.00908},
  eprinttype = {arXiv},
  eprint    = {2205.00908},
  timestamp = {Tue, 03 May 2022 15:52:06 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2205-00908.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
'''
import torch
import os
from algos.model_wrapper import ModelWrapper
from algos.memseg.utils import AverageMeter, SyntheticAnomaly
from algos.memseg.models.memory_module import MemoryBank
from argparse import Namespace
from algos.memseg.focal_loss import FocalLoss
from timm import create_model
from algos.memseg.models.decoder import Decoder
from algos.memseg.models.msff import MSFF
from algos.memseg.scheduler import CosineAnnealingWarmupRestarts
from torch import nn
import torch.nn.functional as F
import wandb

class MemSegModel(nn.Module):
    def __init__(self, memory_bank, feature_extractor):
        super(MemSegModel, self).__init__()

        self.memory_bank = memory_bank
        self.feature_extractor = feature_extractor
        self.msff = MSFF()
        self.decoder = Decoder()

    def forward(self, inputs):
        # extract features
        features = self.feature_extractor(inputs)
        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        # extract concatenated information(CI)
        concat_features = self.memory_bank.select(features = f_ii)

        # Multi-scale Feature Fusion(MSFF) Module
        msff_outputs = self.msff(features = concat_features)

        # decoder
        predicted_mask = self.decoder(
            encoder_output  = f_out,
            concat_features = [f_in] + msff_outputs
        )

        return predicted_mask

class MEMSEG:
    def __init__(self, **params):
        self.params = params
        self.device = params["device"]
        self.synthetic_anomaly = SyntheticAnomaly(texture_source_dir = params["texture_source_dir"],
                                                  always_add_anomaly = params["always_add_anomaly"],
                                                  min_perlin_scale = params["min_perlin_scale"],
                                                  perlin_scale = params["perlin_scale"],
                                                  perlin_noise_threshold = params["perlin_noise_threshold"],
                                                  transparency_range = list(params["transparency_range"]))

        self.feature_extractor = create_model(self.params['feature_extractor_name'], 
                                              pretrained    = True, 
                                              features_only = True,
                                              ).to(self.device)
        ## freeze weight of layer1,2,3
        for l in ['layer1','layer2','layer3']:
            for p in self.feature_extractor[l].parameters():
                p.requires_grad = False

        memory_bank = MemoryBank(
            nb_memory_sample = self.params['nb_memory_sample'],
            device           = self.device
        )

        self.model = MemSegModel(memory_bank=memory_bank, 
                                 feature_extractor=self.feature_extractor)

        self.optimizer = torch.optim.AdamW(params       = filter(lambda p: p.requires_grad, self.model.parameters()), 
                                           lr           = self.params['lr'], 
                                           weight_decay = self.params['weight_decay']
                                           )
        
        if params['use_scheduler']:
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer, 
                first_cycle_steps = params['num_training_steps'],
                max_lr = params['lr'],
                min_lr = params['min_lr'],
                warmup_steps   = int(params['num_training_steps'] * params['warmup_ratio'])
            )
        else:
            self.scheduler = None
        
        self.l1_criterion = nn.L1Loss()
        self.focal_criterion = FocalLoss(
            gamma = self.params['focal_gamma'], 
            alpha = self.params['focal_alpha']
        )
        self.epoch = 0

    def to(self, device):
        self.model.to(device)

    def train_one_epoch(self, dataloader):
        if self.epoch == 0:
            dataloader.dataset.pre_normalisation_transform = None
            self.model.memory_bank.update(self.feature_extractor, dataloader.dataset)

        self.model.train()

        # set optimizer
        self.optimizer.zero_grad()


        dataloader.dataset.pre_normalisation_transform = self.synthetic_anomaly.generate_anomaly
        for i, (inputs, _, masks) in enumerate(dataloader):
            
            inputs, masks = inputs.to(self.device), masks.to(self.device)

            outputs = self.model(inputs)
            outputs = F.softmax(outputs, dim=1)
            l1_loss = self.l1_criterion(outputs[:,1,:], masks)
            focal_loss = self.focal_criterion(outputs, masks)
            loss = (self.params["l1_weight"] * l1_loss) + (self.params["focal_weight"] * focal_loss)

            if self.params["wandb"]:
                wandb.log({"train_loss": loss})
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # update weight
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()

        self.epoch += 1
        
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        self.model.eval()
        with torch.no_grad():
            for ind, (inputs, _) in enumerate(dataloader):
                inputs = inputs.to(self.device)

                outputs_pre_softmax = self.model(inputs)
                outputs = F.softmax(outputs_pre_softmax, dim=1)

                segmentations = {}
                segmentations["original"] = outputs[:,1,:]

                scores = {}
                scores["top_100"] = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)

                if ind==0:
                    segmentations_sets = {}
                    for key, item in segmentations.items():
                        segmentations_sets[key] = item.cpu().detach()

                    scores_sets = {}
                    for key, item in scores.items():
                        scores_sets[key] = item.cpu().detach()

                else:
                    for key, item in segmentations.items():
                        segmentations_sets[key] = torch.vstack([segmentations_sets[key], item.cpu().detach()])
                    
                    for key, item in scores.items():
                        scores_sets[key] = torch.concat([scores_sets[key], item.cpu().detach()])
            
        return segmentations_sets, scores_sets 
                
    def save_model(self, location):
        torch.save(self.model.state_dict(), os.path.join(location, f'model.pt'))
        torch.save(self.model.memory_bank.memory_information, os.path.join(location, f'memory_bank.pt'))

    def load_model(self, location):
        self.model.load_state_dict(torch.load(os.path.join(location, f"model.pt")))
        self.model.memory_bank.memory_information = torch.load(os.path.join(location, f'memory_bank.pt'))
        
class WrapperMemSeg(ModelWrapper):
    def __init__(self, load_model_params=True, **params):
        super(ModelWrapper, self).__init__()
        if load_model_params:
            self.load_model_params(**params)
        self.default_segmentation = "original"
        self.default_score = "top_100"

        
    def load_model_params(self, **params):
        self.__name__ = "MEMSEG"
        self.params = params
        self.model = MEMSEG(**params)
        self.device = params["device"]
        self.model.to(self.device)
    
    def train_one_epoch(self):
        self.model.train_one_epoch(self.dataloader_train)
    
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        return self.model.eval_outputs_dataloader(dataloader, len_dataloader)
    
    def pre_eval_processing(self):
        pass
    
    def save_model(self, location):
        self.save_params(location)
        self.model.save_model(location) 
    
    def load_model(self, location, **params):
        params = self.load_params(location)
        self.load_model_params(**params)
        self.model.load_model(location) 
    
