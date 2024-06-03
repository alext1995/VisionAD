'''
Based on code found here:  https://github.com/cool-xuan/msflow
And based on the the paper: 
}
'''
from algos.msflow.extractor_models import build_extractor
from algos.msflow.flow_models import build_msflow_model
import torch

from algos.model_wrapper import ModelWrapper
import numpy as np
from types import SimpleNamespace
from torch import nn
import torch.nn.functional as F
import math
import os

def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P

def model_forward(c, extractor, parallel_flows, fusion_flow, image):
    h_list = extractor(image)
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    z_list = []
    parallel_jac_list = []
    for idx, (h, parallel_flow, c_cond) in enumerate(zip(h_list, parallel_flows, c.c_conds)):
        y = pool_layer(h)
        B, _, H, W = y.shape
        cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        z, jac = parallel_flow(y, [cond, ])
        z_list.append(z)
        parallel_jac_list.append(jac)

    z_list, fuse_jac = fusion_flow(z_list)
    jac = fuse_jac + sum(parallel_jac_list)

    return z_list, jac

def post_process(c, size_list, outputs_list):
    print('Multi-scale sizes:', size_list)
    logp_maps = [list() for _ in size_list]
    prop_maps = [list() for _ in size_list]
    for l, outputs in enumerate(outputs_list):
        # output = torch.tensor(output, dtype=torch.double)
        outputs = torch.cat(outputs, 0)
        logp_maps[l] = F.interpolate(outputs.unsqueeze(1),
                size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
        output_norm = outputs - outputs.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        prob_map = torch.exp(output_norm) # convert to probs in range [0:1]
        prop_maps[l] = F.interpolate(prob_map.unsqueeze(1),
                size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
    
    logp_map = sum(logp_maps)
    logp_map-= logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    prop_map_mul = torch.exp(logp_map)
    anomaly_score_map_mul = prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
    batch = anomaly_score_map_mul.shape[0]
    top_k = int(c.input_size[0] * c.input_size[1] * c.top_k)
    anomaly_score = np.mean(
        anomaly_score_map_mul.reshape(batch, -1).topk(top_k, dim=-1)[0].detach().cpu().numpy(),
        axis=1)

    prop_map_add = sum(prop_maps)
    prop_map_add = prop_map_add.detach().cpu().numpy()
    anomaly_score_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add

    return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul.detach().cpu().numpy()

def load_weights(parallel_flows, fusion_flow, ckpt_path, optimizer=None):
    print('Loading weights from {}'.format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    
    fusion_state = state_dict['fusion_flow']
    maps = {}
    for i in range(len(parallel_flows)):
        maps[fusion_flow.module_list[i].perm.shape[0]] = i
    temp = dict()
    for k, v in fusion_state.items():
        if 'perm' not in k:
            continue
        temp[k.replace(k.split('.')[1], str(maps[v.shape[0]]))] = v
    for k, v in temp.items():
        fusion_state[k] = v
    fusion_flow.load_state_dict(fusion_state, strict=False)

    for parallel_flow, state in zip(parallel_flows, state_dict['parallel_flows']):
        parallel_flow.load_state_dict(state, strict=False)

    if optimizer:
        optimizer.load_state_dict(state_dict['optimizer'])

    return state_dict['epoch']


class WrapperMSFlow(ModelWrapper):
    def __init__(self, load_params=True, **kwargs):
        super(ModelWrapper, self).__init__()
        if load_params:
            self.load_model_params(**kwargs)
        self.default_segmentation = "method_1"
        self.default_score = "method_1"
        
    def load_model_params(self, **params):
        self.__name__ = "msflow"
        self.params = params
        self.device = params["device"]
        
        if "verbose" in params:
            self.verbose = params["verbose"]
        else:
            self.verbose = 0
        
        if "input_size" not in self.params:
            self.params["input_size"] = 256, 256
            
        self.params_namespace = SimpleNamespace(**params)
        self.extractor, self.output_channels = build_extractor(self.params_namespace)
        self.extractor = self.extractor.to(self.device).eval()
        self.parallel_flows, self.fusion_flow = build_msflow_model(self.params_namespace, self.output_channels)
        self.parallel_flows = [parallel_flow.to(self.device) for parallel_flow in self.parallel_flows]
        self.fusion_flow = self.fusion_flow.to(self.device)
            
        self.fusion_params = list(self.fusion_flow.parameters())
        for parallel_flow in self.parallel_flows:
            self.fusion_params += list(parallel_flow.parameters())

        self.optimizer = torch.optim.Adam(self.fusion_params, lr=self.params["lr"])
        
        if params["warmup_scheduler"]:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 
                                                         start_factor=self.params["lr_warmup_from"], 
                                                         end_factor=1, 
                                                         total_iters=(self.params["lr_warmup_epochs"])*self.params["epochs"])
        else:
            self.warmup_scheduler = None
    
        if self.params["lr_decay_milestones"]:
            self.decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                                   self.params["lr_decay_milestones"], 
                                                                   self.params["lr_decay_gamma"])
        else:
            self.decay_scheduler = None
        
        self.epoch = 0
        
    def train_one_epoch(self,):
        if self.verbose>0:
            print(f"Doing epoch: {self.epoch}")
        self.epoch += 1
        
        self.parallel_flows = [parallel_flow.train() for parallel_flow in self.parallel_flows]
        self.fusion_flow = self.fusion_flow.train()
        
        image_count = 0
        loss_count = 0
        # self.dataloader_train - use this
        for idx, (image, _, _) in enumerate(self.dataloader_train):
            
                
            image = image.to(self.device)
            z_list, jac = model_forward(self.params_namespace, 
                                        self.extractor, 
                                        self.parallel_flows, 
                                        self.fusion_flow, 
                                        image)

            loss = 0.
            for z in z_list:
                loss += 0.5 * torch.sum(z**2, (1, 2, 3))
            loss = loss - jac
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fusion_params, 2)
            self.optimizer.step()
              
            image_count += image.shape[0]
            loss_count += loss
        
        if self.verbose>1:
            print(f"Finished epoch: {self.epoch} - loss per image: {loss_count/image_count}")
            
        if self.warmup_scheduler:
            self.warmup_scheduler.step()
        if self.decay_scheduler:
            self.decay_scheduler.step()
    
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        self.parallel_flows = [parallel_flow.eval() for parallel_flow in self.parallel_flows]
        self.fusion_flow = self.fusion_flow.eval()
    
        length = 0
        self.fusion_flow = self.fusion_flow.eval()
        gt_label_list = list()
        gt_mask_list = list()
        outputs_list = [list() for _ in self.parallel_flows]
        size_list = []
        
        image_count = 0
        loss_count = 0
        
        with torch.no_grad():
            for idx, (image, _) in enumerate(dataloader):
                image = image.to(self.device)
               
                z_list, jac = model_forward(self.params_namespace, 
                                            self.extractor, 
                                            self.parallel_flows, 
                                            self.fusion_flow, image)

                loss = 0.
                for lvl, z in enumerate(z_list):
                    if idx == 0:
                        size_list.append(list(z.shape[-2:]))
                    logp = - 0.5 * torch.mean(z**2, 1)
                    outputs_list[lvl].append(logp)
                    loss += 0.5 * torch.sum(z**2, (1, 2, 3))

                loss = loss - jac
                loss = loss.mean()
#         return post_process(c, size_list, outputs_list)
#         return post_process(self.params_namespace, size_list, outputs_list)
        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(self.params_namespace,
                                                                                   size_list, 
                                                                                   outputs_list)
    
        a_image_ret  = {"method_1":anomaly_score,}
        a_pixel_ret  = {"method_1":anomaly_score_map_add,
                        "method_2":anomaly_score_map_mul}
    
        return a_pixel_ret, a_image_ret

            
        
    def pre_eval_processing(self):
        pass
        
    def save_model(self, location):
        if not os.path.exists(location):
            os.makedirs(location)
        params = self.save_params(location)
        file_name = 'model.pt'
        file_path = os.path.join(location, file_name)
        state = {'fusion_flow': self.fusion_flow.state_dict(),
                 'parallel_flows': [parallel_flow.state_dict() for parallel_flow in self.parallel_flows]}
  
        torch.save(state, file_path)
    
    def load_model(self, location):
        file_name = 'model.pt'
        file_path = os.path.join(location, file_name)
        
        self.load_model_params(**params)
        load_weights(self.parallel_flows, self.fusion_flow, file_path)

