'''
Based on code found here: https://github.com/marco-rudolph/AST
And based on the the paper: 
https://arxiv.org/abs/2210.07829
@inproceedings { RudWeh2023,
author = {Marco Rudolph and Tom Wehrbein and Bodo Rosenhahn and Bastian Wandt},
title = {Asymmetric Student-Teacher Networks for Industrial Anomaly Detection},
booktitle = {Winter Conference on Applications of Computer Vision (WACV)},
year = {2023},
month = jan
}
'''
import torch
from types import SimpleNamespace
from algos.model_wrapper import ModelWrapper
from tqdm import tqdm
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from algos.ast.freia_funcs import (InputNode, 
                         F_conv, 
                         Node, 
                         permute_layer, 
                         glow_coupling_layer_cond,
                         OutputNode,
                         ReversibleGraphNet)
from efficientnet_pytorch import EfficientNet
from types import SimpleNamespace

def get_nf(c, pos_enc, input_dim, channels_hidden):
    nodes = list()
    if pos_enc:
        nodes.append(InputNode(c.pos_enc_dim, name='input'))
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        if pos_enc:
            nodes.append(Node([nodes[-1].out0, nodes[0].out0], glow_coupling_layer_cond,
                              {'clamp': c.clamp,
                               'F_class': F_conv,
                               'cond_dim': c.pos_enc_dim,
                               'F_args': {'channels_hidden': channels_hidden,
                                          'kernel_size': c.kernel_sizes[k]}},
                              name=F'conv_{k}'))
        else:
            nodes.append(Node([nodes[-1].out0], glow_coupling_layer_cond,
                              {'clamp': c.clamp,
                               'F_class': F_conv,
                               'F_args': {'channels_hidden': channels_hidden,
                                          'kernel_size': c.kernel_sizes[k]}},
                              name=F'conv_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    nf = ReversibleGraphNet(nodes, n_jac=1)
    return nf

class FeatureExtractor(nn.Module):
    def __init__(self, layer_idx=35):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        self.layer_idx = layer_idx

    def forward(self, x):
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.layer_idx:
                return x

def positionalencoding2d(D, H, W):
    """
    taken from https://github.com/gudovskiy/cflow-ad
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
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(np.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P[None]

class Model(nn.Module):
    def __init__(self, 
                 nf, 
                 c):
        super(Model, self).__init__()

        
        self.feature_extractor = FeatureExtractor(layer_idx=c.extract_layer)

        if nf:
            self.net = get_nf(c=c,
                              pos_enc=c.pos_enc, 
                              input_dim=c.n_feat, 
                              channels_hidden=c.channels_hidden_teacher)
        else:
            self.net = Student(c=c,
                               pos_enc=c.pos_enc, 
                               channels_hidden=c.channels_hidden_student,
                               n_blocks=c.n_st_blocks)
        self.do_pos_enc = c.pos_enc
        if c.pos_enc:
            self.pos_enc = positionalencoding2d(c.pos_enc_dim, c.map_len, c.map_len).to(c.device)
        else:
            self.pos_enc = None
                    
    def forward(self, x):
      
        with torch.no_grad():
            f = self.feature_extractor(x)

        inp = f
        
        if self.do_pos_enc:
            cond = self.pos_enc.tile(inp.shape[0], 1, 1, 1)
            z = self.net([cond, inp])
        else:
            z = self.net(inp)
        jac = self.net.jacobian(run_forward=False)[0]
        return z, jac


class res_block(nn.Module):
    def __init__(self, channels):
        super(res_block, self).__init__()
        self.l1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.l2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        inp = x
        x = self.l1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.l2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = x + inp
        return x


class Student(nn.Module):
    def __init__(self, c, pos_enc, channels_hidden, n_blocks):
        super(Student, self).__init__()
        inp_feat = c.n_feat if not c.pos_enc else c.n_feat + c.pos_enc_dim
        self.conv1 = nn.Conv2d(inp_feat, channels_hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels_hidden, c.n_feat, kernel_size=3, padding=1)
        self.res = list()
        for _ in range(n_blocks):
            self.res.append(res_block(channels_hidden))
        self.res = nn.ModuleList(self.res)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.act = nn.LeakyReLU()
        self.pos_enc = pos_enc
        
    def forward(self, x):
        if self.pos_enc:
            x = torch.cat(x, dim=1)

        x = self.act(self.conv1(x))
        for i in range(len(self.res)):
            x = self.res[i](x)

        x = self.conv2(x)
        return x

    def jacobian(self, run_forward=False):
        return [0]


class WrapperAST(ModelWrapper):
    def  __init__(self, load_model_params=True, **params):
        super(ModelWrapper, self).__init__()
        if load_model_params:
            self.load_model_params(**params)
        self.default_segmentation = "original"
        self.default_score = "original"
        
    def load_model_params(self, **params):
        self.__name__ = "AST"
        self.params = params
        self.epoch = 0
        self.device = self.params["device"]
        print("algo device: ", params["device"])
                
        self.c = SimpleNamespace(**params)
        
        self.teacher = Model(nf=True, c=self.c)
        self.teacher = self.teacher.to(self.c.device)
        self.teacher_optimizer = torch.optim.Adam(self.teacher.net.parameters(), lr=self.c.lr, eps=1e-08, weight_decay=1e-5)        
        
        self.student = Model(nf=not self.c.asymmetric_student, c=self.c)
        self.student = self.student.to(self.device)
        self.student_optimizer = torch.optim.Adam(self.student.net.parameters(), 
                                                  lr=self.c.lr, 
                                                  eps=1e-08, 
                                                  weight_decay=1e-5)


        
    def train_one_epoch(self):
        
        
        if self.epoch < self.params["switch_teacher_student_training_epoch"]:
            mode = "teacher"
        else:
            mode = "student"
        
        if mode == "teacher":
            print("Doing teacher epoch")
            for _, (data, _, _) in tqdm(enumerate(self.dataloader_train), disable=self.c.hide_tqdm_bar):
                data = data.to(self.c.device)
                
                self.teacher_optimizer.zero_grad()
                
                z, jac = self.teacher(data)

                loss = (0.5 * torch.sum(z ** 2, dim=1) - jac).mean()
                loss.backward()
                self.teacher_optimizer.step()
            
        if mode == "student":
            print("Doing student epoch")
    
            teacher = Model(nf=True, c=self.c).to(self.c.device)
            teacher.net.load_state_dict(self.teacher.net.state_dict())
            teacher.eval()
            #self.student = train_student(self.c, self.teacher, dataloader_train)
            for _, (data, _, _) in tqdm(enumerate(self.dataloader_train), disable=self.c.hide_tqdm_bar):
                data = data.to(self.c.device)
                self.student_optimizer.zero_grad()

                with torch.no_grad():
                    z_t, _ = teacher(data)

                z, jac = self.student(data)
                loss = (torch.mean((z_t - z) ** 2, dim=1)).mean()
                loss.backward()
                self.student_optimizer.step()

        self.epoch += 1
        
    def pre_eval_processing(self):
        pass
    
    def save_model(self, location):
        self.save_params(location)
        torch.save({'teacher': self.teacher.state_dict(),
                    'student': self.student.state_dict()}, os.path.join(location, "models.pt"))

    def load_model(self, location, **kwargs):
        params = self.load_params(location)
        self.load_model_params(**params)
        loaded_weights = torch.load(os.path.join(location, "models.pt"))
        self.teacher.load_state_dict(loaded_weights['teacher'])
        self.student.load_state_dict(loaded_weights['student'])
        
        
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        
        student = Model(nf=False, c=self.c).to(self.c.device)
        student.net.load_state_dict(self.student.net.state_dict())
        
        teacher = Model(nf=True, c=self.c).to(self.c.device)
        teacher.net.load_state_dict(self.teacher.net.state_dict())
        
        teacher.eval()
        student.eval()
            
        with torch.no_grad():
            for ind, (data, _) in tqdm(enumerate(dataloader), total=len_dataloader, disable=self.c.hide_tqdm_bar):
                data = data.to(self.device)
                
                z_t, jac_t = teacher(data)
                z, jac = student(data)

                loss_per_pixel = torch.mean((z_t - z) ** 2, dim=1)
                loss_per_sample = torch.mean(loss_per_pixel, dim=(-1, -2))
        
                loss_per_pixel = torch.unsqueeze(loss_per_pixel, dim=1)
                loss_per_pixel = F.interpolate(loss_per_pixel, size=256, mode='bilinear', align_corners=True)

                ret = {}   
                ret['segmentations'] = {}
                ret['scores']        = {}
                ret['segmentations']["original"] = loss_per_pixel.cpu().numpy()
                ret['scores']["original"]        = loss_per_sample.cpu().numpy()

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