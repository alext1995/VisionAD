'''
    Implementation adapted from: 
    https://github.com/smiler96/PFM-and-PEFM-for-Image-Anomaly-Detection-and-Segmentation
    Credit to the authors the great implementation
    Cite their work here:
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
from tqdm import tqdm
import shutil
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torchvision.models.vgg import vgg16_bn, vgg19_bn
import os
import numpy as np
#from loguru import logger
import argparse
import time
#import matplotlib.pyplot as plt
import json


class Conv_BN_PRelu(nn.Module):
    def __init__(self, in_dim, out_dim, k=1, s=1, p=0, bn=True, prelu=True):
        super(Conv_BN_PRelu, self).__init__()
        self.conv = [
            nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p),
        ]
        if bn:
            self.conv.append(nn.BatchNorm2d(out_dim))
        if prelu:
            self.conv.append(nn.PReLU())

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)

class NonLocalAttention(nn.Module):
    def __init__(self, channel=256, reduction=2, rescale=1.0):
        super(NonLocalAttention, self).__init__()
        # self.conv_match1 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        # self.conv_match2 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        # self.conv_assembly = common.BasicBlock(conv, channel, channel, 1,bn=False, act=nn.PReLU())
         
        self.conv_match1 = Conv_BN_PRelu(channel, channel//reduction, 1, bn=False, prelu=True)
        self.conv_match2 = Conv_BN_PRelu(channel, channel//reduction, 1, bn=False, prelu=True)
        self.conv_assembly = Conv_BN_PRelu(channel, channel, 1,bn=False, prelu=True)
        self.rescale = rescale

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)

        N,C,H,W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0,2,3,1).view((N,H*W,C))
        x_embed_2 = x_embed_2.view(N,C,H*W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(N,-1,H*W).permute(0,2,1)
        x_final = torch.matmul(score, x_assembly)
        x_final = x_final.permute(0,2,1).view(N,-1,H,W)
        return x_final + input*self.rescale

class PretrainedModel(nn.Module):
    def __init__(self, model_name):
        super(PretrainedModel, self).__init__()
        if "resnet" in model_name:
            model = eval(model_name)(pretrained=True)
            modules = list(model.children())
            self.block1 = nn.Sequential(*modules[0:4])
            self.block2 = modules[4]
            self.block3 = modules[5]
            self.block4 = modules[6]
            self.block5 = modules[7]
        elif "vgg" in model_name:
            if model_name == "vgg16_bn":
                self.block1 = nn.Sequential(*self.modules[0:14])
                self.block2 = nn.Sequential(*self.modules[14:23])
                self.block3 = nn.Sequential(*self.modules[23:33])
                self.block4 = nn.Sequential(*self.modules[33:43])
            else:
                self.block1 = nn.Sequential(*self.modules[0:14])
                self.block2 = nn.Sequential(*self.modules[14:26])
                self.block3 = nn.Sequential(*self.modules[26:39])
                self.block4 = nn.Sequential(*self.modules[39:52])
        else:
            raise NotImplementedError

    def forward(self, x):
        # B x 64 x 64 x 64
        out1 = self.block1(x)
        # B x 128 x 32 x 32
        out2 = self.block2(out1)
        # B x 256 x 16 x 16
        # 32x32x128
        out3 = self.block3(out2)
        # 16x16x256
        out4 = self.block4(out3)
        return {"out2": out2,
                "out3": out3,
                "out4": out4
                }

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

class Conv_BN_Relu(nn.Module):
    def __init__(self, in_dim, out_dim, k=1, s=1, p=0, bn=True, relu=True):
        super(Conv_BN_Relu, self).__init__()
        self.conv = [
            nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p),
        ]
        if bn:
            self.conv.append(nn.BatchNorm2d(out_dim))
        if relu:
            self.conv.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


class DualProjectionNet(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, latent_dim=256):
        super(DualProjectionNet, self).__init__()
        self.encoder1 = nn.Sequential(*[
            Conv_BN_Relu(in_dim, in_dim//2+latent_dim),
            Conv_BN_Relu(in_dim//2+latent_dim, 2*latent_dim),
            # Conv_BN_Relu(2*latent_dim, latent_dim),
        ])

        self.shared_coder = Conv_BN_Relu(2*latent_dim, latent_dim, bn=False, relu=False)

        self.decoder1 = nn.Sequential(*[
            Conv_BN_Relu(latent_dim, 2*latent_dim),
            Conv_BN_Relu(2*latent_dim, out_dim//2+latent_dim),
            Conv_BN_Relu(out_dim//2+latent_dim, out_dim, bn=False, relu=False),
        ])

        self.encoder2 = nn.Sequential(*[
            Conv_BN_Relu(out_dim, out_dim // 2 + latent_dim),
            Conv_BN_Relu(out_dim // 2 + latent_dim, 2 * latent_dim),
            # Conv_BN_Relu(2 * latent_dim, latent_dim),
        ])

        self.decoder2 = nn.Sequential(*[
            Conv_BN_Relu(latent_dim, 2 * latent_dim),
            Conv_BN_Relu(2 * latent_dim, in_dim // 2 + latent_dim),
            Conv_BN_Relu(in_dim // 2 + latent_dim, in_dim, bn=False, relu=False),
        ])


    def forward(self, xs, xt):
        xt_hat = self.encoder1(xs)
        xt_hat = self.shared_coder(xt_hat)
        xt_hat = self.decoder1(xt_hat)

        xs_hat = self.encoder2(xt)
        xs_hat = self.shared_coder(xs_hat)
        xs_hat = self.decoder2(xs_hat)

        return xs_hat, xt_hat

class DualProjectionWithPENet(nn.Module):
    def __init__(self, H, W, device, in_dim=512, out_dim=512, latent_dim=256, dual_type="middle", pe_required=True):
        super(DualProjectionWithPENet, self).__init__()

        self.H = H
        self.W = W
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.pe_required = pe_required
        if self.pe_required:
            self.pe1 = positionalencoding2d(self.in_dim, H, W)
            self.pe1 = self.pe1.to(device)
            self.pe2 = positionalencoding2d(self.out_dim, H, W)
            self.pe2 = self.pe2.to(device)
        else:
            self.pe1 = torch.zeros([1]).to(device)
            self.pe2 = torch.zeros([1]).to(device)

        self.dual_type = dual_type
        # assert self.dual_type in ["less", "small", "middle", "large"]

        if self.dual_type == "small":
            self.encoder1 = nn.Sequential(*[
                Conv_BN_Relu(2*in_dim, in_dim//2+latent_dim),
                Conv_BN_Relu(in_dim//2+latent_dim, 2*latent_dim),
                # Conv_BN_Relu(2*latent_dim, latent_dim),
            ])

            self.shared_coder = Conv_BN_Relu(2*latent_dim, latent_dim, bn=False, relu=False)

            self.decoder1 = nn.Sequential(*[
                Conv_BN_Relu(latent_dim, 2*latent_dim),
                Conv_BN_Relu(2*latent_dim, out_dim//2+latent_dim),
                Conv_BN_Relu(out_dim//2+latent_dim, out_dim, bn=False, relu=False),
            ])


            self.encoder2 = nn.Sequential(*[
                Conv_BN_Relu(2*out_dim, out_dim // 2 + latent_dim),
                Conv_BN_Relu(out_dim // 2 + latent_dim, 2 * latent_dim),
                # Conv_BN_Relu(2 * latent_dim, latent_dim),
            ])

            self.decoder2 = nn.Sequential(*[
                Conv_BN_Relu(latent_dim, 2 * latent_dim),
                Conv_BN_Relu(2 * latent_dim, in_dim // 2 + latent_dim),
                Conv_BN_Relu(in_dim // 2 + latent_dim, in_dim, bn=False, relu=False),
            ])
        
        elif self.dual_type == "small_nonlocal": 
            self.encoder1 = nn.Sequential(*[
                Conv_BN_Relu(2*in_dim, in_dim//2+latent_dim),
                Conv_BN_Relu(in_dim//2+latent_dim, 2*latent_dim),
                # Conv_BN_Relu(2*latent_dim, latent_dim),
                NonLocalAttention(channel=2*latent_dim)
            ])

            self.shared_coder = Conv_BN_Relu(2*latent_dim, latent_dim, bn=False, relu=False)

            self.decoder1 = nn.Sequential(*[
                Conv_BN_Relu(latent_dim, 2*latent_dim),
                NonLocalAttention(channel=2*latent_dim),
                Conv_BN_Relu(2*latent_dim, out_dim//2+latent_dim),
                Conv_BN_Relu(out_dim//2+latent_dim, out_dim, bn=False, relu=False),
            ])

            self.encoder2 = nn.Sequential(*[
                Conv_BN_Relu(2*out_dim, out_dim // 2 + latent_dim),
                Conv_BN_Relu(out_dim // 2 + latent_dim, 2 * latent_dim),
                NonLocalAttention(channel=2*latent_dim) 
            ])

            self.decoder2 = nn.Sequential(*[
                Conv_BN_Relu(latent_dim, 2 * latent_dim),
                NonLocalAttention(channel=2*latent_dim),
                Conv_BN_Relu(2 * latent_dim, in_dim // 2 + latent_dim),
                Conv_BN_Relu(in_dim // 2 + latent_dim, in_dim, bn=False, relu=False),
            ])
        
        elif self.dual_type == "less":
            # logger.info(f"less model is Used!")
            self.encoder1 = nn.Sequential(*[
                Conv_BN_Relu(in_dim, latent_dim)
            ]) 
            self.shared_coder = Conv_BN_Relu(latent_dim, latent_dim, bn=True, relu=True) 
            self.decoder1 = nn.Sequential(*[ 
                Conv_BN_Relu(latent_dim, out_dim, bn=False, relu=False),
            ]) 

            self.encoder2 = nn.Sequential(*[
                Conv_BN_Relu(out_dim, latent_dim), 
            ])

            self.decoder2 = nn.Sequential(*[ 
                Conv_BN_Relu(latent_dim, in_dim, bn=False, relu=False),
            ])


    def forward(self, xs, xt):
        # xt_hat = self.encoder1(xs + self.pe1.unsqueeze(0)) 
        B, _, _, _ = xs.shape
        pe1 = self.pe1.unsqueeze(0)
        pe1 = pe1.repeat(B, 1, 1, 1)
        xs_pe1 = torch.cat([xs, pe1], dim=1)
        xt_hat = self.encoder1(xs_pe1)
        xt_hat = self.shared_coder(xt_hat)
        xt_hat = self.decoder1(xt_hat)
        
        pe2 = self.pe2.unsqueeze(0)
        pe2 = pe2.repeat(B, 1, 1, 1)
        xt_pe2 = torch.cat([xt, pe2], dim=1)
        xs_hat = self.encoder2(xt_pe2)
        xs_hat = self.shared_coder(xs_hat)
        xs_hat = self.decoder2(xs_hat)

        return xs_hat, xt_hat

class Base(object):
    def __init__(self, *args, **kwargs): 
        pass
    
class PFMPEFMBase(Base):
    def __init__(self, **kwargs):
        pass

    def _get_agent_out(self, x):
        out_a1 = self.Agent1(x)
        out_a2 = self.Agent2(x)
        for key in out_a2.keys():
            out_a1[key] = F.normalize(out_a1[key], p=2)
            out_a2[key] = F.normalize(out_a2[key], p=2)
        return out_a1, out_a2

    def train_one_epoch(self, dataloader, limit=None):
        self.projector2.train()
        self.projector3.train()
        self.projector4.train()
        for i, (data, _, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if limit and i>limit:
                break
            data = data.to(self.device)
            out_a1, out_a2 = self._get_agent_out(data)

            # project_out2 = self.projector2(out_a1["out2"].detach())
            # loss2 = torch.mean((out_a2["out2"].detach() - project_out2) ** 2)
            project_out21, project_out22 = self.projector2(out_a1["out2"].detach(), out_a2["out2"].detach())
            loss21 = torch.mean((out_a1["out2"] - project_out21) ** 2)
            loss22 = torch.mean((out_a2["out2"] - project_out22) ** 2)
            loss2 = loss21 + loss22
            self.optimizer2.zero_grad()
            loss2.backward()
            self.optimizer2.step()

            project_out31, project_out32 = self.projector3(out_a1["out3"].detach(), out_a2["out3"].detach())
            loss31 = torch.mean((out_a1["out3"].detach() - project_out31) ** 2)
            loss32 = torch.mean((out_a2["out3"].detach() - project_out32) ** 2)
            loss3 = loss31 + loss32
            self.optimizer3.zero_grad()
            loss3.backward()
            self.optimizer3.step()

            project_out41, project_out42 = self.projector4(out_a1["out4"].detach(), out_a2["out4"].detach())
            loss41 = torch.mean((out_a1["out4"].detach() - project_out41) ** 2)
            loss42 = torch.mean((out_a2["out4"].detach() - project_out42) ** 2)
            loss4 = loss41 + loss42
            self.optimizer4.zero_grad()
            loss4.backward()
            self.optimizer4.step() 
       
    def pre_eval_processing(self, train_dataloader):
        self._statistic_var(train_dataloader)

    def _statistic_var(self, dataloader, c=False):
        if self.norm_training_features:
            self.var21 = 0
            self.var22 = 0
            self.var31 = 0
            self.var32 = 0
            self.var41 = 0
            self.var42 = 0
            with torch.no_grad():
                for i, (x, _, _, _) in enumerate(dataloader):
                    torch.cuda.empty_cache()
                    x = x.to(self.device)
                    out_a1, out_a2 = self._get_agent_out(x)
                    project_out21, project_out22 = self.projector2(out_a1["out2"], out_a2["out2"])
                    var21 = (out_a1["out2"] - project_out21) ** 2
                    var22 = (out_a2["out2"] - project_out22) ** 2
                    self.var21 += torch.mean(var21, dim=0, keepdim=True)
                    self.var22 += torch.mean(var22, dim=0, keepdim=True)

                    project_out31, project_out32 = self.projector3(out_a1["out3"], out_a2["out3"])
                    var31 = (out_a1["out3"] - project_out31) ** 2
                    var32 = (out_a2["out3"] - project_out32) ** 2
                    self.var31 += torch.mean(var31, dim=0, keepdim=True)
                    self.var32 += torch.mean(var32, dim=0, keepdim=True)

                    project_out41, project_out42 = self.projector4(out_a1["out4"], out_a2["out4"])
                    var41 = (out_a1["out4"] - project_out41) ** 2
                    var42 = (out_a2["out4"] - project_out42) ** 2
                    self.var41 += torch.mean(var41, dim=0, keepdim=True)
                    self.var42 += torch.mean(var42, dim=0, keepdim=True)

            self.var21 /= len(dataloader)
            self.var22 /= len(dataloader)
            self.var31 /= len(dataloader)
            self.var32 /= len(dataloader)
            self.var41 /= len(dataloader)
            self.var42 /= len(dataloader)
        else:
            self.var21 = 1
            self.var22 = 1
            self.var31 = 1
            self.var32 = 1
            self.var41 = 1
            self.var42 = 1

    def eval_outputs_dataloader(self, dataloader):
        self.projector2.eval()
        self.projector3.eval()
        self.projector4.eval()

        with torch.no_grad():
            test_y_list = []
            test_mask_list = []
            test_img_list = []
            test_img_name_list = []
            # pixel-level
            score_map_list = []
            score_list = []

            score2_map_list = []
            score2_list = []
            score3_map_list = []
            score3_list = []
            score4_map_list = []
            score4_list = []

            paths = []
            score_maxs_out = np.array([])
            score_means_out = np.array([])

            heatmaps_out = np.empty((0, 1, 256, 256)) 
            targets_out  = np.empty((0, 1, 256, 256)) 
            start_t = time.time()
            for ind, (x, _) in enumerate(dataloader):

                x = x.to(self.device)
                _, _, H, W = x.shape
                out_a1, out_a2 = self._get_agent_out(x)

                project_out21, project_out22 = self.projector2(out_a1["out2"], out_a2["out2"])
                loss21_map = torch.sum((out_a1["out2"] - project_out21) ** 2 / self.var21, dim=1, keepdim=True)
                loss22_map = torch.sum((out_a2["out2"] - project_out22) ** 2 / self.var22, dim=1, keepdim=True)

                loss2_map = (loss21_map + loss22_map) / 2.0
                score2_map = F.interpolate(loss2_map, size=(H, W), mode='bilinear', align_corners=False)
                score2_map = score2_map.cpu().detach().numpy()
                score2_map_list.extend(score2_map)
                score2_list.extend(np.squeeze(np.max(np.max(score2_map, axis=2), axis=2), 1))

                project_out31, project_out32 = self.projector3(out_a1["out3"], out_a2["out3"])
                loss31_map = torch.sum((out_a1["out3"] - project_out31) ** 2 / self.var31, dim=1, keepdim=True)
                loss32_map = torch.sum((out_a2["out3"] - project_out32) ** 2 / self.var32, dim=1, keepdim=True)

                loss3_map = (loss31_map + loss32_map) / 2.0
                score3_map = F.interpolate(loss3_map, size=(H, W), mode='bilinear', align_corners=False)
                score3_map = score3_map.cpu().detach().numpy()
                score3_map_list.extend(score3_map)
                score3_list.extend(np.squeeze(np.max(np.max(score3_map, axis=2), axis=2), 1))

                project_out41, project_out42 = self.projector4(out_a1["out4"], out_a2["out4"])
                loss41_map = torch.sum((out_a1["out4"] - project_out41) ** 2 / self.var41, dim=1, keepdim=True)
                loss42_map = torch.sum((out_a2["out4"] - project_out42) ** 2 / self.var42, dim=1, keepdim=True)

                loss4_map = (loss41_map + loss42_map) / 2.0
                score4_map = F.interpolate(loss4_map, size=(H, W), mode='bilinear', align_corners=False)
                score4_map = score4_map.cpu().detach().numpy()
                score4_map_list.extend(score4_map)
                score4_list.extend(np.squeeze(np.max(np.max(score4_map, axis=2), axis=2), 1))

                score_map = (score4_map + score3_map + score2_map) / 3
                # score_map = gaussian_filter(score_map.squeeze(), sigma=4)

                score_map_list.extend(score_map)
                # score_list.extend(np.squeeze(np.max(np.max(score_map, axis=2), axis=2), 1))
                score_map_flattened = np.reshape(score_map, (score_map.shape[0], -1))

                score_max = np.max(score_map_flattened, 1) ## original method
                score_mean = np.mean(score_map_flattened, 1) ## new method

                score_maxs_out = np.concatenate((score_maxs_out, score_max))
                score_means_out = np.concatenate((score_means_out, score_mean))

                heatmaps_out = np.vstack((heatmaps_out, score_map))

        score2_list = np.array(score2_list)
        score3_list = np.array(score3_list)
        score4_list = np.array(score4_list)

        score2_map_list = np.array(score2_map_list)
        score3_map_list = np.array(score3_map_list)
        score4_map_list = np.array(score4_map_list)

        scores_out = {"Original_method_score_max": score_maxs_out,}

        heatmaps_out = {"Original_method": torch.tensor(heatmaps_out),}
        
        return heatmaps_out, scores_out
    
    def save_model(self, location):
        var_tuple = (self.var21,self.var22,self.var31,self.var32,self.var41,self.var42,)
        with open(os.path.join(location, "var_tuple.json"), "w") as f:
            json.dump(var_tuple, f)
            
        torch.save(self.projector2.state_dict(), os.path.join(location, "self.ckpt2"))
        torch.save(self.projector3.state_dict(), os.path.join(location, "self.ckpt3"))
        torch.save(self.projector4.state_dict(), os.path.join(location, "self.ckpt4"))
        
    def load_model(self, location):
        with open(os.path.join(location, "var_tuple.json"), "r") as f:
            var_tuple = json.load(f)
        (self.var21,self.var22,self.var31,self.var32,self.var41,self.var42) = var_tuple
        self.projector2.load_state_dict(torch.load(os.path.join(location, "self.ckpt2")))
        self.projector3.load_state_dict(torch.load(os.path.join(location, "self.ckpt3")))
        self.projector4.load_state_dict(torch.load(os.path.join(location, "self.ckpt4")))
    
class PFM(PFMPEFMBase):
    def __init__(self, **kwargs):
        super(PFMPEFMBase, self).__init__(**kwargs)
        
        self.norm_training_features = kwargs['norm_training_features']

        agent_S = kwargs['agent_S']
        agent_T = kwargs['agent_T']
        
        self.device = kwargs["device"] 
        
        self.s_name = agent_S
        self.t_name = agent_T
        
        if agent_S == "resnet18" or agent_S == "resnet34":
            self.Agent1 = PretrainedModel(model_name=agent_S)
            self.indim = [64, 128, 256]
            # self.outdim = [50, 100, 200]
        elif agent_S == "resnet50":
            self.Agent1 = PretrainedModel(model_name=agent_S)
            self.indim = [256, 512, 1024]
            # self.Agent2 = PretrainedModel(model_name="resnet34")
        if agent_T == "resnet50" or agent_T == "resnet101":
            # self.Agent1 = PretrainedModel(model_name="vgg16")
            self.Agent2 = PretrainedModel(model_name=agent_T)
            self.outdim = [256, 512, 1024]
            self.latent_dim = [200, 400, 900]
            
        self.projector2 = DualProjectionNet(in_dim=self.indim[0], out_dim=self.outdim[0], latent_dim=self.latent_dim[0])
        self.optimizer2 = torch.optim.Adam(self.projector2.parameters(), lr=kwargs["lr2"], weight_decay=kwargs["weight_decay"])
        self.projector3 = DualProjectionNet(in_dim=self.indim[1], out_dim=self.outdim[1], latent_dim=self.latent_dim[1])
        self.optimizer3 = torch.optim.Adam(self.projector3.parameters(), lr=kwargs["lr3"], weight_decay=kwargs["weight_decay"])
        self.projector4 = DualProjectionNet(in_dim=self.indim[2], out_dim=self.outdim[2], latent_dim=self.latent_dim[2])
        self.optimizer4 = torch.optim.Adam(self.projector4.parameters(), lr=kwargs["lr4"], weight_decay=kwargs["weight_decay"])

        self.Agent1.to(self.device).eval()
        self.Agent2.to(self.device).eval()

        self.projector2.to(self.device)
        self.projector3.to(self.device)
        self.projector4.to(self.device)

class PEFM(PFMPEFMBase):
    def __init__(self, **kwargs):
        super(PFMPEFMBase, self).__init__(**kwargs)
        
        self.norm_training_features = kwargs['norm_training_features']

        agent_S = kwargs['agent_S']
        agent_T = kwargs['agent_T']
        
        self.device = kwargs["device"]
        
        self.s_name = agent_S
        self.t_name = agent_T
        
        self.dual_type = kwargs['dual_type']
        self.pe_required = True
        
        if agent_S == "resnet18" or agent_S == "resnet34":
            self.Agent1 = PretrainedModel(model_name=agent_S)
            self.indim = [64, 128, 256]
            # self.outdim = [50, 100, 200]
        elif agent_S == "resnet50" or agent_T == "resnet101":
            self.Agent1 = PretrainedModel(model_name=agent_S)
            self.indim = [256, 512, 1024]
            # self.Agent2 = PretrainedModel(model_name="resnet34")
        if agent_T == "resnet50" or agent_T == "resnet101" or agent_T == "resnet152":
            # self.Agent1 = PretrainedModel(model_name="vgg16")
            self.Agent2 = PretrainedModel(model_name=agent_T)
            self.outdim = [256, 512, 1024]

        self.latent_dim = [200, 400, 800]
        
        if kwargs["resize"] == 256:
            H2, W2 = 64, 64
            H3, W3 = 32, 32
            H4, W4 = 16, 16
        elif kwargs["resize"] == 512:
            H2, W2 = 128, 128
            H3, W3 = 64, 64
            H4, W4 = 32, 32
        
        self.projector2 = DualProjectionWithPENet(H2, W2, self.device, in_dim=self.indim[0], out_dim=self.outdim[0], latent_dim=self.latent_dim[0], dual_type=self.dual_type, pe_required=self.pe_required)
        self.optimizer2 = torch.optim.Adam(self.projector2.parameters(), lr=kwargs["lr2"], weight_decay=kwargs["weight_decay"])

        self.projector3 = DualProjectionWithPENet(H3, W3, self.device, in_dim=self.indim[1], out_dim=self.outdim[1], latent_dim=self.latent_dim[1], dual_type=self.dual_type, pe_required=self.pe_required)
        self.optimizer3 = torch.optim.Adam(self.projector3.parameters(), lr=kwargs["lr3"], weight_decay=kwargs["weight_decay"])

        self.projector4 = DualProjectionWithPENet(H4, W4, self.device, in_dim=self.indim[2], out_dim=self.outdim[2], latent_dim=self.latent_dim[2], dual_type=self.dual_type, pe_required=self.pe_required)
        self.optimizer4 = torch.optim.Adam(self.projector4.parameters(), lr=kwargs["lr4"], weight_decay=kwargs["weight_decay"])

        self.Agent1.to(self.device).eval()
        self.Agent2.to(self.device).eval()

        self.projector2.to(self.device)
        self.projector3.to(self.device)
        self.projector4.to(self.device)