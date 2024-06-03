from algos.ddad.model import UNetModel
from algos.model_wrapper import ModelWrapper
from algos.ddad.resnet import *
import torch
import numpy as np
import os
import json
from torchvision import transforms
from tqdm import tqdm
import torch
import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torchvision.transforms import transforms
import math 
import numpy as np

def patchify(features, return_spatial_info=False):
    """Convert a tensor into a tensor of respective patches.
    Args:
        x: [torch.Tensor, bs x c x w x h]
    Returns:
        x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
        patchsize]
    """
    patchsize = 3
    stride = 1
    padding = int((patchsize - 1) / 2)
    unfolder = torch.nn.Unfold(
        kernel_size=patchsize, stride=stride, padding=padding, dilation=1
    )
    unfolded_features = unfolder(features)
    number_of_total_patches = []
    for s in features.shape[-2:]:
        n_patches = (
            s + 2 * padding - 1 * (patchsize - 1) - 1
        ) / stride + 1
        number_of_total_patches.append(int(n_patches))
    unfolded_features = unfolded_features.reshape(
        *features.shape[:2], patchsize, patchsize, -1
    )
    unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
    max_features = torch.mean(unfolded_features, dim=(3,4))
    features = max_features.reshape(features.shape[0], 
                                    int(math.sqrt(max_features.shape[1])), 
                                    int(math.sqrt(max_features.shape[1])), 
                                    max_features.shape[-1]).permute(0,3,1,2)
    if return_spatial_info:
        return unfolded_features, number_of_total_patches
    return features

def pixel_distance(output, target):
    '''
    Pixel distance between image1 and image2
    '''
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    output = transform(output)
    target = transform(target)
    distance_map = torch.mean(torch.abs(output - target), dim=1).unsqueeze(1)
    return distance_map

def feature_distance(output, target, FE, params):
    '''
    Feature distance between output and target
    '''
    FE.eval()
    transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / (2)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    target = transform(target)
    output = transform(output)
    inputs_features = FE(target)
    output_features = FE(output)
    out_size = params["input_size"]
    anomaly_map = torch.zeros([inputs_features[0].shape[0] ,1 ,out_size, out_size]).to(params["device"])
    for i in range(len(inputs_features)):
        if i == 0:
            continue
        a_map = 1 - F.cosine_similarity(patchify(inputs_features[i]), patchify(output_features[i]))
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        anomaly_map += a_map
    return anomaly_map 

def heat_map(output, target, FE, params):
    '''
    Compute the anomaly map
    :param output: the output of the reconstruction
    :param target: the target image
    :param FE: the feature extractor
    :param sigma: the sigma of the gaussian kernel
    :param i_d: the pixel distance
    :param f_d: the feature distance
    '''
    sigma = 4
    kernel_size = 2 * int(4 * sigma + 0.5) +1
    anomaly_map = 0

    output = output.to(params["device"])
    target = target.to(params["device"])

    i_d = pixel_distance(output, target)
    f_d = feature_distance((output),  (target), FE, params)
    f_d = torch.Tensor(f_d).to(params["device"])
    # print('image_distance max : ',torch.max(i_d))
    # print('feature_distance max : ',torch.max(f_d))
    # visualalize_distance(output, target, i_d, f_d)
    anomaly_map += f_d + params["v"] *  torch.max(f_d)/ torch.max(i_d) * i_d  
    anomaly_map = gaussian_blur2d(
        anomaly_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )
    anomaly_map = torch.sum(anomaly_map, dim=1).unsqueeze(1)
    return anomaly_map


def Reconstruction(y0, x, seq, model, params):
    '''
    The reconstruction process
    :param y: the target image
    :param x: the input image
    :param seq: the sequence of denoising steps
    :param model: the UNet model
    :param x0_t: the prediction of x0 at time step t
    '''
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(t.long(),params)
            at_next = compute_alpha(next_t.long(),params)
            xt = xs[-1].to(params["device"])
            et = model(xt, t)
            yt = at.sqrt() * y0 + (1- at).sqrt() *  et
            et_hat = et - (1 - at).sqrt() * params["w"] * (yt-xt)
            x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                params["eta"] * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
            xs.append(xt_next.to('cpu'))
    return xs

def compute_alpha(t, params):
    betas = np.linspace(params["beta_start"], 
                        params["beta_end"], 
                        params["trajectory_steps"], dtype=np.float64)
    betas = torch.tensor(betas).type(torch.float)
    beta = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    beta = beta.to(params["device"])
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def compute_alpha(t, params):
    betas = np.linspace(params["beta_start"], 
                        params["beta_end"], 
                        params["trajectory_steps"], dtype=np.float64)
    betas = torch.tensor(betas).type(torch.float)
    beta = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    beta = beta.to(params["device"])
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def get_loss(model, x_0, t, params):
    x_0 = x_0.to(params["device"])
    betas = np.linspace(params["beta_start"], 
                        params["beta_end"], 
                        params["trajectory_steps"], dtype=np.float64)
    b = torch.tensor(betas).type(torch.float).to(params["device"])
    e = torch.randn_like(x_0, device = x_0.device)
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = at.sqrt() * x_0 + (1- at).sqrt() * e
    output = model(x, t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

class WrapperDDAD(ModelWrapper):
    def  __init__(self, load_model_params=True, **params):
        super(ModelWrapper, self).__init__()
        if load_model_params:
            self.load_model_params(**params)
        self.default_segmentation = "method_1"
        self.default_score = "method_1"
        
    def load_model_params(self, **params):
        self.__name__ = "DDAD"
        self.params = params
        self.device = params["device"]
        
        self.model = UNetModel(self.params["input_size"], 
                               64, 
                               dropout=0.0, 
                               n_heads=4,
                               in_channels=self.params["input_channel"])
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.params["learning_rate"], 
                                          weight_decay=self.params["weight_decay"])
        self.get_feature_extractor()
        
    def get_feature_extractor(self):
        if self.params["feature_extractor"] == 'wide_resnet101_2':
            self.feature_extractor = wide_resnet101_2(pretrained=True)
        elif self.params["feature_extractor"] == 'wide_resnet50_2':
            self.feature_extractor = wide_resnet50_2(pretrained=True)
        elif self.params["feature_extractor"] == 'resnet50': 
            self.feature_extractor = resnet50(pretrained=True)
        else:
            self.feature_extractor = wide_resnet101_2(pretrained=True)
        
        self.feature_extractor = self.feature_extractor.to(self.device)
        
    def train_one_epoch(self):
        for batch, _, _ in self.dataloader_train:
            t = torch.randint(0, 
                              self.params["trajectory_steps"], 
                              (batch[0].shape[0],), 
                              device=self.device).long()
            self.optimizer.zero_grad()
            loss = get_loss(self.model, 
                            batch[0], 
                            t, 
                            self.params) 
            loss.backward()
            self.optimizer.step()
                
    def pre_eval_processing(self):
        pass

    def save_model(self, location):
        self.save_params(location)
        torch.save(self.model.state_dict(), os.path.join(location, f'model.pth'))
        
    def load_model(self, location, **kwargs):
        params = self.load_params(location)
        self.load_model_params(**params)
        self.model.load_state_dict(torch.load(os.path.join(location, f'model.pth')))    

    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        predictions= []
        anomaly_map_list = []
        reconstructed_list = []
        forward_list = []

        with torch.no_grad():
            for (data, _) in dataloader:
                data = data.to(self.device)

                test_trajectoy_steps = torch.Tensor([self.params["test_trajectoy_steps"]]).type(torch.int64).to(self.device)
                at = compute_alpha(test_trajectoy_steps.long(), self.params)
                noisy_image = at.sqrt() * data + (1- at).sqrt() * torch.randn_like(data).to(self.device)
                seq = range(0, self.params["test_trajectoy_steps"], self.params["skip"])
                reconstructed = Reconstruction(data, 
                                               noisy_image, 
                                               seq, 
                                               self.model, 
                                               self.params,)
                data_reconstructed = reconstructed[-1]

                anomaly_map = heat_map(data_reconstructed, 
                                       data, 
                                       self.feature_extractor, 
                                       self.params)

                transform = transforms.Compose([
                    transforms.CenterCrop((224)), 
                ])

                anomaly_map = transform(anomaly_map)
                anomaly_map_list.append(anomaly_map)
                reconstructed_list.append(data_reconstructed)

        reconstructed_list = torch.cat(reconstructed_list, dim=0).cpu().numpy()
        anomaly_map_list = transforms.Resize((256, 256))(torch.cat(anomaly_map_list, dim=0)).cpu().numpy()
        
        a_map_list = {"method_1": anomaly_map_list,
                      "method_2": reconstructed_list.mean(1)}
        
        s_map_list = {"method_1": anomaly_map_list.max(-1).max(-1),
                      "method_2": reconstructed_list.max(-1).max(-1).mean(axis=-1)}
        
        return a_map_list, s_map_list