'''
Based on code found here: https://github.com/gathierry/FastFlow
And based on the the papers: 
https://arxiv.org/abs/2111.07677
https://arxiv.org/abs/2210.14913
@misc{https://doi.org/10.48550/arxiv.2210.14913,
  doi = {10.48550/ARXIV.2210.14913},
  url = {https://arxiv.org/abs/2210.14913},
  author = {Kim, Yeongmin and Jang, Huiwon and Lee, DongKeon and Choi, Ho-Jin},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {AltUB: Alternating Training Method to Update Base Distribution of Normalizing Flow for Anomaly Detection},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
'''
import torch
import torch.nn.functional as F
import os
from algos.model_wrapper import ModelWrapper
from algos.fastflow2d.algo_class import WrapperFastFlow2d
from algos.fastflow2d.algo_class import FastFlow
from tqdm import tqdm

def calculate_loss(mode, output, log_jac_dets, mu, log_sigma, i=0):
    if mode=="single_channel_mean_only":
#         wandb.log({
#             f"mean_{i}": output.mean(),
#             f"mean_{i}_pred": mu,
#             f"logvar_{i}_actual": output.var().log(),
#         })
        loss = torch.mean(
                    0.5 * torch.sum((output-mu)**2, dim=(1, 2, 3)) - log_jac_dets
                )
    if mode=="single_channel_mean_var":
#         wandb.log({
#             f"mean_{i}": output.mean(),
#             f"mean_{i}_pred": mu,
#             f"logvar_{i}_actual": output.var().log(),
#             f"logvar_{i}_pred": log_sigma,
#         })
        channelwise_outputs = output.flatten()
        loss = torch.mean(
                    0.5 * torch.sum((log_sigma.exp()**-1)*(output-mu)**2+log_sigma, dim=(1, 2, 3)) - log_jac_dets
                )
    if mode=="multi_channel_mean_only":
        loss = torch.mean(
                    0.5 * torch.sum((output-mu[None,:,None,None])**2, dim=(1, 2, 3)) - log_jac_dets
                )
    if mode=="multi_channel_mean_var":
        loss = torch.mean(
                    0.5 * torch.sum((log_sigma[None,:,None,None].exp()**-1)*(output-mu[None,:,None,None])**2+log_sigma[None,:,None,None], 
                                    dim=(1, 2, 3)) - log_jac_dets
                )                      
    return loss
    
def get_inference_loss(mode, output, mu, log_sigma):
    if mode=="single_channel_mean_only":
        log_prob = (output-mu)**2
    if mode=="single_channel_mean_var":
        log_prob = (log_sigma.exp()**-1)*(output-mu)**2+log_sigma
    if mode=="multi_channel_mean_only":
        log_prob = (output-mu[None,:,None,None])**2
    if mode=="multi_channel_mean_var":
        log_prob = (log_sigma[None,:,None,None].exp()**-1)*(output-mu[None,:,None,None])**2+log_sigma[None,:,None,None]
    return log_prob
    
class WrapperFastFlowAltUB(WrapperFastFlow2d):
    def  __init__(self, load_model_params=True, **params):
        super(WrapperFastFlow2d, self).__init__()
        if load_model_params:
            self.load_model_params(**params)
        
    def load_model_params(self, **params):
        self.__name__ = "fastflow_altub"
        self.params = params
        self.device = params["device"]
        print("Self.device: ", self.device)
        if "device" not in params["fastflow_params"]:
            params["fastflow_params"]["device"] = params["device"]
        self.model = FastFlow(**params["fastflow_params"])
        self.input_size = params["input_size"]
        
        self.mode = params["mode"]
        
        if self.mode=="single_channel_mean_only" or self.mode=="single_channel_mean_var":
            base_mus          = torch.zeros(3, device=self.device)
            self.mus_param    = torch.nn.Parameter(base_mus)

            base_sigmas       = torch.zeros(3, device=self.device)
            self.sigmas_param = torch.nn.Parameter(base_sigmas)
            
            param_list = [self.mus_param, self.sigmas_param]
            
        if self.mode=="multi_channel_mean_only" or self.mode=="multi_channel_mean_var":
            output_sizes = [256, 512, 1024]
            self.mus_param = []
            self.sigmas_param = []
            param_list = []
            for size in output_sizes:
                base_mu = torch.zeros(size, device=self.device)
                base_log_sigma = torch.zeros(size, device=self.device)

                mu_array    = torch.nn.Parameter(base_mu)
                sigma_array = torch.nn.Parameter(base_log_sigma)

                self.mus_param.append(mu_array)
                self.sigmas_param.append(sigma_array)

                param_list.append(mu_array)
                param_list.append(sigma_array)

        
        learning_rate = 1e-3
        
        self.clip_value = 100
        self.normal_optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=learning_rate, 
                                          weight_decay=learning_rate,
                                          )
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=learning_rate, 
                                          weight_decay=learning_rate,
                                          )
        self.base_param_optimizer_1 = torch.optim.Adam(param_list, 
                                                       lr=learning_rate, 
                                                       )
        self.base_param_optimizer_2 = torch.optim.Adam(param_list, 
                                                       lr=0.05, 
                                                           )
        
        self.param_list = param_list
        self.device = params["device"]
        self.model.to(self.device)
            
    def train_one_epoch(self):
        self.model.train()
        self.model.training = True
        for step, (data, _, _) in tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train)):
            data = data.to(self.device)
            out = self.model(data)
            
            loss = 0
            features = out["features"]
            for i, (feature, mu, log_sigma) in enumerate(zip(features, self.mus_param, self.sigmas_param)):
                
                output, log_jac_dets = self.model.nf_flows[i](feature)
                
                loss += calculate_loss(self.mode, output, log_jac_dets, mu, log_sigma, i=0)

            self.optimizer.zero_grad()

            if self.params["zero_grads"]:
                self.base_param_optimizer_1.zero_grad()
                self.base_param_optimizer_2.zero_grad()    
            loss.backward()
            
            # clip all the different gradient parameters
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            torch.nn.utils.clip_grad_norm_(self.param_list, self.clip_value)
            
            if not step%self.params["freezing_interval"]==0:   
                self.optimizer.step()
                self.base_param_optimizer_1.step()
            else:
                self.base_param_optimizer_2.step()
                
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        self.model.eval()

        self.model.training = False
        with torch.no_grad():
            for ind, (data, _) in tqdm(enumerate(dataloader), total=len_dataloader):
                data = data.to(self.device)
                ret = self.model(data)
                features = ret["features"] 
                anomaly_map_list = []
                anomaly_map_list_std = []
                anomaly_map_list_max = []
                for i, (feature, mu, log_sigma) in enumerate(zip(features, self.mus_param, self.sigmas_param)):
                    output, log_jac_dets = self.model.nf_flows[i](feature)    
                    
                    log_prob = get_inference_loss(self.mode, output, mu, log_sigma)
                    
                    log_prob_channel_mean = -torch.mean(log_prob, dim=1, keepdim=True) * 0.5
                    log_prob_channel_std  = torch.std(log_prob, dim=1, keepdim=True)
                    log_prob_channel_max  = log_prob.abs().max(dim=1, keepdim=True)[0]
                    
                    # prob_channel_mean = torch.exp(log_prob_channel_mean)
                    prob_channel_std  = log_prob_channel_std
                    prob_channel_max  = log_prob_channel_max
                    
                    a_map = F.interpolate(
                        -log_prob_channel_mean,
                        size=[self.input_size, self.input_size],
                        mode="bilinear",
                        align_corners=False,
                    )
                    a_map_std = -F.interpolate(
                        prob_channel_std,
                        size=[self.input_size, self.input_size],
                        mode="bilinear",
                        align_corners=False,
                    )
                    a_map_max = -F.interpolate(
                        prob_channel_max,
                        size=[self.input_size, self.input_size],
                        mode="bilinear",
                        align_corners=False,
                    )
                    anomaly_map_list.append(a_map)
                    anomaly_map_list_std.append(a_map_std)
                    anomaly_map_list_max.append(a_map_max)

                anomaly_map_stacked = torch.stack(anomaly_map_list, dim=-1)
                anomaly_map_stacked_std = torch.stack(anomaly_map_list_std, dim=-1)
                anomaly_map_stacked_max = torch.stack(anomaly_map_list_max, dim=-1)
                 
                anomaly_map             = torch.mean(torch.mean(anomaly_map_stacked, dim=-1), dim=1).unsqueeze(1)
                anomaly_map_std         = torch.mean(torch.mean(anomaly_map_stacked_std, dim=-1), dim=1).unsqueeze(1)
                anomaly_map_max         = torch.mean(torch.mean(anomaly_map_stacked_max, dim=-1), dim=1).unsqueeze(1)
                
                segmentations_original = {"original_"+key:item for key, item in ret["segmentations"].items()}
                scores_original        = {"original_"+key:item for key, item in ret["scores"].items()}
                
                ret = {}
                ret["segmentations"] = segmentations_original
                ret["scores"]        = scores_original
                
                ret["segmentations"]["anomaly_map_channel_mean"] = anomaly_map
                ret["segmentations"]["anomaly_map_channel_std"]  = anomaly_map_std
                ret["segmentations"]["anomaly_map_stacked_max"]  = anomaly_map_max
                                
                ret["scores"]["loss_channel_mean_pixel_mean"] = anomaly_map.mean(-1).mean(-1)
                ret["scores"]["loss_channel_mean_pixel_std"]  = anomaly_map.flatten(-2).std(-1)
                
                ret["scores"]["loss_channel_std_pixel_mean"] = anomaly_map_std.mean(-1).mean(-1)
                ret["scores"]["loss_channel_std_pixel_std"]  = anomaly_map_std.flatten(-2).std(-1)
                
                ret["scores"]["loss_channel_max_pixel_mean"] = anomaly_map_max.mean(-1).mean(-1)
                ret["scores"]["loss_channel_max_pixel_std"]  = anomaly_map_max.flatten(-2).std(-1)

                if ind==0:
                    segmentations_sets = {}
                    for key, item in ret['segmentations'].items():
                        segmentations_sets[key] = item.cpu().detach()

                    scores_sets = {}
                    for key, item in ret['scores'].items():
                        scores_sets[key] = item.cpu().detach()

                else:
                    for key, item in ret['segmentations'].items():
                        segmentations_sets[key] = torch.vstack([segmentations_sets[key], item.cpu().detach()])

                    for key, item in ret['scores'].items():
                        scores_sets[key] = torch.concat([scores_sets[key], item.cpu().detach()])

        heatmap_set = segmentations_sets
        image_score_set    = scores_sets

        return heatmap_set, image_score_set
