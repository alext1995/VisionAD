import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from torchvision import transforms

CHECKPOINT_DIR = "_fastflow_experiment_checkpoints"

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
]

backbone_name_shapes = {"deit_base_distilled_patch16_384": (384, 384),
                        "cait_m48_448": (256, 256), # need to check this one
                        "resnet18": (256, 256),
                        "wide_resnet50_2": (256, 256), # need to check this one 
}
        
def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )
    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes

class FastFlow(nn.Module):
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        device, 
        conv3x3_only=False,
        hidden_ratio=1.0,
        specific_model=None, 
        **kwargs,
    ):
        

        print("device; ", device)
        self.device = device
        self.image_model_size = backbone_name_shapes[backbone_name][0]
        super(FastFlow, self).__init__()
        assert (
            backbone_name in SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(SUPPORTED_BACKBONES)
        self.specific_model = False
        self.return_features = False

        if specific_model:
            self.specific_model = True
            self.feature_extractor = specific_model
            scales = self.feature_extractor.scales
            channels = self.feature_extractor.channels
            self.norms = nn.ModuleList()
        elif backbone_name in [BACKBONE_CAIT, BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, 
                                                       pretrained=True,
                                                       )
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()

            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(self.image_model_size / scale), int(self.image_model_size / scale)],
                        elementwise_affine=True,
                    )
                )
        self.feature_extractor = self.feature_extractor.to(self.device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(self.image_model_size / scale), int(self.image_model_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size
        self.backbone_name = backbone_name

    def forward(self, x):
        self.feature_extractor.eval()
        if self.specific_model:
            features = self.feature_extractor(x)
            
        elif isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = transforms.Resize((self.image_model_size, self.image_model_size))(x)
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, :, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            
            x = x[..., -(self.image_model_size//16)**2:].reshape(N, C, self.image_model_size // 16, self.image_model_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            features = self.feature_extractor(x)
            features = [self.norms[i](feature) for i, feature in enumerate(features)]
        
        loss = 0
        outputs = []

        # if self.return_features:
        #     return features

        losses = [0]
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            losses.append(loss)
            if i == 0:
                loss_per_image = 0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
                
            else:
                loss_per_image += 0.5 * torch.sum(output**2, dim=(-1, -2, -3)) - log_jac_dets
                
            outputs.append(output)

        ret = {"loss": loss, "features": features}
        
        if not self.training:
            # save_path = r"C:\Users\alext\Documents\PhD work\crack detection\experiments\debugging_ff"
            # os.makedirs(save_path, exist_ok=True)
            # for i, item in enumerate(outputs):
            #     #print("item.shape, item.dtype: ", item.shape, item.dtype)
            #     out_path = os.path.join(save_path, f"{i}.pt")
            #     torch.save(item, out_path)
            #     #print(f"Saved item in to:\n{out_path}")
            #     # with open(rf"C:\Users\alext\Documents\PhD work\crack detection\experiments\debugging_ff\test_file_anomaly_map_stacked_{i}.pke", "wb") as f:
            #     #     pickle.dump(item, f)
                
            anomaly_map_list = []
            anomaly_map_list_std = []
            anomaly_map_list_max = []
            prob = 0


            for output in outputs:
                log_prob_channel_mean = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                log_prob_channel_std  = torch.std(output, dim=1, keepdim=True)
                log_prob_channel_max  = output.max(dim=1, keepdim=True)[0]

                prob_channel_mean = torch.exp(log_prob_channel_mean)
                prob_channel_std  = log_prob_channel_std
                prob_channel_max  = torch.exp(log_prob_channel_max)

                a_map = F.interpolate(
                    -prob_channel_mean,
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
            
            # ret = {}
            ret["segmentations"] = {} 
            ret["scores"] = {} 
            
            ret["segmentations"]["anomaly_map_channel_mean"] = anomaly_map
            #ret["segmentations"]["anomaly_map_channel_std"]  = anomaly_map_std
            #ret["segmentations"]["anomaly_map_stacked_max"]  = anomaly_map_max
                            
            ret["scores"]["loss_channel_mean_pixel_mean"] = anomaly_map.mean(-1).mean(-1)
            #ret["scores"]["loss_channel_mean_pixel_std"]  = anomaly_map.flatten(-2).std(-1)
            
            #ret["scores"]["loss_channel_std_pixel_mean"] = anomaly_map_std.mean(-1).mean(-1)
            #ret["scores"]["loss_channel_std_pixel_std"]  = anomaly_map_std.flatten(-2).std(-1)
            
            #ret["scores"]["loss_channel_max_pixel_mean"] = anomaly_map_max.mean(-1).mean(-1)
            #ret["scores"]["loss_channel_max_pixel_std"]  = anomaly_map_max.flatten(-2).std(-1)
        
            # for output in outputs:
            #     log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
            #     prob = torch.exp(log_prob)
            #     # print("prob.shape: ", prob.shape)
            #     a_map = F.interpolate(
            #         -prob,
            #         size=[self.input_size, self.input_size],
            #         mode="bilinear",
            #         align_corners=False,
            #     )
            #     a_map_std = -F.interpolate(
            #         -torch.std(output**2, dim=1, keepdim=True) * 0.5,
            #         size=[self.input_size, self.input_size],
            #         mode="bilinear",
            #         align_corners=False,
            #     )
            #     anomaly_map_list.append(a_map)
            #     anomaly_map_list_std.append(a_map_std)
            #     prob += log_prob
            
            # anomaly_map_stacked = torch.stack(anomaly_map_list, dim=-1)
            
            # ret["segmentations"] = {} 
            # ret["scores"] = {} 
            # anomaly_map = torch.mean(torch.mean(anomaly_map_stacked, dim=-1), dim=1).unsqueeze(1)
            # ret["segmentations"]["anomaly_map_mean"] = anomaly_map
            # array_flattened = anomaly_map_stacked.flatten(1)
            # log_pdf  = Normal(loc=0, scale=1).log_prob(value = array_flattened).mean(axis=1)
          
            # ret["scores"]["loss"] = loss_per_image.cpu()
            # ret["scores"]["loss_mean"] = anomaly_map.mean(-1).mean(-1)
            # ret["scores"]["loss_std"] = anomaly_map.flatten(-2).std(-1)

        return ret
