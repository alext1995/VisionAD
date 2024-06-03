import torch
from argparse import Namespace
import math
from algos.cflow.model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from algos.cflow.utils import t2np, get_logp
from algos.cflow.custom_models import warmup_learning_rate, adjust_learning_rate
import os, time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

log_theta = torch.nn.LogSigmoid()

class CFLOW:
    def __init__(self, **params):
        self.gamma = 0.0
        self.theta = torch.nn.Sigmoid()
        self.log_theta = torch.nn.LogSigmoid()

        self.params = params
        self.device = params["device"]
        L = self.params["pool_layers"] # number of pooled layers
        
       
        params_namespace = Namespace(**self.params)
        
        if params_namespace.lr_warm:
            params_namespace.lr_warmup_from = params_namespace.lr/10.0
        if params_namespace.lr_cosine:
            eta_min = params_namespace.lr * (params_namespace.lr_decay_rate ** 3)
            params_namespace.lr_warmup_to = eta_min + (params_namespace.lr - eta_min) * (
                    1 + math.cos(math.pi * params_namespace.lr_warm_epochs / params_namespace.meta_epochs)) / 2
        else:
            params_namespace.lr_warmup_to = params_namespace.lr
            
        encoder, pool_layers, pool_dims = load_encoder_arch(params_namespace, L)
        encoder = encoder.to(self.params["device"]).eval()
        #print(encoder)
        # NF decoder
        decoders = [load_decoder_arch(params_namespace, pool_dim) for pool_dim in pool_dims]
        decoders = [decoder.to(self.params["device"]) for decoder in decoders]
        weights = list(decoders[0].parameters())
        for l in range(1, L):
            weights += list(decoders[l].parameters())
        # optimizer
        optimizer = torch.optim.Adam(weights, lr=self.params["lr"])
        self.epoch = 0
        self.encoder = encoder
        self.decoders = decoders
        self.optimizer = optimizer
        self.pool_layers = pool_layers
        self.sub_epochs = params["num_sub_epochs_per_meta_epoch"]
        self.N = params["N"]
        self.params_namespace = params_namespace
        
    # def train_one_batch(self, image_batch, sub_epoch, i, len_dataloader):
    #     P = self.params["condition_vec"]
    #     L = self.params["pool_layers"]
    #     N = self.N

    #     # warm-up learning rate
    #     lr = warmup_learning_rate(self.params_namespace, 
    #                             self.epoch, 
    #                             i+sub_epoch*len_dataloader, 
    #                             len_dataloader*self.params["sub_epochs"], 
    #                             self.optimizer)
    #     # encoder prediction
    #     image = image_batch.to(self.params["device"])  # single scale
    #     with torch.no_grad():
    #         _ = self.encoder(image)
    
    #     for l, layer in enumerate(self.pool_layers):
    #         if 'vit' in self.params["enc_arch"]:
    #             e = activation[layer].transpose(1, 2)[...,1:]
    #             e_hw = int(np.sqrt(e.size(2)))
    #             e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
    #         else:
    #             e = activation[layer].detach()  # BxCxHxW
    #         #
    #         B, C, H, W = e.size()
    #         #print(f"sub_epoch: {sub_epoch}, i: {i}, layer: {layer} ({B}, {C}, {H}, {W})")

    #         S = H*W
    #         E = B*S    
    #         #
    #         p = positionalencoding2d(P, H, W).to(self.params["device"]).unsqueeze(0).repeat(B, 1, 1, 1)
    #         c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
    #         e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
    #         perm = torch.randperm(E).to(self.params["device"])  # BHW
    #         decoder = self.decoders[l]
    #         #
    #         FIB = E//len_dataloader  # number of fiber batches

    #         if not FIB > 0:
    #             continue
    #         for f in range(FIB):  # per-fiber processing
    #             idx = torch.arange(f*len_dataloader, (f+1)*len_dataloader)
    #             c_p = c_r[perm[idx]]  # NxP
    #             e_p = e_r[perm[idx]]  # NxC
    #             if 'cflow' in self.params["dec_arch"]:
    #                 z, log_jac_det = decoder(e_p, [c_p,])
    #             else:
    #                 z, log_jac_det = decoder(e_p)
    #             #
    #             decoder_log_prob = get_logp(C, z, log_jac_det)
    #             log_prob = decoder_log_prob / C  # likelihood per dim
    #             loss = -self.log_theta(log_prob)
    #             self.optimizer.zero_grad()
    #             loss.mean().backward()
    #             self.optimizer.step()

    def train_one_epoch(self, dataloader):
        self.decoders = [decoder.train() for decoder in self.decoders]

        self.iterator = iter(dataloader)
        if self.epoch % self.sub_epochs == 0:
            self.sub_epoch = 0
            self.decoders = [decoder.train() for decoder in self.decoders]
            adjust_learning_rate(self.params_namespace, self.optimizer, self.epoch)
            self.I = len(dataloader)
            self.iterator = iter(dataloader)
        self.train_sub_epoch(dataloader, self.iterator)

        self.epoch+=1

    
    def train_sub_epoch(self, loader, iterator):
        epoch = self.epoch
        sub_epoch = self.sub_epoch
        optimizer = self.optimizer
        I = self.I
        sub_epochs = self.sub_epochs
        encoder = self.encoder
        pool_layers = self.pool_layers
        decoders = self.decoders
        N = self.N
        P = self.params_namespace.condition_vec     


        train_loss = 0.0
        train_count = 0
        for i in range(I):
            # warm-up learning rate
            warmup_learning_rate(self.params_namespace, epoch, i+sub_epoch*I, I*sub_epochs, optimizer)
            # sample batch
            try:
                image, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image, _, _ = next(iterator)
            # encoder prediction
            image = image.to(self.device)  # single scale
            with torch.no_grad():
                _ = encoder(image)

            for l, layer in enumerate(pool_layers):
                if 'vit' in self.params_namespace.enc_arch:
                    e = activation[layer].transpose(1, 2)[...,1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                else:
                    e = activation[layer].detach()  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S    
                #
                p = positionalencoding2d(P, H, W).to(self.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                perm = torch.randperm(E).to(self.device)  # BHW
                decoder = decoders[l]
                #
                FIB = E//N  # number of fiber batches
                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
                for f in range(FIB):  # per-fiber processing
                    idx = torch.arange(f*N, (f+1)*N)
                    c_p = c_r[perm[idx]]  # NxP
                    e_p = e_r[perm[idx]]  # NxC
                    if 'cflow' in self.params_namespace.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    train_loss += t2np(loss.sum())
                    train_count += len(loss)

        self.sub_epoch += 1

    # def train_meta_epoch(c, epoch, loader, encoder, decoders, optimizer, pool_layers, N):
    #     P = c.condition_vec
    #     L = c.pool_layers
    #     decoders = [decoder.train() for decoder in decoders]
    #     adjust_learning_rate(c, optimizer, epoch)
    #     I = len(loader)
    #     iterator = iter(loader)
    #     for sub_epoch in range(c.sub_epochs):
    #         train_loss = 0.0
    #         train_count = 0
    #         for i in range(I):
    #             # warm-up learning rate
    #             lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizer)
    #             # sample batch
    #             try:
    #                 image, _, _ = next(iterator)
    #             except StopIteration:
    #                 iterator = iter(loader)
    #                 image, _, _ = next(iterator)
    #             # encoder prediction
    #             image = image.to(c.device)  # single scale
    #             with torch.no_grad():
    #                 _ = encoder(image)
    #             # train decoder
    #             e_list = list()
    #             c_list = list()
    #             for l, layer in enumerate(pool_layers):
    #                 if 'vit' in c.enc_arch:
    #                     e = activation[layer].transpose(1, 2)[...,1:]
    #                     e_hw = int(np.sqrt(e.size(2)))
    #                     e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
    #                 else:
    #                     e = activation[layer].detach()  # BxCxHxW
    #                 #
    #                 B, C, H, W = e.size()
    #                 S = H*W
    #                 E = B*S    
    #                 #
    #                 p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
    #                 c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
    #                 e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
    #                 perm = torch.randperm(E).to(c.device)  # BHW
    #                 decoder = decoders[l]
    #                 #
    #                 FIB = E//N  # number of fiber batches
    #                 assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
    #                 for f in range(FIB):  # per-fiber processing
    #                     idx = torch.arange(f*N, (f+1)*N)
    #                     c_p = c_r[perm[idx]]  # NxP
    #                     e_p = e_r[perm[idx]]  # NxC
    #                     if 'cflow' in c.dec_arch:
    #                         z, log_jac_det = decoder(e_p, [c_p,])
    #                     else:
    #                         z, log_jac_det = decoder(e_p)
    #                     #
    #                     decoder_log_prob = get_logp(C, z, log_jac_det)
    #                     log_prob = decoder_log_prob / C  # likelihood per dim
    #                     loss = -log_theta(log_prob)
    #                     optimizer.zero_grad()
    #                     loss.mean().backward()
    #                     optimizer.step()
    #                     train_loss += t2np(loss.sum())
    #                     train_count += len(loss)
    #         #
    #         mean_train_loss = train_loss / train_count
    #         if c.verbose:
    #             print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))
    #     #

    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        encoder = self.encoder
        decoders = self.decoders

        pool_layers = self.pool_layers
        N = self.N
        c = self.params_namespace

        c = self.params_namespace
        P = c.condition_vec
        decoders = [decoder.eval() for decoder in decoders]
        height = list()
        width = list()

        test_dist = [list() for layer in pool_layers]
        test_loss = 0.0
        test_count = 0

        with torch.no_grad():
            for i, (image, _) in enumerate(tqdm(dataloader, total=len_dataloader)):
                # data
                image = image.to(c.device) # single scale
                _ = encoder(image)  # BxCxHxW
                # test decoder
                for l, layer in enumerate(pool_layers):
                    if 'vit' in c.enc_arch:
                        e = activation[layer].transpose(1, 2)[...,1:]
                        e_hw = int(np.sqrt(e.size(2)))
                        e = e.reshape(-1, e.size(1), e_hw, e_hw)  # BxCxHxW
                    else:
                        e = activation[layer]  # BxCxHxW
                    #
                    B, C, H, W = e.size()
                    S = H*W
                    E = B*S
                    #
                    if i == 0:  # get stats
                        height.append(H)
                        width.append(W)
                    #
                    p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                    c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                    e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                    #
                    #m = F.interpolate(mask, size=(H, W), mode='nearest')
                    #m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)  # BHWx1
                    #
                    decoder = decoders[l]
                    FIB = E//N + int(E%N > 0)  # number of fiber batches
                    for f in range(FIB):
                        if f < (FIB-1):
                            idx = torch.arange(f*N, (f+1)*N)
                        else:
                            idx = torch.arange(f*N, E)
                        #
                        c_p = c_r[idx]  # NxP
                        e_p = e_r[idx]  # NxC
                        #m_p = m_r[idx] > 0.5  # Nx1
                        #
                        if 'cflow' in c.dec_arch:
                            z, log_jac_det = decoder(e_p, [c_p,])
                        else:
                            z, log_jac_det = decoder(e_p)
                        #
                        decoder_log_prob = get_logp(C, z, log_jac_det)
                        log_prob = decoder_log_prob / C  # likelihood per dim
                        loss = -log_theta(log_prob)
                        test_loss += t2np(loss.sum())
                        test_count += len(loss)
                        test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()

        test_map = [list() for p in pool_layers]
        for l, p in enumerate(pool_layers):
            test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
            test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
            test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
            test_mask = test_prob.reshape(-1, height[l], width[l])
            test_mask = test_prob.reshape(-1, height[l], width[l])
            # upsample
            test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                size=256, mode='bilinear', align_corners=True).squeeze().numpy()
            score_map = np.zeros_like(test_map[0])

        for l, p in enumerate(pool_layers):
            score_map += test_map[l]

        score_map_flattened = score_map.reshape(score_map.shape[0], -1)
    
        score_max = score_map_flattened.max(axis=1)
        score_mean = score_map_flattened.mean(axis=1) 
        score_std = score_map_flattened.std(axis=1) 

        scores = {"max": score_max,
                "mean": score_mean,
                "std": score_std,
                }

        heatmaps = {"score_map_standard": score_map}

        return heatmaps, scores

    def save_model(self, location):
        location_weights = os.path.join(location, "weights")
        os.makedirs(location_weights, exist_ok=True)
        for ind, decoder in enumerate(self.decoders):
            torch.save(decoder.state_dict(), os.path.join(location_weights, f"{ind}.pt"))
        
    def load_model(self, location):
        location_weights = os.path.join(location, "weights")
        for ind, decoder in enumerate(self.decoders):
            decoder.load_state_dict(torch.load(os.path.join(location_weights, f"{ind}.pt")))