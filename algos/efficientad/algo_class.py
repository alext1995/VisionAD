from algos.model_wrapper import ModelWrapper
import numpy as np
from torchvision import transforms
from torch import nn
from torchvision.datasets import ImageFolder
import os
import itertools
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from algos.efficientad.common import get_autoencoder, get_pdn_small, get_pdn_medium
import pickle


class WrapperEfficientAD(ModelWrapper):
    def __init__(self, load_params=True, **kwargs):
        super(ModelWrapper, self).__init__()
        if load_params:
            self.load_model_params(**kwargs)
        self.default_segmentation = "method_1"
        self.default_score = "method_1"
        
    def load_model_params(self, **params):
        self.__name__ = "EfficientAD"
        self.params = params
        self.device = self.params["device"]
        self.default_transform = transforms.Compose([transforms.Resize((params["input_size"], 
                                                                        params["input_size"])),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                    ])
        self.transform_ae = transforms.RandomChoice([transforms.ColorJitter(brightness=0.2),
                                                     transforms.ColorJitter(contrast=0.2),
                                                     transforms.ColorJitter(saturation=0.2)
                                                    ])
            
        out_channels = 384
        self.out_channels = out_channels
        image_size = 256

        # create models
        if self.params["model_size"] == 'small':
            weights      = os.path.join(os.getcwd(), "algos", "efficientad", "models", "teacher_small.pth")
            self.teacher = get_pdn_small(out_channels)
            self.student = get_pdn_small(2 * out_channels)
        elif self.params["model_size"] == 'medium':
            weights      = os.path.join(os.getcwd(), "algos", "efficientad", "models", "teacher_medium.pth")
            self.teacher = get_pdn_medium(out_channels)
            self.student = get_pdn_medium(2 * out_channels)
        else:
            raise Exception()
        
        state_dict = torch.load(weights, 
                                map_location='cpu')
        self.teacher.load_state_dict(state_dict)
        self.autoencoder = get_autoencoder(out_channels)
        
        self.teacher.eval()
        self.student.train()
        self.autoencoder.train()
        
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        self.autoencoder = self.autoencoder.to(self.device)
        
        self.optimizer = torch.optim.Adam(itertools.chain(self.student.parameters(),
                                                          self.autoencoder.parameters()),
                                                          lr=1e-4, 
                                                          weight_decay=1e-5)
        self.epoch = 0
        self.number_steps = 0 # 70000 recommended
        
    def pre_train_setup(self):

        def do_ae_transform(x):
            if len(np.array(x).shape)==2:
                x_ = np.concatenate([np.array(x)[:,:,None]]*3, axis=-1)
                default_transform = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize((self.params["input_size"], 
                                                                           self.params["input_size"])),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                             std=[0.229, 0.224, 0.225])
                                                    ])
            else:
                default_transform = self.default_transform
                x_ = x
            return x, self.transform_ae(default_transform(x_))

        self.teacher_mean, self.teacher_std = teacher_normalization(self.teacher, 
                                                                    self.dataloader_train, 
                                                                    self.device)
        
        self.dataloader_train.dataset.pre_normalisation_transform = lambda x: do_ae_transform(x)
        
        if self.params["imagenet_train_path"]:
            self.image_penalty = True
            penalty_transform = transforms.Compose([
                transforms.Resize((2 * self.params["input_size"], 2 * self.params["input_size"])),
                transforms.RandomGrayscale(0.3),
                transforms.CenterCrop(self.params["input_size"]),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std  = [0.229, 0.224, 0.225])
            ])
            penalty_set = ImageFolderWithoutTarget(self.params["imagenet_train_path"],
                                                   transform=penalty_transform)
            penalty_loader = DataLoader(penalty_set, 
                                        batch_size=self.dataloader_train.batch_size, 
                                        shuffle=True,
                                        num_workers=4, 
                                        pin_memory=True)
            self.penalty_dl = InfiniteDataloader(penalty_loader)
        else:
            self.image_penalty = False
            self.penalty_dl = itertools.repeat(None)
            
        no_steps = self.params["epochs"]*len(self.dataloader_train)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         step_size=int(0.95 * no_steps), 
                                                         gamma=0.1)
            
    def train_one_epoch(self,):
        if self.epoch == 0:
            self.pre_train_setup()
            
        for (image, _, image_ae), penalty_image in zip(tqdm(self.dataloader_train), self.penalty_dl):
            image_st = image.to(self.device)
            image_ae = image_ae.to(self.device)
            
            with torch.no_grad():
                teacher_output_st = self.teacher(image_st)
                teacher_output_st = (teacher_output_st - self.teacher_mean) / self.teacher_std
            
            student_output_st = self.student(image_st)[:, :self.out_channels]
            distance_st = (teacher_output_st - student_output_st) ** 2
            d_hard = torch.quantile(distance_st, q=self.params["q"])#0.999
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            
            if self.image_penalty:
                student_output_penalty = self.student(penalty_image.to(self.device))[:, :self.out_channels]
                loss_penalty = torch.mean(student_output_penalty**2)
                loss_st = loss_hard + loss_penalty
            else:
                loss_st = loss_hard
                
            ae_output = self.autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = self.teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - self.teacher_mean) / self.teacher_std
            student_output_ae = self.student(image_ae)[:, self.out_channels:]    
            distance_ae = (teacher_output_ae - ae_output)**2
            distance_stae = (ae_output - student_output_ae)**2    
                
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            loss_total = loss_st + loss_ae + loss_stae
            
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            self.number_steps+=1 
            
        self.epoch+=1
        
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        length = 0
        
        self.teacher.eval()
        self.student.eval()
        self.autoencoder.eval()
            
        map_combineds = []
        map_sts       = []
        map_aes       = []
        for (image, _) in dataloader:
            map_combined, map_st, map_ae = predict(image.to(self.device), 
                                                    self.teacher, 
                                                    self.student, 
                                                    self.autoencoder, 
                                                    self.teacher_mean, 
                                                    self.teacher_std,
                                                    self.out_channels,
                                                    q_st_start = self.q_st_start, 
                                                    q_st_end   = self.q_st_end, 
                                                    q_ae_start = self.q_ae_start, 
                                                    q_ae_end   = self.q_ae_end)
            map_combineds.append(map_combined)
            map_sts.append(map_st)
            map_aes.append(map_ae)
            
        map_combineds = torch.concat(map_combineds).cpu()
        map_sts       = torch.concat(map_sts).cpu()
        map_aes = torch.concat(map_aes).cpu()
        
        map_combineds = transforms.Resize((256, 256))(map_combineds)
        map_sts = transforms.Resize((256, 256))(map_sts)
        map_aes = transforms.Resize((256, 256))(map_aes)
        
        print(map_combineds.shape)
        print(map_sts.shape)
        print(map_aes.shape)
        
        a_pixel_ret  = {"method_1":map_combineds, 
                        "method_2":map_sts, 
                        "method_3":map_aes}
        a_image_ret  = {"method_1":map_combineds.max(-1)[0].max(-1)[0].cpu(), 
                        "method_2":map_sts.max(-1)[0].cpu(), 
                        "method_3":map_aes.max(-1)[0].cpu()}
        return a_pixel_ret, a_image_ret
    
    def pre_eval_processing(self):
        self.q_st_start, self.q_st_end, self.q_ae_start, self.q_ae_end = map_normalization(
                                                                            validation_loader=self.dataloader_train, 
                                                                            teacher=self.teacher,
                                                                            student=self.student, 
                                                                            autoencoder=self.autoencoder,
                                                                            teacher_mean=self.teacher_mean, 
                                                                            teacher_std=self.teacher_std,
                                                                            device=self.device,
                                                                            out_channels=self.out_channels)
        
        

    def save_model(self, location):
        with open(os.path.join(location, "teacher_mean_std.pkl"), "wb") as f:
            pickle.dump((self.teacher_mean, 
                         self.teacher_std,
                         self.q_st_start,
                         self.q_st_end,
                         self.q_ae_start,
                         self.q_ae_end), f)

        self.save_params(location)
        torch.save(self.student.state_dict(), os.path.join(location,
                                             'student.pk'))
        torch.save(self.autoencoder.state_dict(), os.path.join(location,
                                             'autoencoder.pk'))

    def load_model(self, location):
        with open(os.path.join(location, "teacher_mean_std.pkl"), "rb") as f:
            (self.teacher_mean, 
             self.teacher_std,
             self.q_st_start,
             self.q_st_end,
             self.q_ae_start,
             self.q_ae_end) = pickle.load(f)

        params = self.load_params(location)
        self.load_model_params(**params)


        self.student.load_state_dict(torch.load(os.path.join(location, 
                                                             'student.pk')))
        self.autoencoder.load_state_dict(torch.load(os.path.join(location, 
                                                                'autoencoder.pk')))
        
@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, device, out_channels):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _, _ in tqdm(validation_loader):
     
        image = image.to(device)
        map_combined, map_st, map_ae = predict(image=image, 
                                                teacher=teacher, 
                                                student=student,
                                                autoencoder=autoencoder, 
                                                teacher_mean=teacher_mean,
                                                teacher_std=teacher_std,
                                                out_channels=out_channels)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std, out_channels,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def teacher_normalization(teacher, train_loader, device):

    mean_outputs = []
    for train_image, _, _ in tqdm(train_loader):
        
        train_image = train_image.to(device)
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _, _ in tqdm(train_loader):
        
        train_image = train_image.to(device)
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)