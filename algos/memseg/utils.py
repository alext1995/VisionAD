from typing import Union, List, Tuple
import imgaug.augmenters as iaa
import cv2 
import numpy as np
import torch
import os
from glob import glob
from algos.memseg.data.perlin import rand_perlin_2d_np
from einops import rearrange
from PIL.PngImagePlugin import PngImageFile
# from torch.utils.data import Dataset

class SyntheticAnomaly:
    def __init__(
        self, 
        resize: Tuple[int, int] = (256,256),
        texture_source_dir: str = None, 
        structure_grid_size: str = 8,
        transparency_range: List[float] = [0.15, 1.],
        perlin_scale: int = 6, 
        min_perlin_scale: int = 0, 
        perlin_noise_threshold: float = 0.5,
        always_add_anomaly = False, 
    ):  
        if not texture_source_dir:
            print("Fill in 'texture_source_dir' with location of DTD dataset")
        texture_source_dir = os.path.join(texture_source_dir, "images")
        self.texture_source_dir = texture_source_dir
        self.texture_source_file_list = glob(os.path.join(texture_source_dir,'*/*'))
        self.transparency_range = transparency_range
        self.perlin_scale = perlin_scale
        self.min_perlin_scale = min_perlin_scale
        self.perlin_noise_threshold = perlin_noise_threshold
        self.structure_grid_size = structure_grid_size
        self.resize = resize
        self.switch = True
        self.always_add_anomaly = always_add_anomaly

    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        
        return aug
        
    def generate_anomaly(self, img: PngImageFile) -> List[np.ndarray]:
        '''
        step 1. generate mask
            - target foreground mask
            - perlin noise mask
            
        step 2. generate texture or structure anomaly
            - texture: load DTD
            - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation, 
            and hue on the input image  𝐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid 
            and randomly arranged to obtain the disordered image  𝐼 
            
        step 3. blending image and anomaly source
        '''
        
        img = np.array(img)
        if len(img.shape)==2:
            img = np.stack([img]*3).transpose(2, 1, 0)

        # step 1. generate mask
        img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))
        
        if self.always_add_anomaly or self.switch:
            self.switch = False
            
            ## target foreground mask
            target_foreground_mask = self.generate_target_foreground_mask(img=img)
            
            ## perlin noise mask
            perlin_noise_mask = self.generate_perlin_noise_mask()
            
            ## mask
            mask = perlin_noise_mask * target_foreground_mask
            mask_expanded = np.expand_dims(mask, axis=2)
            
            # step 2. generate texture or structure anomaly
            
            ## anomaly source
            anomaly_source_img = self.anomaly_source(img=img)
            
            ## mask anomaly parts
            factor = np.random.uniform(*self.transparency_range, size=1)[0]
            anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
            
            # step 3. blending image and anomaly source
            anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img
            
            return (anomaly_source_img.astype(np.uint8), mask)
        else:
            self.switch = True
            return (img.astype(np.uint8), np.zeros(img.shape[:-1], dtype=np.float32))
        
    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)

        # invert mask for foreground mask
        target_foreground_mask = -(target_background_mask - 1)
        
        return target_foreground_mask
    
    def generate_perlin_noise_mask(self) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np((self.resize[0], self.resize[1]), (perlin_scalex, perlin_scaley))
        
        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)
        
        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold, 
            np.ones_like(perlin_noise), 
            np.zeros_like(perlin_noise)
        )
        
        return mask_noise
    
    def anomaly_source(self, img: np.ndarray) -> np.ndarray:
        p = np.random.uniform()
        if p < 0.5:
            # TODO: None texture_source_file_list
            anomaly_source_img = self._texture_source()
        else:
            anomaly_source_img = self._structure_source(img=img)
            
        return anomaly_source_img
        
    def _texture_source(self) -> np.ndarray:
        if len(self.texture_source_file_list)==0:
            print(f"No images found in {self.texture_source_dir}.")
            print(f"Ensure config: 'texture_source_dir' points to the root directory of the DTD dataset")
        idx = np.random.choice(len(self.texture_source_file_list))
        texture_source_img = cv2.imread(self.texture_source_file_list[idx])
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img, dsize=(self.resize[1], self.resize[0])).astype(np.float32)
        
        return texture_source_img
        
    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        structure_source_img = self.rand_augment()(image=img)
        
        assert self.resize[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = self.resize[1] // self.structure_grid_size
        grid_h = self.resize[0] // self.structure_grid_size
        
        structure_source_img = rearrange(
            tensor  = structure_source_img, 
            pattern = '(h gh) (w gw) c -> (h w) gw gh c',
            gw      = grid_w, 
            gh      = grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor  = structure_source_img[disordered_idx], 
            pattern = '(h w) gw gh c -> (h gh) (w gw) c',
            h       = self.structure_grid_size,
            w       = self.structure_grid_size
        ).astype(np.float32)
        
        return structure_source_img
        
    def __len__(self):
        return len(self.file_list)
    
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count