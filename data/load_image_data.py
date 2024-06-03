import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import traceback
import torch


class ImageDataset(Dataset):
    def __init__(self, **kwargs):

        folder_path = kwargs["folder_path"] 
        label = kwargs["label"] 
        test = kwargs["test"]
        pre_pro_transforms = kwargs["pre_pro_transforms"]
        training_transformations = kwargs["training_transformations"]
        identical_transforms = kwargs["identical_transforms"]
        start_ind = kwargs["start_ind"] 
        end_ind = kwargs["end_ind"]
        novel = kwargs["novel"]
        unnormalise = kwargs["unnormalise"]
        skip = kwargs["skip"]
        
        self.pre_normalisation_transform = None
        self.image_names = sorted(os.listdir(folder_path))[slice(start_ind, end_ind)][::skip]
        self.folder_path = folder_path
        self.label = label
        self.test = test
        self.pp_transforms = pre_pro_transforms
        self.identical_transforms = identical_transforms 
        if self.test:
            self.training_transformations = None
        else:
            self.training_transformations = training_transformations
        self.label_dict = {"regular": 0, "novel": 1}
        self.unnormalise = unnormalise
        self.novel = novel
        if 'folder_path_novel_masks' in kwargs:
            self.pixel_labels = True
            self.mask_path = kwargs['folder_path_novel_masks']
        else:
            self.pixel_labels = False

    def __len__(self,):
        return len(self.image_names)

    def __getitem__(self, idx):
        try:
            path = os.path.join(self.folder_path, self.image_names[idx])
            image = Image.open(path)

            ## if enabled, apply the pre_normalisation_transform
            ## this is useful for algorithms that add synthetic defects to images
            ## these algorithms just pass a callback, and will recieve the amended image
            ## and the relevant meta data (such as a mask) in pre_normalisation_transform_data
            if self.pre_normalisation_transform:
                image, pre_normalisation_transform_data = self.pre_normalisation_transform(image)
            else:
                pre_normalisation_transform_data = 0

            if self.pixel_labels and self.novel:
                ## if testing and pixel labels available
                mask_path = os.path.join(self.mask_path, 
                                         self.image_names[idx])

                ## load the target path
                mask_image_paths = [mask_path, 
                                    mask_path[:-4]+".png", 
                                    mask_path[:-4]+"_mask.png",
                                    mask_path[:-4]+".jpg"]
                for item in mask_image_paths:
                    if os.path.exists(item):
                        mask_image = Image.open(item)
                        break
             
                ## if not a tensor already, tranform the image to a tensor
                try:
                    image = transforms.ToTensor()(image)
                except:
                    print("Failed to make the input into a tensor")
                    print("image type:")
                    print(image)
                    pass
                
                if image.shape[0]==4:
                    image = Image.open(path)
                    image = image.convert('RGB')
                    image = transforms.ToTensor()(image)
                    
                # if it has 4 channels remove the alpha
                mask_image = transforms.ToTensor()(mask_image)
                mask_image[mask_image>0] = 1
                mask_image[mask_image<=0] = 0

                ## rehape the mask if needed (ideally won't be needed)
                image_size = image.shape[-2:]
                mask_image = transforms.Resize(image_size, antialias=True)(mask_image)

                ## if black and white, make it 3 channels
                if image.shape[0]==1:
                    image = torch.vstack([image]*3)

                ## do the preprocessing transforms
                image = self.pp_transforms(image)

                ## if identical transforms exist, do them
                if self.identical_transforms:
                    image      = self.identical_transforms(image)
                    mask_image = self.identical_transforms(mask_image)

                        

                ## return the image, mask, path_name, and pre_normalisation_transform_data
                return image, mask_image, self.image_names[idx], pre_normalisation_transform_data
            else:
                ## if training, or no pixel labels available

                ## if not a tensor already, tranform the image to a tensor
                try:
                    image = transforms.ToTensor()(image)
                except:
                    pass
                
                # if it has 4 channels, remove the alpha
                if image.shape[0]==4:
                    image = Image.open(path)
                    image = image.convert('RGB')
                    image = transforms.ToTensor()(image)

                if image.shape[0]==1:
                    image = torch.vstack([image]*3)

                ## do the preprocessing transformations
                if self.pp_transforms:
                    image = self.pp_transforms(image)

                ## if identical transforms exist, do them
                if self.identical_transforms:
                    image = self.identical_transforms(image)

                ## do the training transforms
                if self.training_transformations:
                    image = self.training_transformations(image)

                ## return the image, path, and pre_normalisation_transform_data
                return image, self.image_names[idx], pre_normalisation_transform_data


        except Exception as e:
            print(traceback.format_exc())
            print(path)
            print(mask_path)
            print("error loading the data")
            print(e)
            return None

def create_dataloader(**kwargs):
    pin_memory = True
    if "device" not in kwargs:
        kwargs["device"] = "cpu"

    if kwargs["device"]=="cpu":
        pin_memory = False
    dataset = ImageDataset(**kwargs)
    image_dataloader = DataLoader(dataset, kwargs["batch_size"], kwargs["shuffle"], 
                                  pin_memory=pin_memory,
                                  )
    return image_dataloader

def create_dataloaders(folder_path_regular_train,
                       folder_path_regular_test,
                       folder_path_novel_test, 
                       pre_pro_transforms,
                       training_transformations,
                        identical_transforms,
                        batch_size,
                        regular_train_start,
                        regular_train_end,
                        regular_test_start, 
                        regular_test_end,
                        novel_test_start, 
                        novel_test_end, 
                        shuffle,
                        unnormalise,
                        skip_train_regular=1, 
                        skip_test_regular=1, 
                        skip_test_novel=1,
                        **kwargs):
    if 'test_batch_size' in kwargs:
        test_batch_size = kwargs["test_batch_size"] 
    else:
        test_batch_size = batch_size
    
    label_regular       = "regular"
    label_cracked       = "novel"

    dataloader_regular_train = create_dataloader(folder_path = folder_path_regular_train, 
                                                 label = label_regular, 
                                                 test = False, 
                                                 batch_size = batch_size, 
                                                 pre_pro_transforms = pre_pro_transforms, 
                                                 training_transformations = training_transformations, 
                                                 identical_transforms= identical_transforms,
                                                 shuffle = shuffle,
                                                 start_ind = regular_train_start, 
                                                 end_ind = regular_train_end,
                                                 novel = False,
                                                 unnormalise = unnormalise,
                                                 skip = skip_train_regular,
                                                 **kwargs)

    dataloader_regular_test = create_dataloader(folder_path = folder_path_regular_test, 
                                                label = label_regular, 
                                                test = True, 
                                                batch_size = test_batch_size, 
                                                pre_pro_transforms = pre_pro_transforms, 
                                                training_transformations = training_transformations, 
                                                identical_transforms= identical_transforms,
                                                shuffle = shuffle,
                                                start_ind = regular_test_start, 
                                                end_ind = regular_test_end,
                                                novel = False,
                                                unnormalise = unnormalise,
                                                skip = skip_test_regular,
                                                **kwargs)

    dataloader_cracked_test = create_dataloader(folder_path = folder_path_novel_test, 
                                                label = label_cracked, 
                                                test = True, 
                                                batch_size = test_batch_size, 
                                                pre_pro_transforms = pre_pro_transforms, 
                                                training_transformations = training_transformations,  
                                                identical_transforms= identical_transforms,
                                                shuffle = shuffle,
                                                start_ind = novel_test_start, 
                                                end_ind = novel_test_end,
                                                novel = True,
                                                unnormalise = unnormalise,
                                                skip = skip_test_novel,
                                                **kwargs)
     
    return dataloader_regular_train, dataloader_regular_test, dataloader_cracked_test

def make_unnormalise(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225]):
    '''
    Useful for plotting and debugging in Jupyter notebooks
    This function reverses the normalisation, so an image
    out of the dataloader can be converted back to normal colours
    '''
    return transforms.Compose([transforms.Normalize(mean = [ 0.]*len(mean),
                                                    std = 1/np.array(std)),
                               transforms.Normalize(mean = -np.array(mean),
                                                    std = [ 1.]*len(std)),
                               ])
