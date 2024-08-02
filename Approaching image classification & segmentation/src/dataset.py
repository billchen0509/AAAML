# dataset.py

import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

# sometimes, images throw an error if they are truncated
# this code handles the error
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset:
    '''
    A general classification dataset class that loads images and their labels
    '''
    def __init__(self, image_paths, targets, resize = None, augmentation = None):
        '''
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param resize: tuple or None
        :param augmentation: albumentations augmentations
        '''
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentation = augmentation
    
    def __len__(self):
        # return  the total number of samples in the dataset
        return len(self.image_paths)
    
    def __getitem__(self, item):
        '''
        For a given "item" index, return the image and the targets
        '''
        # use PIL to open the image
        image = Image.open(self.image_paths[item])
        # convert the image to RGB
        iamge = image.convert("RGB")
        # grab the targets for this image
        targets = self.targets[item]

        # resize if needed
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]),
                resample = Image.BILINEAR
            )
            # convert the image to numpy array
            image = np.array(image)
        if self.augmentation is not None:
            augmented = self.augmentation(image = image)
            image = augmented["image"]
        
        # PyTorch expects CHW instead of HWC
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # return the image and targets
        return {
            "image": torch.tensor(image, dtype = torch.float),
            "targets": torch.tensor(targets, dtype = torch.long)
        }