# dataset_2.py

import os
import glob
import torch
import numpy as np 
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from collections import defaultdict 
from torchvision import transforms
from albumentations import (Compose,OneOf, RandomBrightnessContrast, RandomGamma, ShiftScaleRotate,) 


ImageFile.LOAD_TRUNCATED_IMAGES = True

class SIIMDataset(torch.utils.data.Dataset):
    def __init__(
        self, image_ids, transform = True, preprocessing_fn = None
    ):
        self.data = defaultdict(dict)
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn

        self.aug = Compose(
                [
                    ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.8
                    ), OneOf(
                            [
                                RandomGamma(
                                    gamma_limit=(90, 110) ),
                                    RandomBrightnessContrast( brightness_limit=0.1, contrast_limit=0.1), 
                            ],
                            p=0.5,
                    ),
                ] 
            )
        
        for image_id in tqdm(image_ids):
            files = glob.glob(os.path.join(TRAIN_PATH, image_id, "*.jpg"))   
            self.data[couter] = {
                'image_path':os.path.join(TRAIN_PATH, image_id, "image.jpg"),
                'mask_path':os.path.join(TRAIN_PATH, image_id, "mask.jpg"),
            }
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        image = Image.open(self.data[item]['image_path'])
        mask = Image.open(self.data[item]['mask_path'])
        
        image = image.convert("RGB")
        mask = mask.convert("L")
        
        if self.transform:
            image = self.aug(image = np.array(image))['image']
        
        image = np.array(image)
        mask = np.array(mask)
        
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image)
        
        return {
            'image':torch.tensor(image, dtype = torch.float),
            'mask':torch.tensor(mask, dtype = torch.float)
        }
