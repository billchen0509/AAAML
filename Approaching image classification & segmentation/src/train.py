# train.py

import os

import pandas as pd
import numpy as np

import albumentations
import torch

from sklearn import metrics
from sklearn.model_selection import train_test_split

import dataset
import engine
from model import get_model

if __name__ == "__main__":
    data_path = '../input/'

    # cuda or cpu
    device = 'cuda'

    epochs = 10

    # load the data
    df = pd.read_csv(os.path.join(data_path, 'train.csv'))

    # fetch all image ids
    images = df.ImageId.values.tolist()

    # a list of image locations
    images = [os.path.join(data_path, 'train_png', i + '.png') for i in images]

    # binary targets 
    targets = df.target.values

    # fetch out the model
    model = get_model(pretrained = True)

    # move the model to device
    model.to(device)

    # mean and std values for the pretrained model
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # albumentations for the training
    # albumentations is a library for image augmentations
    # alwasys_apply = true applies normalization
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value = 255.0, always_apply = True)
        ]
    )

    # split the data into training and validation
    train_images, valid_images, train_targets, valid_targets = train_test_split(
        images, targets, stratify = targets, random_state = 42
    )

    # fetch the ClassificationDataset class
    train_dataset = dataset.ClassificationDataset(
        image_paths = train_images,
        targets = train_targets,
        resize = (227, 227),
        augmentations = aug
    )

    # fetch the DataLoader class
    # torch DataLoader class is used to load data in batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = 16, shuffle = True, num_workers = 4
    )

    # same for the validation
    valid_dataset = dataset.ClassificationDataset(
        image_paths = valid_images,
        targets = valid_targets,
        resize = (227, 227),
        augmentations = aug
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size = 16, shuffle = False, num_workers = 4
    )

    # simple Adam optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)

    # train the model
    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device)
        predictions, valid_targets = engine.evaluate(valid_loader, model, device)
        roc_auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch = {epoch}, ROC-AUC = {roc_auc}")
        