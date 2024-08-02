# train_2.py
import os
import sys
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp 
import torch.nn as nn
import torch.optim as optim
from apex import amp
from collections import OrderedDict 
from sklearn import model_selection 
from tqdm import tqdm
from torch.optim import lr_scheduler
from dataset import SIIMDataset

# training csv file path
TRAINING_CSV = "../input/train_pneumothorax.csv"
# training and test batch sizes
TRAINING_BATCH_SIZE = 16 
TEST_BATCH_SIZE = 4
# number of epochs
EPOCHS = 10

ENCODER_WEIGHTS = 'imagenet'

# get the device
DEVICE = 'cuba'

def train(dataset, data_loader, model, criterion, optimizer):
    model.train()
    num_batches = int(len(data_loader) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)
    for data in tk0:
        inputs = data["image"].to(DEVICE)
        targets = data["mask"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
    tk0.close()

def evaluate(dataset, data_loader, model, criterion):
    model.eval()
    final_loss = 0
    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)
    with torch.no_grad(): 
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]
            inputs = inputs.to(DEVICE, dtype=torch.float) 
            targets = targets.to(DEVICE, dtype=torch.float) 
            output = model(inputs)
            loss = criterion(output, targets)
            # add loss to final loss
            final_loss += loss
    tk0.close()
    return final_loss / num_batches

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_CSV)

    # split data into training and validation
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.1, random_state=42
    )

    # training and validation images lists/arrays
    train_images = df_train.image_id.values
    valid_images = df_valid.image_id.values

    # fetch unet model
    model = smp.Unet(
        encoder_name = ENCODEER,
        encoder_weights = ENCODER_WEIGHTS,
        classes = 1,
        activation = NOne)
    
    prep_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # set the model to device
    model.to(DEVICE)

    train_dataset = SIIMDataset(
        training_iamges,
        transform = True,
        preprocessing_fn = prep_fn
    )

    # wwrap the dataset in DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = TRAINING_BATCH_SIZE,
        shuffle = True,
        num_workers = 12
    )

    valid_loader = SIMMDataset(
        valid_images,
        transform = False,
        preprocessing_fn = prep_fn
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = TEST_BATCH_SIZE,
        shuffle = False,
        num_workers = 12
    )

    # define the criterion
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # define the scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, verbose=True
    )

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # some logging
    print(f'training batch size: {TRAINING_BATCH_SIZE}')
    print(f'validation batch size: {TEST_BATCH_SIZE}')
    print(f'epochs: {EPOCHS}')
    print(f'image size: {IMAGE_SIZE}')
    
    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch}')

        train(
            train_dataset,
            train_loader,
            model,
            criterion,
            optimizer
        )
        val_log = evaluate(
            valid_dataset,
            valid_loader,
            model,
            criterion
        )
        