# engine.py

import torch
import torch.nn as nn

def train(data_loader, model, optimizer, device):
    '''
    This function is used for training the model
    '''
    # set the model to training mode
    model.train()
    
    for data in data_loader:
        # fetch the input and output from the data loader
        reviews = data['review']
        targets = data['target']
        
        # move the input and output to the GPU
        reviews = reviews.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        # zero the gradients
        optimizer.zero_grad()
        
        # make predictions
        predictions = model(reviews)

        # calculate the loss
        loss = nn.BCEWithLogitsLoss()(predictions, targets.view(-1, 1))

        # compute gradient of loss w.r.t. all parameters
        loss.backward()

        # single optimization step
        optimizer.step()

def evaluate(data_loader, model, device):
    final_predictions = []
    final_targets = []

    # put the model in eval mode
    model.eval()

    with torch.no_grad():
        for data in data_loader:
            reviews = data['review']
            targets = data['target']

            reviews = reviews.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            predictions = model(reviews)

            predictions = predictions.cpu().numpy().tolist()
            targets = data['target'].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)
    return final_predictions, final_targets