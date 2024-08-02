# engine.py
import torch 
import torch.nn as nn

from tqdm import tqdm

def train(data_loader, model, optimizer, device):
    '''
    This function does the training for one epoch
    '''
    # put the model in training mode
    model.train()
    
    # go over all the batches
    for data in tqdm(data_loader, total = len(data_loader)):
        # move the data to device
        inputs = data["image"].to(device)
        targets = data["targets"].to(device)
        
        # zero the gradients
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(inputs)
        
        # calculate the loss
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
        
        # backward pass
        loss.backward()
        
        # step the optimizer
        optimizer.step()
    
def evaluate(data_loader, model, device):
    '''
    This function does the evaluation for one epoch
    :param data_loader: a PyTorch dataloader
    :param model: a PyTorch model
    :param device: str, "cuda" or "cpu"
    '''
    # put the model in evaluation
    model.eval()

    # init lists to store targets and outputs
    final_targets = []
    final_outputs = []

    # we use no_grad context
    with torch.no_grad():
        # go over all the batches
        for data in tqdm(data_loader, total = len(data_loader)):
            # move the data to device
            inputs = data["image"].to(device)
            targets = data["targets"].to(device)

            # do the forward step to generate predictions
            output = model(inputs)

            # convert targets and outputs to lists
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()

            # extend the final lists
            final_targets.extend(targets)
            final_outputs.extend(output)
        # return final output and final targets
        return final_outputs, final_targets