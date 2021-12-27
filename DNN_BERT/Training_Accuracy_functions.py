import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from Utils import plot_grad_flow


def accuracy(net, dataloader, criterion): # pour calculer la précision
    net.eval() # Switch to evaluation mode
    
    MAE = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:

            current_batch_size = y.shape[0]

            preds = net(X)
            preds = preds.view(-1)# Readjusting size from (batch_size,1) to (batchsize)

            #print(outputs, labels)
            total += current_batch_size

            loss = criterion(preds, y)

            MAE += loss.detach()*current_batch_size
        

    net.train() # Switch to training mode
    return MAE / total

def train_epoch(net, optimizer, trainloader, criterion, plotGrad): # pour entraîner sur un epoch
    
    net.train()
    
    # Init:
    losses = []
    total = 0
    
    for X, y in trainloader:
        ## Getting batch size
        current_batch_size = y.shape[0]
        total += current_batch_size

        
        optimizer.zero_grad()#required since pytorch accumulates the gradients
        
        # Forward
        y_pred = net(X)
        y_pred = y_pred.view(-1)
        
        # Compute diff
        loss = criterion(y_pred, y)

        # Compute gradients
        loss.backward()

        # Plots the gradient behavior
        if(plotGrad):
            plot_grad_flow(net.named_parameters())          

        # update weights
        optimizer.step()
        losses.append(loss.data*current_batch_size)
        
    losses = np.array(losses)
    return losses.sum()/total