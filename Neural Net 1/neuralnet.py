# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader



class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.h = 128 # You can adjust this value
        self. LinearLayer1 = nn.Linear(in_size, self.h)
        self. LinearLayer2 = nn.Linear(self.h, self.h)
        self. LinearLayer3 = nn.Linear(self.h, out_size)
        self.optimizer = torch.optim.Adamax(self.parameters(), lr=lrate)

        # raise NotImplementedError("You need to write this part!")
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # raise NotImplementedError("You need to write this part!")
        # Apply ReLU activation for all but last layer
        x = F.relu(self. LinearLayer1(x))
        x = F.relu(self. LinearLayer2(x))
        x = self. LinearLayer3(x) # No activation for last layer
        return x
        # return torch.ones(x.shape[0], 1)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        # raise NotImplementedError("You need to write this part!")
        self.optimizer.zero_grad()
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        # return 0.0



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    trainMean = torch.mean(train_set, dim=0)
    trainStd = torch.std(train_set, dim=0)
    train_set = (train_set - trainMean) / trainStd

    # Standardize the dev data using the same mean and std
    dev_set = (dev_set - trainMean) / trainStd

    net = NeuralNet(0.01, nn.CrossEntropyLoss(), train_set.shape[1], len(torch.unique(train_labels)))
    losses = []
    trainData = get_dataset_from_arrays(train_set, train_labels)
    trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=False)
    print(trainLoader)
    for epoch in range(epochs):
        for batch in trainLoader:
            features = batch['features']
            labels = batch['labels']
            loss = net.step(features, labels)
        losses.append(loss)

    yhats = torch.argmax(net(dev_set), dim=1).cpu().numpy()
    yhats = yhats.astype(int)

    return losses,yhats, net
    # return [],[],None