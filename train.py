"""
train.py
~~~~~~~~~

I created a convolutional neural networks and data
pipeline. So it is time to train them in main.py
"""

# load packages
import torch
import torch.optim as optimize
import torch.nn as neuralnetwork
from config import learning_rate


class Train:

    def __init__(
            self,
            model, # my convolutional neural network is coming from here
            device, # device, cpu or GPU
            train_loader, # train dataset is coming here
            validation_loader, # validation dataset is coming here
    ):
        self.device = device
        self.model = model.to(device)
        self.train_data = train_loader
        self.validation_data = validation_loader
        
        # optimizer part
        self.criterion = neuralnetwork.CrossEntropyLoss() # cost function or loss function
        self.optimizer = optimize.Adam( # I am using ADAM optimizer for optimization
            self.model.parameters(), # parameters of my model
            lr=learning_rate, # learning rate
        )