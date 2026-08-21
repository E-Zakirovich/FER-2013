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
from config import learning_rate, total_channels


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

    # following method will help me to train
    def fit(self, epochs):

        for epoch in range(epochs): # I need a loop to train epochs times

            self.model.train() # train the model

            # I need to measure train loss, running loss and correct values for evaluation part

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.train_data: # I need a loop to try each train images

                images = images.to(self.device) # I am loading images to hardware
                labels = labels.to(self.device) # I am loading labels to hardware

                self.optimizer.zero_grad() # it is better to make zero grad before get the output from convolutional neural network

                output = self.model(images) # I get the output from convolutional neural network

                loss = self.criterion( # find loss function
                    output, # output from convolutional neural network
                    labels # labels to check the correctness of output
                )

                loss.backward()  # backpropagation algorithm is working here 

                self.optimizer.step() # after calculation of loss function better optimize weights. Actually, I used ADAM to optimize

                # running loss, total corrects, total, corrects
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            # evaluation part