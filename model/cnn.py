"""
cnn.py
~~~~~~~

I created this file to make AI. This file is core.
It will use data pipeline and trained on train.py
file.
"""

import torch.nn as neuralnetwork
from config import in_out_channels, kernel_size, stride, padding, kernel_and_stride_size_for_pooling, input_layer, hidden_layer, output_layer, dropout


class CNN(neuralnetwork.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # first convolutional layer
        self.first_convolutional_layer = neuralnetwork.Conv2d(
            in_channels = in_out_channels[0], # in channels (1)
            out_channels = in_out_channels[1], # out channels (32)
            kernel_size = kernel_size, # the size of kernel which is 3
            stride = stride, # the size of stride which is 1, it is amount of movement of kernel
            padding = padding, # extra +1 size for top, down right and left
        )
        # first batch normalization
        self.first_batch_normalization = neuralnetwork.BatchNorm2d(
            in_out_channels[1] # batch normalizer value which is 32 (help me to decreases number to make easy to train)
        )

        # second convolutional layer
        self.second_convolutional_layer = neuralnetwork.Conv2d(
            in_channels = in_out_channels[1],  # in channels (32)
            out_channels = in_out_channels[2], # out channels (64)
            kernel_size = kernel_size, # the size of kernel which is 3
            stride = stride, # the size of stride which is 1, it is amount of movement of kernel
            padding = padding, # extra +1 size for top, down right and left
        )
        # second batch normalization
        self.second_batch_normalization = neuralnetwork.BatchNorm2d(
            in_out_channels[2] # batch normalizer value which is 64 (help me to decreases number to make easy to train)
        )

        # third convolutional layer
        self.third_convolutional_layer = neuralnetwork.Conv2d(
            in_channels = in_out_channels[2],  # in channels (64)
            out_channels = in_out_channels[3],  # out channels (128)
            kernel_size = kernel_size,  # the size of kernel which is 3
            stride = stride,  # the size of stride which is 1, it is amount of movement of kernel
            padding = padding,  # extra +1 size for top, down right and left
        )
        # third batch normalization
        self.third_batch_normalization = neuralnetwork.BatchNorm2d(
            in_out_channels[3]  # batch normalizer value which is 128 (help me to decreases number to make easy to train)
        )

        # ReLU function
        self.relu = neuralnetwork.ReLU()

        # pooling
        self.pool = neuralnetwork.MaxPool2d(
            kernel_size = kernel_and_stride_size_for_pooling, # kernel size must be 2 in order to decrease the size of the image 2x
            stride = kernel_and_stride_size_for_pooling, # stride also must be 2 in order to decrease the size 2x times
        )

        # neural networks part

        # connection between input layer and hidden layer
        self.input_layer_and_hidden_layer_connection = neuralnetwork.Linear(
            input_layer, # input layer, size is 6 * 6 * 128
            hidden_layer, # hidden layer, size is 512 (my decision)
        )

        # dropout some weights in order to avoid underfitting or overfitting
        self.dropout = neuralnetwork.Dropout(
            p = dropout # amount, (I choose 30%, bcz it is not big net and dataset is also small)
        )

        # connection between hidden layer and output layer
        self.hidden_layer_and_output_layer_connection = neuralnetwork.Linear(
            hidden_layer, # hidden layer, size is 512 (my decision)
            output_layer # output layer, total is 7 because 7 moods we got.
        )

    # forward propagation
    def forward(self, x):

        # first cycle for convolutional neural network
        x = self.pool( # pooling (size is decreased by 2x)
            self.relu( # get only positive values
                self.first_batch_normalization( # normalize the values to make easy to train
                    self.first_convolutional_layer( # first convolutional layer
                        x # value
                    )
                )
            )
        )

        # second cycle for convolutional neural network
        x = self.pool(  # pooling (size is decreased by 2x)
            self.relu(  # get only positive values
                self.second_batch_normalization(  # normalize the values to make easy to train
                    self.second_convolutional_layer(  # second convolutional layer
                        x  # value
                    )
                )
            )
        )

        # third cycle for convolutional neural network
        x = self.pool(  # pooling (size is decreased by 2x)
            self.relu(  # get only positive values
                self.third_batch_normalization(  # normalize the values to make easy to train
                    self.third_convolutional_layer(  # third convolutional layer
                        x  # value
                    )
                )
            )
        )

        x = x.view(x.size(0), -1)

        # connection between input layer and hidden layer
        x = self.relu( # get only positive values
            self.input_layer_and_hidden_layer_connection( # connection part
                x # values
            )
        )

        # drop 30% of weights
        x = self.dropout(
            x
        )

        # connection between hidden layer and output layer
        x = self.hidden_layer_and_output_layer_connection(
                x
            )

        return x # return the result