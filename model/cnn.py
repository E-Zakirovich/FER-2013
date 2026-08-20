"""
cnn.py
~~~~~~~

I created this file to make AI. This file is core.
It will use data pipeline and trained on train.py
file.
"""

import torcn.nn as neuralnetwork
from config import in_out_channels, kernel_size, stride, padding


class CNN(neuralnetwork.NeuralNetwork):
    def __init__(self):
        super(CNN, self).__init__()

        # first convolutional layer
        self.first_convolutional_layer = nn.Conv2d(
            in_channels = in_out_channels[0], # in channels (3)
            out_channels = in_out_channels[1], # out channels (32)
            kernel_size = kernel_size, # the size of kernel which is 3
            stride = stride, # the size of stride which is 1, it is amount of movement of kernel
            padding = padding, # extra +1 size for top, down right and left
        )
        # first batch normalization
        self.first_batch_normalization = nn.BatchNorm2d(
            in_out_channels[1] # batch normalizer value which is 32 (help me to decreases number to make easy to train)
        )

        # second convolutional layer
        self.second_convolutional_layer = nn.Conv2d(
            in_channels = in_out_channels[1],  # in channels (32)
            out_channels = in_out_channels[2], # out channels (64)
            kernel_size = kernel_size, # the size of kernel which is 3
            stride = stride, # the size of stride which is 1, it is amount of movement of kernel
            padding = padding, # extra +1 size for top, down right and left
        )
        # second batch normalization
        self.second_batch_normalization = nn.BatchNorm2d(
            in_out_channels[2] # batch normalizer value which is 64 (help me to decreases number to make easy to train)
        )