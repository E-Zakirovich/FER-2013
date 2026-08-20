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
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
        )
        # first batch normalization
        self.first_batch_normalization = nn.BatchNorm2d(
            in_out_channels[1] 
        )