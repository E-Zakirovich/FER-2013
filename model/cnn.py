"""
cnn.py
~~~~~~~

I created this file to make AI. This file is core.
It will use data pipeline and trained on train.py
file.
"""

import torcn.nn as neuralnetwork
from config import *


class CNN(neuralnetwork.NeuralNetwork):
    def __init__(self):
        super(CNN, self).__init__()