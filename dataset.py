"""
dataset.py
~~~~~~~~~~~

I created this file to make a data pipeline to convolutional
neural network. It will read the images from ./data and will
do some operations to  adjust for convolutional neural nets.

It will do following operations:
 - augmentation [same size for all images, normalization, ...]
 - divide the dataset with manual seed with their indices
 - Use dataloader method to actually split it.

"""

# import packages
import torch
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets, transforms
import config


class LoadDataset:
    def __init__(self):
        ...