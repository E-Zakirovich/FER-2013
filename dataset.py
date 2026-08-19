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
from torchvision.models.detection import transform

from config import image_size, amount_of_flip, angle_range, mean, std


class LoadDataset:
    def __init__(self):

        # make a transform for train data
        self.transform_for_train = transform.Compose([
            # resize the image size
            transforms.Resize(
                # images size coming from config.py file
                (
                    image_size,
                    image_size,
                )
            ),

            # flipping the half of the data to avoid underfitting or overfitting
            transforms.RandomHorizontalFlip(
                p = amount_of_flip # amount_of_flip is coming from config.py file
            ),

            # rotate the image between -angle_range : angle_range to avoid underfitting or overfitting
            transforms.RandomRotation(
                p = angle_range # angle_range is coming from config.py file
            ),

            # make a tensor from images to fit the data to convolutional neural networks.
            transforms.ToTensor(),

            # normalization part
            transforms.Normalize(
                mean = mean, # mean is coming from config.py file
                std = std, # std is coming from config.py file
            )

        ])

        # make a transform for validation and test
        self.transform_for_validation_and_test = transform.Compose([
            # resize the image size
            transforms.Resize(
                # images size coming from config.py file
                (
                    image_size,
                    image_size,
                )
            ),

            # make a tensor from images to fit the data to convolutional neural networks.
            transforms.ToTensor(),

            # normalization part
            transforms.Normalize(
                mean=mean,  # mean is coming from config.py file
                std=std,  # std is coming from config.py file
            )
        ])