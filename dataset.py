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

from config import image_size, amount_of_flip, angle_range, mean, std, dataset_path, batch_size, num_workers


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

    # following method will help me to load the images, better use static method
    @staticmethod
    def __load_images(self, _path, _transform):
        # load the images
        images = datasets.Images(
            root = _path, # path is coming from user
            transform = _transform, # transform is coming from user
        )
        return images # return the result

    # following method will help me to load the dataset, it will use other methods to return the result.
    def __private_dataset_loader(self):
        # image loader for train, validation and testing
        train_images = self.__load_images(
            _path = dataset_path, # path is coming from config
            _transform = self.transform_for_train # transform source
        )

        # image loader for validation
        validation_images = self.__load_images(
            _path = dataset_path,  # path is coming from config
            _transform = self.transform_for_validation_and_test # transform source
        )

        # image loader for testing
        test_images = self.__load_images(
            _path = dataset_path,  # path is coming from config
            _transform = self.transform_for_validation_and_test # transform source
        )

        # manual seed

        # split the dataset according to their indices

        # make a subset for train, validation and testing

        # load the dataset with Data Loader

        return True # temporary

    # it will use private dataset loader to return the dataset, (encapsulation for safety).
    def load_dataset(self):
        result = self.__private_dataset_loader()
        return result