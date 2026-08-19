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

from config import image_size, amount_of_flip, angle_range, mean, std, dataset_path, batch_size, num_workers, train_split, validation_split, test_split


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

    # following method will help you to make a subset
    @staticmethod
    def _make_subset(self, indices : list, data):
        # subset parameters are coming from user
        subset = Subset(
            dataset = data, # path is coming from user
            indices = indices.indices # indices are coming from user
        )
        return subset # return the result


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
        manual_seed = torch.Generator().manual_seed(
            seed = seed, # seed is coming from config.py
        )

        # split the dataset according to their indices
        train_indices, validation_indices, test_indices = random_split(
            train_images,
            [
                train_split, # train split is coming form config.py file
                validation_split, # validation split is coming form config.py file
                test_split # test split is coming from config.py file
            ],
            generator = manual_seed # try to control random split (split randomly once and use it forever)
        )

        # make a subset for train
        train_subset = self._make_subset(
            train_images,
            train_indices
        )

        # make a subset for validation
        validation_subset = self._make_subset(
            validation_images,
            validation_indices

        )

        # make a subset for test
        test_subset = self._make_subset(
            test_images,
            test_indices
        )

        # load the dataset with Data Loader

        return True # temporary

    # it will use private dataset loader to return the dataset, (encapsulation for safety).
    def load_dataset(self):
        result = self.__private_dataset_loader()
        return result