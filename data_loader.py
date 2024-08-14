import utils
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

# This file needs to be implemented by the client or the enduser the source code


class DataClientLoader():
    def __init__(self, client_id: int, toy: bool = False):
        self.client_id = client_id
        self.toy = toy
        self.trainset, self.testset = self.load_data()

    def load_data(self):
        """Load and return the train and test datasets for the client."""

        trainset, testset = utils.load_partition(self.client_id)

        if self.toy:
            trainset = trainset.select(range(10))
            testset = testset.select(range(10))

        return trainset, testset
        # pass
        # return load_alexnet(classes=10)

    def get_data_loaders(self, trainset, validation_split, batch_size):
        """Create and return the train and validation data loaders."""
        train_valid = trainset.train_test_split(validation_split, seed=42)
        trainset = train_valid["train"]
        valset = train_valid["test"]

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=batch_size)

        return train_loader, val_loader

    def get_test_loader(self, testset):
        """Create and return the test data loader."""
        return DataLoader(testset, batch_size=16)
