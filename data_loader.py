import utils
import torch
from datasets import Dataset

# This file needs to be implemented by the client or the enduser the source code
class DataClientLoader():
    def load_data(self, client_id: int, toy: bool = False):
        """Load and return the train and test datasets for the client."""
        
        trainset, testset = utils.load_partition(client_id)
        
        if toy:
            trainset = trainset.select(range(10))
            testset = testset.select(range(10))
        
        return trainset, testset
        # pass
        # return load_alexnet(classes=10)