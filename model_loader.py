import utils
import torch.nn as nn


class ModelLoader:
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def load_model(self) -> nn.Module:
        """Load and return the specified model."""
        return utils.load_efficientnet(classes=self.num_classes)
