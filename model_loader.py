import utils
import torch.nn as nn

class ModelLoader:
    def __init__(self, model_str: str = "efficientnet", num_classes: int = 10):
        self.model_str = model_str
        self.num_classes = num_classes

    def load_model(self) -> nn.Module:
        """Load and return the specified model."""
        if self.model_str == "alexnet":
            print("Using AlexNet")
            return utils.load_alexnet(classes=self.num_classes)
        else:
            print("Using EfficientNet")
            return utils.load_efficientnet(classes=self.num_classes)
