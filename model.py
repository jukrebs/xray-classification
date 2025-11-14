import torch.nn as nn
from torchvision import models


def build_model():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.model = models.resnet18(weights=None)
            self.model.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 1)

        def forward(self, x):
            return self.model(x)

    return Net()
