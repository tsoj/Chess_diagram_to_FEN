import torch
import torch.nn as nn


class OrientationModel(nn.Module):
    def __init__(self):
        super(OrientationModel, self).__init__()

        self.model = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(out_features=256),
            nn.ReLU(),
            torch.nn.LazyLinear(out_features=128),
            nn.ReLU(),
            torch.nn.LazyLinear(out_features=16),
            nn.ReLU(),
            torch.nn.LazyLinear(out_features=1),
        )

    def forward(self, x):
        return self.model(x)
