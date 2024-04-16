import torch
import torch.nn as nn

from src import consts

from torchvision import models


class ChessExistence(nn.Module):
    def __init__(self):
        super(ChessExistence, self).__init__()

        self.model = models.regnet_x_800mf(
            weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2
        )
        self.model.fc = torch.nn.LazyLinear(out_features=1)

        # print(self.model)
        # assert False

    def forward(self, x):
        return self.model(x)
