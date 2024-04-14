import torch
import torch.nn as nn

from src import consts, common
import src.board_image_rotation.dataset as dataset

from torchvision import models


class ImageRotation(nn.Module):
    def __init__(self):
        super(ImageRotation, self).__init__()

        self.model = models.regnet_x_800mf(
            weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2
        )
        self.model.fc = torch.nn.LazyLinear(out_features=len(dataset.ROTATIONS))

        # print(self.model)
        # assert False

    def forward(self, x):
        return self.model(x)
