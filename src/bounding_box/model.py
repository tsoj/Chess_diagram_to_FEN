import torch
import torch.nn as nn

from src import consts

from torchvision import models


class ChessBoardBBox(nn.Module):
    def __init__(self):
        super(ChessBoardBBox, self).__init__()

        self.model = models.segmentation.lraspp_mobilenet_v3_large()
        self.model.classifier = models.segmentation.lraspp.LRASPPHead(40, 960, 1, 128)

        # print(self.model)

        # img = torch.rand([2, 3, consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE])
        # output = self.model(img)
        # print(output["out"].shape)

        # assert False

    def forward(self, img):
        batch_size, ch, h, w = img.shape

        assert h == consts.BBOX_IMAGE_SIZE
        assert w == consts.BBOX_IMAGE_SIZE
        assert ch == 3
        assert batch_size >= 2 or not self.training

        x = self.model(img)["out"]

        assert list(x.shape) == [
            batch_size,
            1,
            consts.BBOX_IMAGE_SIZE,
            consts.BBOX_IMAGE_SIZE,
        ]

        return x


if __name__ == "__main__":
    model = ChessBoardBBox()

    print(torch.cuda.is_available())
