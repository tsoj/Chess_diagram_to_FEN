import torch
import torch.nn as nn

from src import consts, common

from torchvision import models


def get_tile_model():

    result = models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2)
    result.fc = nn.Sequential(
        torch.nn.LazyLinear(out_features=512),
        nn.ReLU(),
    )

    # print(result)
    # assert False
    return result


def get_full_img_model():

    result = models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2)
    result.fc = nn.Sequential(
        torch.nn.LazyLinear(out_features=512),
        nn.ReLU(),
    )

    # print(result)
    # assert False
    return result


def get_dense_model():
    return nn.Sequential(
        torch.nn.LazyLinear(out_features=768),
        nn.ReLU(),
        torch.nn.LazyLinear(out_features=512),
        nn.ReLU(),
        torch.nn.LazyLinear(out_features=len(common.PIECE_TYPES)),
    )


class ChessRec(nn.Module):
    def __init__(self):
        super(ChessRec, self).__init__()

        self.tile = get_tile_model()

        self.full = get_full_img_model()

        self.dense = get_dense_model()

    def forward(self, img):
        batch_size, ch, h, w = img.shape

        assert h == consts.BOARD_PIXEL_WIDTH
        assert w == consts.BOARD_PIXEL_WIDTH
        assert ch == 3

        x = img
        x = x.unfold(2, consts.SQUARE_SIZE, consts.SQUARE_SIZE)
        x = x.unfold(3, consts.SQUARE_SIZE, consts.SQUARE_SIZE)
        x = x.permute(0, 2, 3, 1, 4, 5)

        assert list(x.shape) == [
            batch_size,
            8,
            8,
            ch,
            consts.SQUARE_SIZE,
            consts.SQUARE_SIZE,
        ]

        x = x.reshape(batch_size * 8 * 8, ch, consts.SQUARE_SIZE, consts.SQUARE_SIZE)

        x = self.tile(x)
        assert len(x.shape) == 2, "Should be [batch_size, flattened]"

        x = x.reshape(batch_size, 64, -1)

        z = self.full(img)
        assert len(z.shape) == 2, "Should be [batch_size, flattened]"
        z = z.reshape(batch_size, 1, -1)
        z = z.expand(-1, 64, -1)

        x = torch.cat((x, z), dim=-1)

        x = x.reshape(batch_size * 64, -1)

        x = self.dense(x)

        x = x.reshape(batch_size, 64, len(common.PIECE_TYPES))

        return x


if __name__ == "__main__":
    model = ChessRec()

    print(torch.cuda.is_available())
