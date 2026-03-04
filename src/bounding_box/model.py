import torch.nn as nn

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class BoardBBox(nn.Module):
    def __init__(self):
        super(BoardBBox, self).__init__()

        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights="DEFAULT",
            box_detections_per_img=1,
        )
        # Replace head for single-class detection (background + chessboard)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def forward(self, images, targets=None):
        return self.model(images, targets)


if __name__ == "__main__":
    model = BoardBBox()
    print(model)
