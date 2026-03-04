import torch
from torchvision.transforms import v2

from src import consts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bbox(model, img: torch.Tensor):
    """Run inference and return the top detection box, or None.

    Accepts an image tensor in [0, 1] range with shape [C, H, W].
    Returns box in BBOX_IMAGE_SIZE pixel coordinates [x1, y1, x2, y2].
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        assert (
            len(img.shape) == 3
        ), "Need input to be of shape [C, H, W] but is: " + str(img.shape)
        assert img.shape[0] == 3, "Channel dimension must be 3 (RGB)"

        detections = model([img.to(device)])[0]

        if len(detections["boxes"]) == 0:
            return None

        # Take highest-scoring detection
        best = detections["scores"].argmax()
        box = detections["boxes"][best].cpu()

    return box
