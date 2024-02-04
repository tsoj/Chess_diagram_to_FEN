import torch
import time
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
from PIL import Image
from pathlib import Path

from src.common import to_rgb_tensor, MinMaxMeanNormalization, AddGaussianNoise
from src import consts


def box_to_mask(box: torch.Tensor):
    # box should be in x1, y1, x2, y2 format and the values should be relative
    mask = torch.zeros([consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE])
    x1, y1, x2, y2 = box.to(int)
    mask[y1:y2, x1:x2] = 1.0
    return mask.unsqueeze(0)


default_transforms = torch.nn.Sequential(
    v2.ToDtype(torch.float32),
    v2.Resize(
        size=(consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE),
        interpolation=v2.InterpolationMode.BICUBIC,
    ),
    MinMaxMeanNormalization(),
)

augment_transforms = torch.nn.Sequential(
    v2.RandomApply([v2.RandomAffine(degrees=1.0, shear=1.0)], p=0.3),
    v2.RandomInvert(p=0.1),
    v2.RandomApply([AddGaussianNoise(std=0.1, scale_to_input_range=True)], p=0.4),
    v2.RandomApply(
        [v2.ElasticTransform(alpha=30.0), v2.ElasticTransform(alpha=40.0)], p=0.4
    ),
    v2.RandomGrayscale(p=0.4),
    v2.RandomPosterize(bits=2, p=0.2),
    v2.RandomApply(
        [v2.ColorJitter(brightness=0.9, contrast=(0.1, 1.5), hue=0.3)], p=0.3
    ),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3))], p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5))], p=0.1),
    v2.RandomAdjustSharpness(sharpness_factor=10, p=0.1),
    v2.RandomEqualize(p=0.8),
)


class ChessBoardBBoxDataset(Dataset):
    """Chess board bounding box dataset."""

    def __init__(
        self, root_dir, augment_ratio=0.5, max=None, device=torch.device("cpu")
    ):

        self.device = device
        self.augment_ratio = augment_ratio

        root_dir = Path(root_dir)
        assert root_dir.is_dir(), f"With root_dir = {root_dir}"
        self.image_files = list(root_dir.glob("**/*.jpg"))
        random.shuffle(self.image_files)
        if max is not None:
            self.image_files = self.image_files[0 : min(len(self.image_files), max)]

        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = self.image_files[idx]

        try:
            img = Image.open(file_path)
        except RuntimeError:
            print("Error:", file_path)
            raise
        input_img = to_rgb_tensor(img).to(self.device)

        assert input_img.shape[1] == consts.BBOX_IMAGE_SIZE
        assert input_img.shape[2] == consts.BBOX_IMAGE_SIZE

        center_x, center_y, width, height = [int(x) for x in file_path.stem.split("_")]

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2

        box = torch.tensor([x1, y1, x2, y2])

        do_augment = self.augment_ratio > random.uniform(0, 1)

        while True:

            if do_augment:
                input_img = augment_transforms(input_img)

            if input_img.isnan().any():
                print("WARNING: Found nan after augmentation. Trying again.")
                continue

            input_img = default_transforms(input_img)

            if input_img.isnan().any():
                print(
                    f"WARNING: Found nan after default transform (do_augment = {do_augment}). Trying again."
                )
                continue
            break

        return (input_img, box, box_to_mask(box))


def test_data_set():

    c = ChessBoardBBoxDataset(
        root_dir="resources/generated_images/chessboards_bbox",
        augment_ratio=0.5,
        max=1000,
    )

    for i in range(0, len(c)):
        img, target_box, target_mask = c[i]

        assert not img.isnan().any()

        img *= 255.0
        img += 128.0
        img = img.to(torch.uint8)
        img = draw_bounding_boxes(img, target_box.unsqueeze(0), width=5, colors="red")

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow((img.permute(1, 2, 0) - img.min()) / (img.max() - img.min()))
        ax2.imshow(target_mask.squeeze(0))
        plt.show()
