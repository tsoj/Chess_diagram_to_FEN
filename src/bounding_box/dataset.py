import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import v2

from src import consts
from src.bounding_box.generate_chessboards_bbox import BboxGenerator
from src.common import AddGaussianNoise, to_rgb_tensor

default_transforms = torch.nn.Sequential(
    v2.ToDtype(torch.float32),
    v2.Resize(
        size=(consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE),
        interpolation=v2.InterpolationMode.BICUBIC,
    ),
    v2.ClampBoundingBoxes(),
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


def _normalize_to_01(img):
    """Normalize image to [0, 1] range for Faster R-CNN input."""
    mn, mx = img.min(), img.max()
    if mn >= mx:
        return torch.zeros_like(img)
    return (img - mn) / (mx - mn)


def collate_fn(batch):
    """Custom collate for Faster R-CNN: returns list of images and list of targets."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


class GenerativeBboxDataset(IterableDataset):
    def __init__(
        self,
        game: str,
        augment_ratio: float = 0.5,
    ):
        self.generator = BboxGenerator(game)
        self.augment_ratio = augment_ratio

    def __iter__(self):
        while True:
            image, (center_x, center_y, width, height) = self.generator.generate_one()
            input_img = to_rgb_tensor(image)

            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2

            do_augment = self.augment_ratio > random.uniform(0, 1)

            while True:
                if do_augment:
                    input_img = augment_transforms(input_img)
                if input_img.isnan().any():
                    print("WARNING: Found nan after augmentation. Trying again.")
                    continue
                break

            input_img = v2.ToDtype(torch.float32)(input_img)
            input_img = v2.Resize(
                size=(consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE),
                interpolation=v2.InterpolationMode.BICUBIC,
            )(input_img)
            if input_img.isnan().any():
                print("WARNING: Found nan after resize. Trying again.")
                continue

            input_img = _normalize_to_01(input_img)

            box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            target = {
                "boxes": box,
                "labels": torch.tensor([1], dtype=torch.int64),
            }

            yield input_img, target


def generate_fixed_test_set(
    game: str,
    size: int = 500,
    seed: int = 42,
) -> list[tuple[torch.Tensor, dict]]:
    rng_state = random.getstate()
    random.seed(seed)

    generator = BboxGenerator(game)
    data = []
    for _ in range(size):
        image, (center_x, center_y, width, height) = generator.generate_one()
        input_img = to_rgb_tensor(image)
        input_img = v2.ToDtype(torch.float32)(input_img)
        input_img = v2.Resize(
            size=(consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE),
            interpolation=v2.InterpolationMode.BICUBIC,
        )(input_img)
        input_img = _normalize_to_01(input_img)

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        target = {
            "boxes": box,
            "labels": torch.tensor([1], dtype=torch.int64),
        }

        data.append((input_img, target))

    random.setstate(rng_state)
    return data


def test_data_set(game: str, size: int = 1000):
    ds = GenerativeBboxDataset(game=game, augment_ratio=0.5)

    for i, (img, target) in enumerate(ds):
        if i >= size:
            break
        assert not img.isnan().any()

        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(img.permute(1, 2, 0))
        x1, y1, x2, y2 = target["boxes"].squeeze(0).tolist()
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=(1.0, 0.0, 0.0, 0.8),
            facecolor=(1.0, 0.0, 0.0, 0.1),
            linestyle="--",
        )
        ax1.add_patch(rect)
        plt.show()
