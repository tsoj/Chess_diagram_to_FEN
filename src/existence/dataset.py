import torch
import time
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path

from src.common import to_rgb_tensor, MinMaxMeanNormalization, AddGaussianNoise
from src import consts


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

affine_transforms = v2.RandomAffine(
    degrees=1.5, translate=(0.01, 0.01), scale=(0.99, 1.01), shear=1.5
)

class ExistenceDataset(Dataset):

    def __init__(
        self,
        with_board_root_dir,
        no_board_root_dir,
        augment_ratio=0.5,
        affine_augment_ratio=0.8,
        max=None,
        device=torch.device("cpu"),
    ):

        self.device = device
        self.augment_ratio = augment_ratio

        self.augments = torch.nn.Sequential(
            v2.RandomApply([affine_transforms], p=affine_augment_ratio),
            v2.RandomApply([augment_transforms], p=augment_ratio),
        )

        self.image_files = []

        for dir, target in [(with_board_root_dir, 1.0), (no_board_root_dir, 0.0)]:

            dir = Path(dir)
            assert dir.is_dir(), f"With root_dir = {dir}"
            files = list(dir.glob("**/*.jpg"))
            random.shuffle(files)

            assert len(files) > 0
            if len(self.image_files) != 0:
                size_per_dir = min(len(self.image_files), len(files))
                self.image_files = self.image_files[0:size_per_dir]
                files = files[0:size_per_dir]

            self.image_files.extend([(file, target) for file in files])

        random.shuffle(self.image_files)

        if max is not None:
            self.image_files = self.image_files[0 : min(len(self.image_files), max)]

        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path, target = self.image_files[idx]

        try:
            img = Image.open(file_path)
        except RuntimeError:
            print("Error:", file_path)
            raise
        input_img = to_rgb_tensor(img).to(self.device)

        assert input_img.shape[1] == consts.BBOX_IMAGE_SIZE
        assert input_img.shape[2] == consts.BBOX_IMAGE_SIZE

        while True:

            input_img = self.augments(input_img)

            if input_img.isnan().any():
                print("WARNING: Found nan after augmentation. Trying again.")
                continue

            input_img = default_transforms(input_img)

            if input_img.isnan().any():
                print(f"WARNING: Found nan after default transform. Trying again.")
                continue
            break

        return (input_img, torch.tensor(target).unsqueeze(0))


def test_data_set():

    c = ExistenceDataset(
        with_board_root_dir="resources/generated_images/chessboards_bbox",
        no_board_root_dir="resources/generated_images/no_chessboards",
        augment_ratio=0.5,
        max=1000,
    )

    for i in range(0, len(c)):
        img, target = c[i]

        assert not img.isnan().any()

        print(target)

        img *= 255.0
        img += 128.0
        img = img.to(torch.uint8)

        plt.imshow((img.permute(1, 2, 0) - img.min()) / (img.max() - img.min()))
        plt.show()
