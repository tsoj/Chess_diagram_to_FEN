import chess
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
from src import common, consts


default_transforms = torch.nn.Sequential(
    v2.ToDtype(torch.float32),
    v2.Resize(
        size=(consts.BOARD_PIXEL_WIDTH, consts.BOARD_PIXEL_WIDTH),
        interpolation=v2.InterpolationMode.BICUBIC,
    ),
    common.MinMaxMeanNormalization(),
)

augment_transforms = torch.nn.Sequential(
    v2.RandomApply(
        [common.AddGaussianNoise(std=0.1, scale_to_input_range=True)], p=0.4
    ),
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

ROTATIONS = [0, 90, 180, 270]


class BoardImageDataset(Dataset):

    def __init__(
        self,
        root_dir,
        augment_ratio=0.5,
        affine_augment_ratio=0.8,
        max=None,
        device=torch.device("cpu"),
    ):

        self.device = device
        self.augments = torch.nn.Sequential(
            v2.RandomApply([affine_transforms], p=affine_augment_ratio),
            v2.RandomApply([augment_transforms], p=augment_ratio),
        )

        root_dir = Path(root_dir)
        assert root_dir.is_dir(), f"With root_dir = {root_dir}"
        self.image_files = list(root_dir.glob("**/*.png"))
        random.shuffle(self.image_files)
        if max is not None:
            self.image_files = self.image_files[0 : min(len(self.image_files), max)]

        print(f"Found {len(self.image_files)} files")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = self.image_files[idx]

        try:
            img = Image.open(file_path)
        except RuntimeError:
            print("Error:", file_path)
            raise

        target = random.randint(0, len(ROTATIONS) - 1)

        img = img.rotate(ROTATIONS[target], expand=True)

        input_img = common.to_rgb_tensor(img).to(self.device)

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

        return (input_img, target)


def test_data_set():

    root_dir = "resources/generated_images/chessboards_fen"

    dataset = BoardImageDataset(root_dir, max=1000)

    for i in range(0, len(dataset)):
        img, target = dataset[i]

        assert not img.isnan().any()
        print(target, ROTATIONS[target])

        # fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))

        img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())

        plt.imshow(img)
        plt.axis("off")
        plt.show()
