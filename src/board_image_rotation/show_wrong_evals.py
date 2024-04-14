import torch
import matplotlib.pyplot as plt

from src.board_image_rotation import dataset
from src.board_image_rotation.model import ImageRotation


def show_wrong_image_rotations(model_path="models/best_model_image_rotation_0.996_2024-04-14-22-59-55.pth"):

    max_data = None
    data_root_dir = "resources/generated_images/chessboards_fen"

    board_image_set = dataset.BoardImageDataset(
        root_dir=data_root_dir,
        augment_ratio=0.8,
        affine_augment_ratio=0.8,
        max=max_data,
    )

    model = ImageRotation()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    correct = 0
    total = 0
    for img, target in board_image_set:

        total += 1

        with torch.no_grad():
            output = model(img.unsqueeze(0)).squeeze(0)

        pred = output.argmax().item()

        if pred == target:
            correct += 1
            continue

        print("WRONG:")
        print(correct / total)

        img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())

        plt.imshow(img)
        plt.axis("off")
        plt.show()
