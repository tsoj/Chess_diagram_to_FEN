import torch
import matplotlib.pyplot as plt

from src.existence import dataset
from src.existence.model import ChessExistence


def show_wrong_existence(model_path="models/best_model_existence_0.998_2024-04-16-23-44-48.pth"):

    max_data = None

    board_image_set = dataset.ExistenceDataset(
        with_board_root_dir="resources/generated_images/chessboards_bbox",
        no_board_root_dir="resources/generated_images/no_chessboards",
        augment_ratio=0.8,
        affine_augment_ratio=0.8,
        max=max_data,
    )

    model = ChessExistence()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    correct = 0
    total = 0
    for img, target in board_image_set:

        total += 1

        with torch.no_grad():
            output = model(img.unsqueeze(0)).squeeze(0)

        pred = output.round()

        if pred == target:
            correct += 1
            continue

        print("WRONG:")
        print(f"(should be: {target.item()})")
        print(correct / total)

        img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())

        plt.imshow(img)
        plt.axis("off")
        plt.show()
