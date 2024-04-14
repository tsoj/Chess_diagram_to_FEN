import torch
from torchvision.transforms import functional
import matplotlib.pyplot as plt
import chess
import argparse
import random
import os
from PIL import Image, ImageOps
from pathlib import Path
from src.bounding_box.model import ChessBoardBBox
from src.fen_recognition.model import ChessRec
from src.board_orientation.model import OrientationModel
from src.board_image_rotation.model import ImageRotation
import src.fen_recognition.dataset as fen_dataset
import src.board_image_rotation.dataset as rotation_dataset

from src.bounding_box.inference import get_bbox
from src import consts, common

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SomeModel:

    def __init__(self, model_class: type, default_path=None) -> None:
        self.model = None
        self.model_path = default_path
        self.model_class = model_class

    def get(self):
        if self.model is None:
            if self.model_path is None:
                raise Exception(
                    "Model path not set. Use set_model_path to set the model path."
                )

            self.model = self.model_class()
            self.model.load_state_dict(
                torch.load(
                    self.model_path,
                    map_location=torch.device("cpu"),
                )
            )
            self.model.to(device)
            self.model.eval()
        return self.model

    def set_model_path(self, model_path: str):
        self.model = None
        self.model_path = model_path


script_dir = os.path.abspath(os.path.dirname(__file__))

bbox_model = SomeModel(
    ChessBoardBBox,
    script_dir + "/models/best_model_bbox_0.958_2024-01-28-22-49-40.pth",
)
image_rotation_model = SomeModel(
    ImageRotation,
    script_dir + "/models/best_model_image_rotation_0.996_2024-04-14-22-59-55.pth",
)
fen_model = SomeModel(
    ChessRec,
    script_dir + "/models/best_model_fen_0.953_2024-02-03-13-49-31.pth",
)
orientation_model = SomeModel(
    OrientationModel,
    script_dir + "/models/best_model_orientation_0.987_2024-02-04-17-34-05.pth",
)


@torch.no_grad()
def crop_to_chessboard(img: Image.Image, max_num_tries=10) -> Image.Image:

    pad_factor = 0.05
    pad_x = img.width * pad_factor
    pad_y = img.height * pad_factor

    img = common.pad(img, pad_x, pad_y)

    for _ in range(0, max_num_tries):
        if img.width == 0 or img.height == 0:
            return None

        img_tensor = common.to_rgb_tensor(img)
        img_tensor = functional.resize(
            img_tensor, [consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE]
        )
        img_tensor = common.MinMaxMeanNormalization()(img_tensor)

        bbox = get_bbox(bbox_model.get(), img_tensor)
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        x_factor = img.width / consts.BBOX_IMAGE_SIZE
        y_factor = img.height / consts.BBOX_IMAGE_SIZE
        x1 *= x_factor
        x2 *= x_factor
        y1 *= y_factor
        y2 *= y_factor

        x1 = int(x1.clamp(0, img.width - 1))
        x2 = int(x2.clamp(0, img.width - 1))
        y1 = int(y1.clamp(0, img.height - 1))
        y2 = int(y2.clamp(0, img.height - 1))

        new_width = x2 - x1
        new_height = y2 - y1

        # We only accept the bounding box if it is relatively big compared to the entire image.
        # Otherwise we try again by cropping the image a little closer to the estimated true bbox
        if new_width / img.width > 0.7 and new_height / img.height > 0.7:
            return img.crop((x1, y1, x2, y2))

        x_addition = new_width * 0.1
        y_addition = new_height * 0.1
        x1 = max(x1 - x_addition, 0)
        x2 = min(x2 + x_addition, img.width)
        y1 = max(y1 - y_addition, 0)
        y2 = min(y2 + y_addition, img.height)

        img = img.crop((x1, y1, x2, y2))

    return None


@torch.no_grad()
def rotate_board_image_if_necessary(
    img: Image.Image, mirror_when_180_rotation
) -> Image.Image:
    input_img = common.to_rgb_tensor(img)
    input_img = rotation_dataset.default_transforms(input_img).to(device)
    pred = (
        image_rotation_model.get()(input_img.unsqueeze(0))
        .cpu()
        .squeeze(0)
        .argmax()
        .item()
    )
    img = img.rotate(-rotation_dataset.ROTATIONS[pred])

    if rotation_dataset.ROTATIONS[pred] == 180 and mirror_when_180_rotation:
        img = ImageOps.mirror(img)

    return img


@torch.no_grad()
def rotate_position_if_necessary(board: chess.Board, no_rotate_bias=0.2) -> chess.Board:
    board_tensor = common.chess_board_to_tensor(board)
    output = (
        orientation_model.get()(board_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
    )

    if output.item() - no_rotate_bias > 0.5:
        print("Rotating board:", output.item())
        board_tensor = common.rotate_board_tensor(board_tensor)

    return common.tensor_to_chess_board(board_tensor)


@torch.no_grad()
def get_board_from_cropped_img(img: Image.Image, num_tries=20) -> chess.Board:
    MIN_SIZE = 32
    if img.width < MIN_SIZE or img.height < MIN_SIZE:
        return None

    img = common.to_rgb_tensor(img).to(device)
    sum = None
    with torch.no_grad():
        tries = 0
        while tries < num_tries:
            input = img

            if tries >= 2:
                input = fen_dataset.augment_transforms(input)

            color_flipped = tries % 2 == 1
            if color_flipped:
                input = -input

            input = fen_dataset.default_transforms(input)

            if input.isnan().any():
                print("WARNING: Found nan after transforms.")
                continue

            output = fen_model.get()(input.unsqueeze(0)).squeeze(0)
            output = output.clamp(0, 1)
            # print(output)

            if color_flipped:
                output = common.flip_color(output)

            if sum is None:
                sum = output
            else:
                sum += output
            tries += 1

    board = common.tensor_to_chess_board(sum.cpu())
    if board.occupied == 0:
        return None
    return board


def get_fen(
    img: Image.Image,
    num_tries=10,
    return_cropped_img=False,
    auto_rotate_image=True,
    mirror_when_180_rotation=False,
    auto_rotate_board=True,
):
    """Takes an image and returns an FEN (Forsyth-Edwards Notation) string.

    Args:
        - `img (PIL.Image.Image)`: The image of a chess diagram.
        - `num_tries (int)`: The more higher this number is, the more accurate the returned FEN will be, with diminishing returns.
        - `return_cropped_img (bool)`: If this is set to `True`, this function will return a tuple [`str`, `Image`] where the image is
        the input image cropped to the chess diagram.
        - `auto_rotate_image (bool)`: If this is set to `True`, this function will try to guess if the image is rotated 0°, 90°, 180°,
        or 270° and rotate the image accordingly.
        - `mirror_when_180_rotation (bool)`: If this  and `auto_rotate_image` is set to `True`, this function will also mirror the image
        (left to right) if it was rotated 180°.
        - `auto_rotate_board (bool)`: If this is set to `True`, this function will try to guess if the diagram is from whites or blacks
        perspective and rotate the board accordingly.

    Returns:
        - `[str | None, PIL.Image.Image] | str | None`: Returns the FEN, or if `return_cropped_img` is set to `True`, returns the FEN
        and the image cropped to the chess diagram. If no chess diagram can be detected, the returned FEN will be `None`.
    """

    fen = None

    img = img.convert("RGB")
    img = crop_to_chessboard(img, max_num_tries=num_tries)
    if auto_rotate_image:
        img = rotate_board_image_if_necessary(img, mirror_when_180_rotation)

    if img is not None:
        board = get_board_from_cropped_img(img, num_tries=num_tries)

        if board is not None:
            if auto_rotate_board:
                board = rotate_position_if_necessary(board)

            fen = board.fen()

    if return_cropped_img:
        return fen, img
    return fen


def demo(root_dir: str, shuffle_files: bool):

    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        print("Using CPU")

    torch.set_printoptions(precision=1, sci_mode=False)

    path = Path(root_dir)
    if path.is_dir():
        file_names = list(
            (
                p.resolve()
                for p in Path(path).glob("**/*")
                if p.suffix in {".png", ".jpeg", ".jpg"}
            )
        )
    else:
        file_names = [path]

    if shuffle_files:
        random.shuffle(file_names)

    for file_name in file_names:
        print(file_name)

        img = Image.open(file_name).convert("RGB")

        img = img.rotate(random.choice(rotation_dataset.ROTATIONS))

        fen, cropped = get_fen(img, return_cropped_img=True)

        if fen is None:
            print("Couldn't detect chessboard:", file_name)
            continue

        print(fen)

        true_fen = common.normalize_fen(Path(file_name).stem)
        if true_fen is None:
            print(f"WARNING: Couldn't find ground truth FEN")
        else:
            if fen == true_fen:
                print("Correct")
            else:
                print(true_fen)
                print("WRONG")

        fen_img = common.get_image(chess.Board(fen), width=512, height=512)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

        ax1.imshow(img)
        ax2.imshow(cropped)
        ax3.imshow(fen_img)
        ax1.axis("off")
        ax2.axis("off")
        ax3.axis("off")
        ax1.title.set_text("Original image")
        ax2.title.set_text("Cropped to board")
        ax3.title.set_text("Recognized board")
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="directory that contains images of chess diagrams",
    )
    parser.add_argument(
        "--bbox_model",
        type=str,
        default=bbox_model.model_path,
        help="path to bbox model parameters",
    )
    parser.add_argument(
        "--fen_model",
        type=str,
        default=fen_model.model_path,
        help="path to fen model parameters",
    )
    parser.add_argument(
        "--orientation_model",
        type=str,
        default=orientation_model.model_path,
        help="path to orientation_model model parameters",
    )
    parser.add_argument("--shuffle_files", action="store_true")
    args = parser.parse_args()

    bbox_model.set_model_path(args.bbox_model)
    fen_model.set_model_path(args.fen_model)
    orientation_model.set_model_path(args.orientation_model)

    demo(root_dir=args.dir, shuffle_files=args.shuffle_files)
