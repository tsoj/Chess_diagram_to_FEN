import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional

import src.board_image_rotation.dataset as rotation_dataset
import src.fen_recognition.dataset as fen_dataset
from src import common, consts
from src.board_image_rotation.model import ImageRotation
from src.board_orientation.model import OrientationModel
from src.bounding_box.inference import get_bbox
from src.bounding_box.model import BoardBBox
from src.existence.model import BoardExistence
from src.fen_recognition.model import BoardRec
from src.games import GAMES, get_game

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SomeModel:
    def __init__(self, model_class: type, default_path=None, name: str = "") -> None:
        self.model = None
        self.model_path = default_path
        self.model_class = model_class
        self.name = name

    def get(self):
        if self.model is None:
            if self.model_path is None:
                desc = f" '{self.name}'" if self.name else ""
                raise FileNotFoundError(
                    f"No model file found{desc}. "
                    f"Make sure a trained .pth file exists in the expected directory, "
                    f"or pass an explicit path via set_model_path()."
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


def _find_latest_model(game: str, *patterns: str) -> str | None:
    model_dir = Path(script_dir) / "models" / game
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(model_dir.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


@dataclass
class _GameModels:
    existence: SomeModel
    bbox: SomeModel
    image_rotation: SomeModel
    fen: SomeModel
    orientation: SomeModel


_models_cache: dict[str, _GameModels] = {}


def _get_models(game: str) -> _GameModels:
    if game not in _models_cache:
        model_dir = f"models/{game}/"
        _models_cache[game] = _GameModels(
            existence=SomeModel(
                BoardExistence,
                _find_latest_model(game, "best_model_existence_*.pth"),
                name=f"{model_dir}best_model_existence_*.pth",
            ),
            bbox=SomeModel(
                BoardBBox,
                _find_latest_model(game, "best_model_bbox_*.pth"),
                name=f"{model_dir}best_model_bbox_*.pth",
            ),
            image_rotation=SomeModel(
                ImageRotation,
                _find_latest_model(game, "best_model_image_rotation_*.pth"),
                name=f"{model_dir}best_model_image_rotation_*.pth",
            ),
            fen=SomeModel(
                lambda g=game: BoardRec(game=g),
                _find_latest_model(
                    game, "best_model_position_*.pth", "best_model_fen_*.pth"
                ),
                name=f"{model_dir}best_model_position_*.pth",
            ),
            orientation=SomeModel(
                lambda g=game: OrientationModel(game=g),
                _find_latest_model(game, "best_model_orientation_*.pth"),
                name=f"{model_dir}best_model_orientation_*.pth",
            ),
        )
    return _models_cache[game]


@torch.no_grad()
def check_for_board_existence(img: Image.Image, game: str) -> bool:

    img_tensor = common.to_rgb_tensor(img)
    img_tensor = functional.resize(
        img_tensor, [consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE]
    )
    img_tensor = img_tensor.to(device)
    img_tensor = common.MinMaxMeanNormalization()(img_tensor)

    output = _get_models(game).existence.get()(img_tensor.unsqueeze(0)).squeeze(0)

    return output.cpu().item() > 0.5


@torch.no_grad()
def crop_to_board(img: Image.Image, game: str, max_num_tries=10) -> Image.Image:

    pad_factor = 0.05
    pad_x = img.width * pad_factor
    pad_y = img.height * pad_factor

    img = common.pad(img, pad_x, pad_y)

    bbox_model = _get_models(game).bbox

    for _ in range(0, max_num_tries):
        if img.width == 0 or img.height == 0:
            return None

        img_tensor = common.to_rgb_tensor(img).float()
        img_tensor = functional.resize(
            img_tensor, [consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE]
        )
        # Normalize to [0, 1] for Faster R-CNN
        mn, mx = img_tensor.min(), img_tensor.max()
        if mx > mn:
            img_tensor = (img_tensor - mn) / (mx - mn)
        else:
            img_tensor = torch.zeros_like(img_tensor)

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
def board_image_rotation(img: Image.Image, game: str) -> int:
    input_img = common.to_rgb_tensor(img)
    input_img = rotation_dataset.default_transforms(input_img).to(device)
    pred = (
        _get_models(game)
        .image_rotation.get()(input_img.unsqueeze(0))
        .cpu()
        .squeeze(0)
        .argmax()
        .item()
    )
    return pred


@torch.no_grad()
def is_board_flipped(board: common.Position, game: str, no_rotate_bias=0.2) -> bool:
    board_tensor = common.position_to_tensor(board)
    output = (
        _get_models(game)
        .orientation.get()(board_tensor.unsqueeze(0).to(device))
        .squeeze(0)
        .cpu()
    )

    return output.item() - no_rotate_bias > 0.5


@torch.no_grad()
def rotate_board(board: common.Position, game: str) -> common.Position:
    spec = get_game(game)
    board_tensor = common.position_to_tensor(board)
    board_tensor = common.rotate_tensor_180(board_tensor, spec)
    return common.tensor_to_position(board_tensor, spec)


@torch.no_grad()
def get_board_from_cropped_img(img: Image.Image, game: str) -> common.Position:
    spec = get_game(game)
    models = _get_models(game)
    position_transforms = fen_dataset.get_default_transforms(game)

    MIN_SIZE = 32
    if img.width < MIN_SIZE or img.height < MIN_SIZE:
        return None

    img = common.to_rgb_tensor(img).to(device)
    input = position_transforms(img)

    if input.isnan().any():
        print("WARNING: Found nan after transforms.")
        return None

    output = models.fen.get()(input.unsqueeze(0)).squeeze(0)
    output = output.clamp(0, 1)

    board = common.tensor_to_position(output.cpu(), spec)
    if board.occupied == 0:
        return None
    return board


@dataclass
class FenResult:
    fen: str = None
    notation: str = None
    game: str = None
    cropped_image: Image.Image = None
    image_rotation_angle: int = None
    board_is_flipped: bool = None


def get_supported_games() -> list[str]:
    return list(GAMES.keys())


def get_fen(
    img: Image.Image,
    game: str,
    num_tries=10,
    auto_rotate_image=True,
    mirror_when_180_rotation=False,
    auto_rotate_board=True,
):
    """Takes an image and returns an FEN (Forsyth-Edwards Notation) string.

    Args:
        - `img (PIL.Image.Image)`: The image of a chess diagram.
        - `game (str)`: The game key (e.g. 'chess'). See get_supported_games().
        - `num_tries (int)`: The more higher this number is, the more accurate the returned FEN will be, with diminishing returns.
        - `auto_rotate_image (bool)`: If this is set to `True`, this function will try to guess if the image is rotated 0°, 90°, 180°,
        or 270° and rotate the image accordingly.
        - `mirror_when_180_rotation (bool)`: If this  and `auto_rotate_image` is set to `True`, this function will also mirror the image
        (left to right) if it was rotated 180°.
        - `auto_rotate_board (bool)`: If this is set to `True`, this function will try to guess if the diagram is from whites or blacks
        perspective and rotate the board accordingly.

    Returns:
        - `FenResult | None`: Returns a dataclass that contains the fields `fen`, `notation`, `game`, `cropped_image`, `image_rotation_angle`, and `board_is_flipped`.
        Returns `None` if there is no board detectable.
    """
    spec = get_game(game)

    img = img.convert("RGB")

    if not check_for_board_existence(img, spec.key):
        return None

    result = FenResult()
    result.cropped_image = crop_to_board(img, spec.key, max_num_tries=num_tries)
    if result.cropped_image is not None:
        result.image_rotation_angle = board_image_rotation(
            result.cropped_image, spec.key
        )

        if auto_rotate_image:
            result.cropped_image = result.cropped_image.rotate(
                -rotation_dataset.ROTATIONS[result.image_rotation_angle], expand=True
            )

            if (
                mirror_when_180_rotation
                and rotation_dataset.ROTATIONS[result.image_rotation_angle] == 180
            ):
                result.cropped_image = ImageOps.mirror(result.cropped_image)

        board = get_board_from_cropped_img(result.cropped_image, spec.key)

        if board is not None:
            result.board_is_flipped = is_board_flipped(board, spec.key)

            if auto_rotate_board and result.board_is_flipped:
                board = rotate_board(board, spec.key)

            result.fen = board.fen()
            result.notation = result.fen
            result.game = spec.key

    return result


def demo(root_dir: str, shuffle_files: bool, game: str):
    spec = get_game(game)

    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        print("Using CPU")

    torch.set_printoptions(precision=1, sci_mode=False)

    path = Path(root_dir)
    if path.is_dir():
        file_names = common.glob_all_image_files_recursively(path)
    else:
        file_names = [path]

    if shuffle_files:
        random.shuffle(file_names)

    for file_name in file_names:
        print(file_name)

        img = Image.open(file_name).convert("RGB")

        img = img.rotate(random.choice(rotation_dataset.ROTATIONS), expand=True)

        fen_result = get_fen(img, game=game)

        if fen_result is None:
            print("Couldn't detect board:", file_name)

        elif fen_result.fen is None:
            print("Couldn't detect FEN:", file_name)
        else:
            print(fen_result.fen)

        true_fen = common.normalize_position_notation(Path(file_name).stem, spec)
        if true_fen is None:
            print(f"WARNING: Couldn't find ground truth FEN")
        else:
            if fen_result is not None and fen_result.fen.split()[0] == true_fen.split()[0]:
                print("Correct")
            else:
                print(true_fen)
                print("WRONG")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

        ax1.imshow(img)
        if fen_result is not None:
            if fen_result.cropped_image is not None:
                ax2.imshow(fen_result.cropped_image)
            if fen_result.fen is not None:
                pos = common.position_from_notation(fen_result.fen, spec)
                if pos is not None:
                    fen_img = common.get_image(pos, width=512, height=512)
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
        help="directory that contains diagram images",
    )
    parser.add_argument(
        "--bbox_model",
        type=str,
        default=None,
        help="path to bbox model parameters",
    )
    parser.add_argument(
        "--fen_model",
        type=str,
        default=None,
        help="path to fen model parameters",
    )
    parser.add_argument(
        "--orientation_model",
        type=str,
        default=None,
        help="path to orientation_model model parameters",
    )
    parser.add_argument("--shuffle_files", action="store_true")
    parser.add_argument("--game", type=str, required=True)
    args = parser.parse_args()

    models = _get_models(args.game)
    if args.bbox_model is not None:
        models.bbox.set_model_path(args.bbox_model)
    if args.fen_model is not None:
        models.fen.set_model_path(args.fen_model)
    if args.orientation_model is not None:
        models.orientation.set_model_path(args.orientation_model)

    demo(root_dir=args.dir, shuffle_files=args.shuffle_files, game=args.game)
