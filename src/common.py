import torch
from torchvision.transforms import v2
from PIL import Image
import chess
from io import BytesIO
from cairosvg import svg2png
import chess.svg
import re
from pathlib import Path


def to_rgb_tensor(img):
    if isinstance(img, Image.Image):
        img = v2.PILToTensor()(img)

    # get the shape of the tensor
    ch, h, w = img.shape
    # check if the channel dimension is valid
    if ch not in [1, 3, 4]:
        raise ValueError("Channel dimension must be 1, 3 or 4")
    # check if the tensor is of type uint8
    if not img.dtype == torch.uint8:
        raise TypeError("Image must be of type uint8")
    img = img.float() / 255.0
    # if the channel dimension is 4, remove the last (alpha) channel
    if ch == 4:
        img = img[:3, :, :]
    # if the channel dimension is 1 (grayscale image), duplicate the channel three times
    if ch == 1:
        img = img.repeat(3, 1, 1)
    # return the converted tensor
    return img


class MinMaxMeanNormalization(torch.nn.Module):
    def forward(self, tensor):
        min = tensor.min()
        max = tensor.max()
        if min >= max:
            return torch.zeros_like(tensor)
        tensor = (tensor - min) / (max - min)
        tensor -= tensor.mean()
        if torch.isnan(tensor).any():
            print("WARNING: Encountered NaN in input for MinMaxMeanNormalization")
            tensor = torch.zeros_like(tensor)
        assert tensor.mean().abs() < 0.0001, tensor.mean()
        assert 0.0 <= tensor.max() <= 1.0
        assert -1.0 <= tensor.min() <= 0.0
        return tensor


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0, scale_to_input_range=False):
        super().__init__()
        self.std = std
        self.mean = mean
        self.scale_to_input_range = scale_to_input_range

    def forward(self, tensor):
        std = self.std
        mean = self.mean
        if self.scale_to_input_range:
            range = tensor.max() - tensor.min()
            std *= range
            mean *= range
        return tensor + torch.randn_like(tensor) * std + mean


def pad(img: Image, x, y):
    x = int(x)
    y = int(y)
    # create a new image with the padded size
    new_width = img.width + x * 2
    new_height = img.height + y * 2
    new_img = Image.new("RGB", (new_width, new_height), "white")

    # paste the original image on the new image
    new_img.paste(img.convert("RGB"), (x, y))
    return new_img


PIECE_TYPES = [
    chess.Piece(chess.PAWN, chess.WHITE),
    chess.Piece(chess.KNIGHT, chess.WHITE),
    chess.Piece(chess.BISHOP, chess.WHITE),
    chess.Piece(chess.ROOK, chess.WHITE),
    chess.Piece(chess.QUEEN, chess.WHITE),
    chess.Piece(chess.KING, chess.WHITE),
    chess.Piece(chess.PAWN, chess.BLACK),
    chess.Piece(chess.KNIGHT, chess.BLACK),
    chess.Piece(chess.BISHOP, chess.BLACK),
    chess.Piece(chess.ROOK, chess.BLACK),
    chess.Piece(chess.QUEEN, chess.BLACK),
    chess.Piece(chess.KING, chess.BLACK),
    None,
]


def square_to_idx(sq):
    return (sq % 8) + (7 - (sq // 8)) * 8


def chess_board_to_tensor(board: chess.Board):
    result = torch.zeros(64, len(PIECE_TYPES))

    for square in chess.SQUARES:
        sq_idx = square_to_idx(square)
        for i, piece in enumerate(PIECE_TYPES):
            if piece is None:
                result[sq_idx, i] = 1.0
            elif board.piece_at(square) == piece:
                result[sq_idx, i] = 1.0
                break

    return result


def tensor_to_chess_board(tensor: torch.Tensor):
    board = chess.Board(None)

    for square in chess.SQUARES:
        index = tensor[square_to_idx(square)].argmax().item()
        if PIECE_TYPES[index] is not None:
            board.set_piece_at(square, PIECE_TYPES[index])

    return board


def flip_color(tensor: torch.Tensor):
    flipped = torch.zeros_like(tensor)

    for square in chess.SQUARES:
        sq_idx = square_to_idx(square)
        for i, piece in enumerate(PIECE_TYPES):
            if piece is None:
                flipped[sq_idx, i] = tensor[sq_idx, i]
            else:
                other_piece_i = PIECE_TYPES.index(
                    chess.Piece(
                        piece.piece_type,
                        chess.BLACK if piece.color == chess.WHITE else chess.WHITE,
                    )
                )
                flipped[sq_idx, other_piece_i] = tensor[sq_idx, i]
    return flipped


def rotate_board_tensor(tensor: torch.Tensor):
    mirrored = torch.zeros_like(tensor)

    for square in chess.SQUARES:
        sq_idx = square_to_idx(square)
        for i in range(0, len(PIECE_TYPES)):
            mirrored_sq_idx = 7 - (sq_idx % 8) + (7 - (sq_idx // 8)) * 8
            mirrored[mirrored_sq_idx, i] = tensor[sq_idx, i]
    return mirrored


def get_image(board: chess.Board, width, height):
    fen_svg_data = chess.svg.board(board, size=height)

    fen_png_data = svg2png(
        bytestring=fen_svg_data, output_width=width, output_height=height
    )

    # Open the PNG data as a PIL image and convert it to RGB mode
    return Image.open(BytesIO(fen_png_data)).convert("RGBA")


def normalize_fen(pseudo_fen: str) -> str:
    fen = pseudo_fen
    fen = fen.replace("_", "/").replace("+", " ").replace(".", " ")
    fen = fen.split()[0]
    fen = fen.replace("11111111", "8")
    fen = fen.replace("1111111", "7")
    fen = fen.replace("111111", "6")
    fen = fen.replace("11111", "5")
    fen = fen.replace("1111", "4")
    fen = fen.replace("111", "3")
    fen = fen.replace("11", "2")
    fen = fen.replace("-", "/")

    fen += " w - - 0 1"

    pattern = r"^([rnbqkpRNBQKP1-8]+/){7}([rnbqkpRNBQKP1-8]+) ([wb]) ([-kqKQ]+|-) ([a-h][36]|-) \d+ \d+$"
    if not re.match(pattern, fen):
        return None

    try:
        return chess.Board(fen).fen()
    except ValueError:
        return None


def glob_all_image_files_recursively(dir) -> list:
    return list(
        (
            p.resolve()
            for p in Path(dir).glob("**/*")
            if p.suffix in {".png", ".jpeg", ".jpg"}
        )
    )
