# Import the modules
import chess
import numpy as np
import random
import os
import pyfastnoisesimd as fns
from io import BytesIO
from cairosvg import svg2png
from PIL import Image
from PIL import ImageOps
from pathlib import Path
from tqdm import tqdm

from src import consts

import chess
import random

RANDOM_OFFSET = max(1, consts.SQUARE_SIZE // 40)

PIECE_SETS = [
    ("lichess", "alpha"),
    ("lichess", "caliente"),
    ("lichess", "california"),
    ("lichess", "cardinal"),
    ("lichess", "cburnett"),
    ("lichess", "celtic"),
    ("lichess", "chess7"),
    ("lichess", "chessnut"),
    ("lichess", "companion"),
    ("lichess", "dubrovny"),
    ("lichess", "fantasy"),
    ("lichess", "fresca"),
    ("lichess", "gioco"),
    ("lichess", "governor"),
    ("lichess", "icpieces"),
    ("lichess", "kiwen-suwi"),
    ("lichess", "kosal"),
    ("lichess", "leipzig"),
    ("lichess", "libra"),
    ("lichess", "maestro"),
    ("lichess", "merida"),
    ("lichess", "mpchess"),
    ("lichess", "pirouetti"),
    ("lichess", "pixel"),
    ("lichess", "reillycraig"),
    ("lichess", "riohacha"),
    ("lichess", "spatial"),
    ("lichess", "staunty"),
    ("lichess", "tatiana"),
    ("extra", "glass"),
    ("extra", "8_bit"),
    ("extra", "bases"),
    ("extra", "book"),
    ("extra", "bubblegum"),
    ("extra", "cases"),
    ("extra", "celtic"),
    ("extra", "chicago"),
    ("extra", "classic"),
    ("extra", "club"),
    ("extra", "condal"),
    ("extra", "dash"),
    ("extra", "eyes"),
    ("extra", "falcon"),
    ("extra", "fantasy_alt"),
    ("extra", "game_room"),
    ("extra", "gothic"),
    ("extra", "graffiti"),
    ("extra", "icy_sea"),
    ("extra", "iowa"),
    ("extra", "light"),
    ("extra", "lolz"),
    ("extra", "marble"),
    ("extra", "maya"),
    ("extra", "metal"),
    ("extra", "modern"),
    ("extra", "nature"),
    ("extra", "neo"),
    ("extra", "neon"),
    ("extra", "neo_wood"),
    ("extra", "newspaper"),
    ("extra", "ocean"),
    ("extra", "oslo"),
    ("extra", "royale"),
    ("extra", "sky"),
    ("extra", "space"),
    ("extra", "spatial"),
    ("extra", "tigers"),
    ("extra", "tournament"),
    ("extra", "vintage"),
    ("extra", "wood"),
    ("custom", "a"),
    ("custom", "b"),
    ("custom", "c"),
    ("custom", "d"),
    ("custom", "e"),
]

PIECE_FILE_NAMES = {
    "lichess": [
        "bB.svg",
        "bK.svg",
        "bN.svg",
        "bP.svg",
        "bQ.svg",
        "bR.svg",
        "wB.svg",
        "wK.svg",
        "wN.svg",
        "wP.svg",
        "wQ.svg",
        "wR.svg",
    ],
    "extra": [
        "bb.png",
        "bk.png",
        "bn.png",
        "bp.png",
        "bq.png",
        "br.png",
        "wb.png",
        "wk.png",
        "wn.png",
        "wp.png",
        "wq.png",
        "wr.png",
    ],
    "custom": [
        "bb.png",
        "bk.png",
        "bn.png",
        "bp.png",
        "bq.png",
        "br.png",
        "wb.png",
        "wk.png",
        "wn.png",
        "wp.png",
        "wq.png",
        "wr.png",
    ],
}


def getUniformRandomBoard():
    max_color = 255
    min_color = 0
    if random.randint(0, 1) == 0:
        if random.randint(0, 1) == 0:
            max_color = random.randint(0, 255)
        else:
            min_color = random.randint(0, 255)
    color = (
        random.randint(min_color, max_color),
        random.randint(min_color, max_color),
        random.randint(min_color, max_color),
        255,
    )
    return Image.new(
        "RGBA", (consts.BOARD_PIXEL_WIDTH, consts.BOARD_PIXEL_WIDTH), color
    )


def getNoisyRandomGrayBoard():
    noise = fns.Noise()
    noise.noise_type = fns.NoiseType.Simplex
    noise.frequency = random.uniform(0.001, 0.06)
    # noise.seed=1234

    noise_array = noise.genAsGrid(
        shape=(consts.BOARD_PIXEL_WIDTH, consts.BOARD_PIXEL_WIDTH), start=(0, 0)
    )
    noise_array = np.interp(
        noise_array, (noise_array.min(), noise_array.max()), (0, 255)
    )
    noise_array = noise_array.astype(np.uint8)
    im = Image.fromarray(noise_array, mode="L")

    return im


def getNoisyRandomBoard():
    imgR = getNoisyRandomGrayBoard()
    imgG = getNoisyRandomGrayBoard()
    imgB = getNoisyRandomGrayBoard()

    return Image.merge("RGB", (imgR, imgG, imgB))


def getNoisyRandomBoardWithAlpha():
    imgR = getNoisyRandomGrayBoard()
    imgG = getNoisyRandomGrayBoard()
    imgB = getNoisyRandomGrayBoard()
    imgA = getNoisyRandomGrayBoard()

    return Image.merge("RGBA", (imgR, imgG, imgB, imgA))


BOARD_THEMES = [
    (None, getUniformRandomBoard),
    (None, getNoisyRandomGrayBoard),
    (None, getNoisyRandomBoard),
    ("lichess", "blue2.jpg"),
    ("lichess", "blue3.jpg"),
    ("lichess", "blue-marble.jpg"),
    ("lichess", "canvas2.jpg"),
    ("lichess", "green-plastic.png"),
    ("lichess", "grey.jpg"),
    ("lichess", "horsey.jpg"),
    ("lichess", "leather.jpg"),
    ("lichess", "maple2.jpg"),
    ("lichess", "maple.jpg"),
    ("lichess", "marble.jpg"),
    ("lichess", "metal.jpg"),
    ("lichess", "metal.orig.jpg"),
    ("lichess", "ncf-board.png"),
    ("lichess", "newspaper.png"),
    ("lichess", "olive.jpg"),
    ("lichess", "wood2.jpg"),
    ("lichess", "wood3.jpg"),
    ("lichess", "wood4.jpg"),
    ("lichess", "wood.jpg"),
    ("extra", "burled_wood.png"),
    ("extra", "christmas_alt.png"),
    ("extra", "christmas.png"),
    ("extra", "dark_wood.png"),
    ("extra", "dash.png"),
    ("extra", "glass.png"),
    ("extra", "graffiti.png"),
    ("extra", "icy_sea.png"),
    ("extra", "lolz.png"),
    ("extra", "marble.png"),
    ("extra", "metal.png"),
    ("extra", "neon.png"),
    ("extra", "newpaper.png"),
    ("extra", "parchment.png"),
    ("extra", "sand.png"),
    ("extra", "sea.png"),
    ("extra", "stone.png"),
    ("extra", "tournament.png"),
    ("extra", "walnut.png"),
    ("custom", "a.png"),
    ("custom", "b.png"),
]


# Define a function that takes an FEN string as input and returns a chessboard object
def fen_to_board(fen):
    # Create a new chessboard object from the FEN string
    board = chess.Board(fen)
    # Return the board object
    return board


# Define a function that takes an svg file name as input and returns a numpy array of the image
def svg_to_image(svg_file):
    # Open the local .svg file as bytes
    with open(svg_file, "rb") as f:
        svg_data = f.read()

    # Convert the SVG data to PNG format
    png_data = svg2png(
        bytestring=svg_data,
        output_width=consts.SQUARE_SIZE,
        output_height=consts.SQUARE_SIZE,
    )

    # Open the PNG data as a PIL image and convert it to RGB mode
    pil_img = Image.open(BytesIO(png_data)).convert("RGBA")

    return pil_img


# Define a function that takes a chessboard object and a dictionary of piece images as input and returns a PIL image object of the chessboard image with the pieces
def board_to_image(board, board_image, piece_images):
    # Load the chessboard image as a PIL image object
    board_image = board_image.copy()  # Image.open("./lichess_images/board/blue2.jpg")
    # Get the size of the chessboard image
    width, height = board_image.size
    # Get the size of each square on the chessboard image
    assert consts.SQUARE_SIZE == height // 8
    # Loop through each square on the chessboard object
    for square in chess.SQUARES:
        # Get the piece on the square
        piece = board.piece_at(square)
        # If there is a piece on the square
        if piece:
            # Get the color and the symbol of the piece
            color = piece.color
            symbol = piece.symbol()
            # Get the file name of the piece image
            piece_key = (color and "w" or "b") + symbol.lower()
            # Get the PIL image object of the piece image
            piece_image = piece_images[piece_key]
            if random.randint(0, 1) == 1:
                piece_image = ImageOps.mirror(piece_image)

            # Get the coordinates of the square on the chessboard image
            row = 7 - square // 8
            col = square % 8

            x = col * consts.SQUARE_SIZE + random.randint(-RANDOM_OFFSET, RANDOM_OFFSET)
            y = row * consts.SQUARE_SIZE + random.randint(-RANDOM_OFFSET, RANDOM_OFFSET)
            # Resize the piece image to fit the square size
            assert (consts.SQUARE_SIZE, consts.SQUARE_SIZE) == piece_image.size
            # Overlay the piece image on the chessboard image with transparency
            board_image.paste(piece_image, (x, y), piece_image)
    # Return the chessboard image with the pieces
    return board_image


def flip_piece_colors(board):
    new_board = chess.Board(None)

    for square in chess.SQUARES:
        p = board.piece_at(square)
        if p is not None:
            new_board.set_piece_at(
                square,
                chess.Piece(
                    p.piece_type, chess.BLACK if p.color == chess.WHITE else chess.WHITE
                ),
            )

    return new_board


# Define a function that returns a random chess position
def random_board():
    # Create an empty chessboard object
    board = chess.Board(None)
    # Loop through each square on the board
    approx_num_empty_squares = random.randint(1, 64)
    for square in chess.SQUARES:
        # Sometimes we leave the square empty
        if random.randint(1, 64) < approx_num_empty_squares:
            continue
        # Otherwise, create a piece object with the corresponding color and type
        else:
            # Generate a random number between 1 and 11
            n = random.randint(1, 12)
            # The color is white if the number is odd, black if the number is even
            color = n % 2 == 1
            # The type is determined by the number as follows:
            # 1 or 2: pawn
            # 3 or 4: knight
            # 5 or 6: bishop
            # 7 or 8: rook
            # 9 or 10: queen
            # 11: king
            piece_type = (n + 1) // 2
            piece = chess.Piece(piece_type, color)
            # Set the piece on the square
            board.set_piece_at(square, piece)

    return board


def generate_fen_training_data(
    num_total_out_positions=300000,
    outdir_root="resources/generated_images/chessboards_fen",
    max_files_per_folder=10000,
):

    num_fens_per_combo = max(
        1, num_total_out_positions // (len(PIECE_SETS) * len(BOARD_THEMES))
    )
    print("num_fens_per_combo:", num_fens_per_combo)

    current_files_in_folder = 0
    current_outdir = None

    for piece_dir, piece_set in tqdm(PIECE_SETS):
        piece_images = {}
        # Loop through the 12 svg file names
        for file_name in PIECE_FILE_NAMES[piece_dir]:
            # Convert the svg file to a numpy array and store it in the dictionary
            path = Path(f"./resources/pieces/{piece_dir}/{piece_set}/{file_name}")
            if path.suffix == ".svg":
                img = svg_to_image(path)
            else:
                img = Image.open(path).convert("RGBA")
                img = img.resize((consts.SQUARE_SIZE, consts.SQUARE_SIZE))

            piece_key = path.stem.lower()
            piece_images[piece_key] = img

        for board_dir, board_theme in BOARD_THEMES:
            if board_dir is not None:
                board_image = Image.open(
                    f"./resources/board_themes/{board_dir}/{board_theme}"
                ).convert("RGBA")
                board_image.putalpha(255)
                board_image = board_image.resize(
                    (consts.BOARD_PIXEL_WIDTH, consts.BOARD_PIXEL_WIDTH)
                )

            # print(piece_dir, piece_set, "|", board_dir, board_theme)

            for i in range(0, num_fens_per_combo):
                if board_dir is None:
                    current_board_image = board_theme()
                else:
                    current_board_image = board_image.copy()

                    if random.randint(0, 1) == 1:
                        current_board_image = ImageOps.mirror(current_board_image)
                    if random.randint(0, 1) == 1:
                        current_board_image = ImageOps.flip(current_board_image)
                    if random.randint(0, 1) == 1:
                        noise = (
                            getNoisyRandomBoard()
                            if random.randint(0, 1) == 1
                            else getNoisyRandomGrayBoard()
                        )
                        current_board_image.paste(noise, mask=getNoisyRandomGrayBoard())

                if (
                    current_outdir is None
                    or current_files_in_folder >= max_files_per_folder
                ):
                    current_files_in_folder = 0
                    current_outdir = None
                    for i in range(0, num_total_out_positions):
                        potential_dir = outdir_root + "/" + str(i)
                        if not os.path.exists(potential_dir):
                            current_outdir = potential_dir
                            break
                    assert current_outdir is not None
                    os.makedirs(current_outdir, exist_ok=True)
                    # print("Current dir:", current_outdir)

                board = random_board()
                image = board_to_image(
                    board, current_board_image, piece_images
                ).convert("RGB")

                if random.randint(0, 1) == 1:
                    board = flip_piece_colors(board)
                    image = ImageOps.invert(image)

                fake_fen = board.fen().replace("/", "_").replace(" ", "+")
                image.save(current_outdir + "/" + fake_fen + ".png")
                current_files_in_folder += 1
