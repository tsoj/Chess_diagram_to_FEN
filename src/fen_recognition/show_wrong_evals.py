import torch
import matplotlib.pyplot as plt
from io import BytesIO
from cairosvg import svg2png
from PIL import Image
import chess.svg

from src import consts, common
from src.fen_recognition import dataset
from src.fen_recognition.model import ChessRec


def show_wrong_fens(model_path="models/best_model_fen_0.953_2024-02-03-13-49-31.pth"):

    max_data = None
    data_root_dir = "resources/fen_images"

    chess_board_set = dataset.ChessBoardDataset(
        root_dir=data_root_dir,
        augment_ratio=0.5,
        affine_augment_ratio=0.8,
        max=max_data
    )

    model = ChessRec()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    correct = 0
    total = 0
    for img, target in chess_board_set:

        total += 1

        with torch.no_grad():
            output = model(img.unsqueeze(0)).squeeze(0)

        board = common.tensor_to_chess_board(output)
        true_board = common.tensor_to_chess_board(target)

        if board.fen() == true_board.fen():
            correct += 1
            print("Correct:", board.fen())
            continue

        print("WRONG:")
        print(true_board.fen())
        print(board.fen())
        print(correct / total)

        svg_data = chess.svg.board(board, size=consts.BOARD_PIXEL_WIDTH)

        png_data = svg2png(
            bytestring=svg_data,
            output_width=consts.BOARD_PIXEL_WIDTH,
            output_height=consts.BOARD_PIXEL_WIDTH,
        )

        # Open the PNG data as a PIL image and convert it to RGB mode
        pil_img = Image.open(BytesIO(png_data)).convert("RGBA")

        f, ax = plt.subplots(1, 2, figsize=(16, 8))

        # # Display the image using matplotlib
        ax[0].imshow(pil_img)
        ax[0].axis("off")

        ax[1].imshow((img.permute(1, 2, 0) - img.min()) / (img.max() - img.min()))
        ax[1].axis("off")
        plt.show()
