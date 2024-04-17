import chess
import chess.pgn
import torch
import random
from torch.utils.data import Dataset
from src import common


class ChessBoardOrientationDataset(Dataset):

    def __init__(self, pgn_file_name, rotate_probability=0.3, max=100000):
        self.board_list = []
        self.rotate_probability = rotate_probability

        with open(pgn_file_name) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    self.board_list.append(board.copy())

                if len(self.board_list) >= max:
                    break

        random.shuffle(self.board_list)

        print(f"Found {len(self.board_list)} positions")

    def __len__(self):
        return len(self.board_list)

    def __getitem__(self, idx):
        rotate = random.uniform(0.0, 1.0) < self.rotate_probability

        input = common.chess_board_to_tensor(self.board_list[idx])
        if rotate:
            input = common.rotate_board_tensor(input)

        return (input, torch.tensor([1.0 if rotate else 0.0]))


def test_data_set():
    pgn_file = "resources/lichess_games/lichess_db_standard_rated_2013-05.pgn"

    c = ChessBoardOrientationDataset(pgn_file, max=100)

    for i in range(0, len(c)):
        input, target = c[i]

        assert not input.isnan().any()
        print(common.tensor_to_chess_board(input).fen())
        print("flipped:", target.item() == 1.0)
