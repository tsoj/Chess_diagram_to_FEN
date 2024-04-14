
import argparse

import src.bounding_box.generate_chessboards_bbox as bbox_gen
import src.fen_recognition.generate_chessboards as fen_gen

import src.bounding_box.dataset as bbox_data
import src.fen_recognition.dataset as fen_data
import src.board_orientation.dataset as orienation_data

import src.bounding_box.train as bbox_train
import src.fen_recognition.train as fen_train
import src.board_orientation.train as orienation_train

import src.fen_recognition.show_wrong_evals as fen_eval
import src.board_orientation.show_wrong_evals as orienation_eval

import src.board_image_rotation.dataset as image_rotation_data
import src.board_image_rotation.train as image_rotation_train
import src.board_image_rotation.show_wrong_evals as image_rotation_eval



GENERATE= "generate"
DATASET= "dataset"
TRAIN= "train"
EVAL= "eval"

BBOX="bbox"
FEN="fen"
ORIENTATION="orientation"
IMAGE_ROTATION="image_rotation"

functions = {
    (GENERATE, BBOX): bbox_gen.generate_bbox_training_data,
    (DATASET, BBOX): bbox_data.test_data_set,
    (TRAIN, BBOX): bbox_train.train,

    (GENERATE, FEN): fen_gen.generate_fen_training_data,
    (DATASET, FEN): fen_data.test_data_set,
    (TRAIN, FEN): fen_train.train,
    (EVAL, FEN): fen_eval.show_wrong_fens,

    (DATASET, ORIENTATION): orienation_data.test_data_set,
    (TRAIN, ORIENTATION): orienation_train.train,
    (EVAL, ORIENTATION): orienation_eval.show_wrong_orientation_evals,

    (DATASET, IMAGE_ROTATION): image_rotation_data.test_data_set,
    (TRAIN, IMAGE_ROTATION): image_rotation_train.train,
    (EVAL, IMAGE_ROTATION): image_rotation_eval.show_wrong_image_rotations,
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("function", choices=[GENERATE, DATASET, TRAIN, EVAL], help="the function to run the program")
    parser.add_argument("model", nargs='?', choices=[BBOX, FEN, ORIENTATION, IMAGE_ROTATION, None], default=None, help="the model to use")
    parser.add_argument("--dir", type=str, help="directory that contains images of chess diagrams")
    args = parser.parse_args()

    selection = (args.function, args.model)
    if selection not in functions:
        raise Exception(f"Selection {selection} not supported\nSupported selections: {list(functions.keys())}")

    functions[selection]()
