from pathlib import Path
import random
import time
import os
from PIL import Image
from tqdm import tqdm

from src import consts, common

MIN_CROP_SIZE = consts.BOARD_PIXEL_WIDTH + 10
MAX_CROP_SIZE = consts.BOARD_PIXEL_WIDTH * 4

TARGET_SIZE = consts.BOARD_PIXEL_WIDTH * 2


def generate_bbox_training_data(
    outdir="resources/chessboards_bbox_images/chessboards_bbox",
    background_root_dir="resources/website_screenshots",
    board_root_dir="resources/fen_images/generated_chessboards_fen",
    num_total_out_positions=50000,
    chessboard_middleground_probability=0.4,
):
    os.makedirs(outdir, exist_ok=True)

    background_root_dir = Path(background_root_dir)
    assert (
        background_root_dir.is_dir()
    ), f"With background_root_dir = {background_root_dir}"
    background_image_files = common.glob_all_image_files_recursively(background_root_dir)
    random.shuffle(background_image_files)
    if num_total_out_positions is not None:
        background_image_files = background_image_files[
            0 : min(len(background_image_files), num_total_out_positions)
        ]

    board_root_dir = Path(board_root_dir)
    assert board_root_dir.is_dir(), f"With board_root_dir = {board_root_dir}"
    board_image_files = common.glob_all_image_files_recursively(board_root_dir)
    random.shuffle(board_image_files)

    assert len(board_image_files) >= len(
        background_image_files
    ), "There at least as many board image files as background images"

    for background_file_name, board_file_name in zip(
        tqdm(background_image_files), board_image_files
    ):
        img = Image.open(background_file_name).convert("RGB")

        # Get the image dimensions
        img_width, img_height = img.size

        # Choose a random width and height
        max_size = min(img_width, img_height, MAX_CROP_SIZE)
        crop_width = random.randint(MIN_CROP_SIZE, max_size)
        crop_height = random.randint(MIN_CROP_SIZE, max_size)

        # Choose a random position for the crop
        x = random.randint(0, img_width - crop_width)
        y = random.randint(0, img_height - crop_height)

        # Crop the image
        img = img.crop((x, y, x + crop_width, y + crop_height))

        # Load the second image
        board_image = Image.open(board_file_name).convert("RGB")

        # Get the dimensions of the second image
        board_image_width, board_image_height = board_image.size
        new_size = random.randint(
            min(crop_width, crop_height, board_image_width // 2),
            min(crop_width, crop_height),
        )
        board_image = board_image.resize((new_size, new_size))
        board_image_width, board_image_height = board_image.size

        assert board_image_width <= crop_width
        assert board_image_height <= crop_height

        # Choose a random position for the second image
        board_x = random.randint(0, crop_width - board_image_width)
        board_y = random.randint(0, crop_height - board_image_height)

        center_x = board_x + board_image_width // 2
        center_y = board_y + board_image_height // 2

        if random.uniform(0.0, 1.0) < chessboard_middleground_probability:
            # we randomly select a maximum relative size to the foreground board since the size of
            # the middleground board would otherwise be biased towards being bigger than the foreground board
            # because of the way we make sure that the boards have to overlap
            max_relative_size = random.uniform(0.1, 3.0)
            max_aspect_ratio = random.uniform(1.0, 4.0)
            while True:
                middleground_width = random.randint(
                    min(crop_width, 4, board_image_width // 2), crop_width
                )
                middleground_height = random.randint(
                    min(crop_height, 4, board_image_height // 2), crop_height
                )
                middleground_x = random.randint(0, crop_width - middleground_width)
                middleground_y = random.randint(0, crop_height - middleground_height)

                # make sure the foreground board doesn't overlap the middleground completely
                if (
                    middleground_x > board_x
                    and middleground_y > board_y
                    and middleground_x + middleground_width
                    < board_x + board_image_width
                    and middleground_y + middleground_height
                    < board_y + board_image_height
                ):
                    continue

                # check if the max size constraint holds
                if (
                    board_image_width * board_image_height * max_relative_size
                    < middleground_height * middleground_width
                ):
                    continue

                # check if the aspect ration constraint holds
                if (
                    max(
                        middleground_height / middleground_width,
                        middleground_width / middleground_height,
                    )
                    > max_aspect_ratio
                ):
                    continue

                # check if middleground image would overlap with foreground board (otherwise it wouldn't be possible to decide which board is meant to be detected)
                if (
                    middleground_x > board_x + board_image_width
                    or board_x > middleground_x + middleground_width
                ):
                    continue
                if (
                    middleground_y > board_y + board_image_height
                    or board_y > middleground_y + middleground_height
                ):
                    continue

                middleground_img = Image.open(random.choice(board_image_files)).convert(
                    "RGB"
                )
                middleground_img = middleground_img.resize(
                    (middleground_width, middleground_height)
                )
                img.paste(middleground_img, (middleground_x, middleground_y))

                break

        # Paste the second image on the cropped image
        img.paste(board_image, (board_x, board_y))

        scale_x = TARGET_SIZE / crop_width
        scale_y = TARGET_SIZE / crop_height

        img = img.resize((TARGET_SIZE, TARGET_SIZE))
        center_x *= scale_x
        center_y *= scale_y
        board_image_width *= scale_x
        board_image_height *= scale_y

        # draw = ImageDraw.Draw(img)
        # draw.rectangle(
        #     (
        #         center_x - board_image_width // 2,
        #         center_y - board_image_height // 2,
        #         center_x + board_image_width // 2,
        #         center_y + board_image_height // 2,
        #     ),
        #     outline="red",
        #     fill=None,
        # )

        file_name = (
            outdir
            + f"/{int(center_x)}_{int(center_y)}_{int(board_image_width)}_{int(board_image_height)}.jpg"
        )

        # # Save the result
        img.save(file_name)
