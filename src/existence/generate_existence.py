from pathlib import Path
import random
import time
import os
from PIL import Image
from tqdm import tqdm

from src import consts

MIN_CROP_SIZE = consts.BOARD_PIXEL_WIDTH + 10
MAX_CROP_SIZE = consts.BOARD_PIXEL_WIDTH * 4

TARGET_SIZE = consts.BOARD_PIXEL_WIDTH * 2


def generate_existence_training_data(
    outdir="resources/generated_images/no_chessboards",
    background_root_dir="resources/website_screenshots",
    num_total_out_positions=50000,
):
    os.makedirs(outdir, exist_ok=True)

    background_root_dir = Path(background_root_dir)
    assert (
        background_root_dir.is_dir()
    ), f"With background_root_dir = {background_root_dir}"
    background_image_files = list(background_root_dir.glob("**/*.jpg"))
    random.shuffle(background_image_files)

    if num_total_out_positions is not None:
        background_image_files = background_image_files[
            0 : min(len(background_image_files), num_total_out_positions)
        ]

    for i, background_file_name in enumerate(tqdm(background_image_files)):
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
        img = img.resize((TARGET_SIZE, TARGET_SIZE))

        file_name = (
            outdir
            + f"/{i}.jpg"
        )

        # # Save the result
        img.save(file_name)
