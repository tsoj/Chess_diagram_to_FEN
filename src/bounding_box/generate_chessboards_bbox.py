import random
from pathlib import Path

from PIL import Image, ImageOps

from src import common, consts
from src.fen_recognition.generate_chessboards import BoardGenerator

MIN_CROP_SIZE = consts.BOARD_PIXEL_WIDTH + 10
MAX_CROP_SIZE = consts.BOARD_PIXEL_WIDTH * 4

TARGET_SIZE = consts.BOARD_PIXEL_WIDTH * 2


class BboxGenerator:
    def __init__(
        self,
        game: str,
        background_root_dir: str = "resources/website_screenshots",
        board_middleground_probability: float = 0.4,
    ):
        self.board_generator = BoardGenerator(game)
        self.board_middleground_probability = board_middleground_probability

        background_root_dir = Path(background_root_dir)
        assert background_root_dir.is_dir(), (
            f"With background_root_dir = {background_root_dir}"
        )
        self.background_image_files = common.glob_all_image_files_recursively(
            background_root_dir
        )
        assert len(self.background_image_files) > 0, "No background images found"

    def generate_one(self) -> tuple[Image.Image, tuple[int, int, int, int]]:
        use_board_background = random.random() >= 0.1
        board_bg = (
            self.board_generator.generate_board_background(
                self.board_generator.board_w, self.board_generator.board_h
            )
            if use_board_background
            else None
        )
        board_image, _ = self.board_generator.generate_one(
            use_board_background=use_board_background,
            board_background=board_bg,
        )

        # Pick random background screenshot and crop
        bg_path = random.choice(self.background_image_files)
        img = Image.open(bg_path).convert("RGB")

        img_width, img_height = img.size
        max_size = min(img_width, img_height, MAX_CROP_SIZE)
        crop_width = random.randint(MIN_CROP_SIZE, max_size)
        crop_height = random.randint(MIN_CROP_SIZE, max_size)

        x = random.randint(0, img_width - crop_width)
        y = random.randint(0, img_height - crop_height)
        img = img.crop((x, y, x + crop_width, y + crop_height))

        # Resize board (and its background) to fit within crop
        board_image_width, board_image_height = board_image.size
        board_aspect = board_image_width / max(board_image_height, 1)
        max_w = min(crop_width, max(4, int(crop_height * board_aspect)))
        max_h = min(crop_height, max(4, int(crop_width / max(board_aspect, 1e-6))))
        new_width = random.randint(max(4, min(max_w, board_image_width // 2)), max_w)
        new_height = max(4, int(new_width / max(board_aspect, 1e-6)))
        if new_height > max_h:
            new_height = max_h
            new_width = max(4, int(new_height * board_aspect))
        board_image = board_image.resize((new_width, new_height))
        board_image_width, board_image_height = board_image.size
        if board_bg is not None:
            board_bg = board_bg.resize((board_image_width, board_image_height))

        assert board_image_width <= crop_width
        assert board_image_height <= crop_height

        board_x = random.randint(0, crop_width - board_image_width)
        board_y = random.randint(0, crop_height - board_image_height)

        center_x = board_x + board_image_width // 2
        center_y = board_y + board_image_height // 2

        if random.uniform(0.0, 1.0) < self.board_middleground_probability and not use_board_background:
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

                if (
                    middleground_x > board_x
                    and middleground_y > board_y
                    and middleground_x + middleground_width
                    < board_x + board_image_width
                    and middleground_y + middleground_height
                    < board_y + board_image_height
                ):
                    continue

                if (
                    board_image_width * board_image_height * max_relative_size
                    < middleground_height * middleground_width
                ):
                    continue

                if (
                    max(
                        middleground_height / middleground_width,
                        middleground_width / middleground_height,
                    )
                    > max_aspect_ratio
                ):
                    continue

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

                # Generate middleground board in-memory
                middleground_img, _ = self.board_generator.generate_one()
                middleground_img = middleground_img.resize(
                    (middleground_width, middleground_height)
                )
                img.paste(middleground_img, (middleground_x, middleground_y))
                break

        # Some cases (opaque boards only): show thin strips of the board background
        # tiled in a mirror-grid pattern around the board, making edges ambiguous.
        if use_board_background and random.random() < 0.8:
            assert board_bg is not None
            strip_w = random.randint(1, max(1, board_image_width // 5))
            strip_h = random.randint(1, max(1, board_image_height // 5))

            bg_mirror = ImageOps.mirror(board_bg)
            bg_flip = ImageOps.flip(board_bg)
            bg_both = ImageOps.flip(bg_mirror)

            for tx in (-1, 0, 1):
                for ty in (-1, 0, 1):
                    if tx == 0 and ty == 0:
                        continue

                    if tx != 0 and ty != 0:
                        tile = bg_both
                    elif tx != 0:
                        tile = bg_mirror
                    else:
                        tile = bg_flip

                    # Destination rectangle in crop coords, limited to strip_w/strip_h
                    if tx == -1:
                        dst_x0 = board_x - strip_w
                        dst_x1 = board_x
                        src_x0 = board_image_width - strip_w
                    elif tx == 1:
                        dst_x0 = board_x + board_image_width
                        dst_x1 = board_x + board_image_width + strip_w
                        src_x0 = 0
                    else:
                        dst_x0 = board_x
                        dst_x1 = board_x + board_image_width
                        src_x0 = 0

                    if ty == -1:
                        dst_y0 = board_y - strip_h
                        dst_y1 = board_y
                        src_y0 = board_image_height - strip_h
                    elif ty == 1:
                        dst_y0 = board_y + board_image_height
                        dst_y1 = board_y + board_image_height + strip_h
                        src_y0 = 0
                    else:
                        dst_y0 = board_y
                        dst_y1 = board_y + board_image_height
                        src_y0 = 0

                    # Clip to crop bounds
                    clip_x0 = max(dst_x0, 0)
                    clip_y0 = max(dst_y0, 0)
                    clip_x1 = min(dst_x1, crop_width)
                    clip_y1 = min(dst_y1, crop_height)

                    if clip_x1 <= clip_x0 or clip_y1 <= clip_y0:
                        continue

                    src = tile.crop(
                        (
                            src_x0 + clip_x0 - dst_x0,
                            src_y0 + clip_y0 - dst_y0,
                            src_x0 + clip_x1 - dst_x0,
                            src_y0 + clip_y1 - dst_y0,
                        )
                    )
                    img.paste(src, (clip_x0, clip_y0))

        if use_board_background:
            img.paste(board_image, (board_x, board_y))
        else:
            img.paste(board_image, (board_x, board_y), board_image)

        scale_x = TARGET_SIZE / crop_width
        scale_y = TARGET_SIZE / crop_height

        img = img.resize((TARGET_SIZE, TARGET_SIZE))
        center_x = int(center_x * scale_x)
        center_y = int(center_y * scale_y)
        board_image_width = int(board_image_width * scale_x)
        board_image_height = int(board_image_height * scale_y)

        return img, (center_x, center_y, board_image_width, board_image_height)
