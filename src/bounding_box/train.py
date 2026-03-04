import datetime
import os

import matplotlib
import torch
import torch.optim as optim
import torchvision
from tqdm.auto import tqdm

from src.bounding_box import dataset
from src.bounding_box.model import BoardBBox
from src.bounding_box.inference import get_bbox
from src import consts

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_box_iou(test_data, model):
    model.eval()
    model.to(device)
    box_iou = 0.0
    num_samples = 0
    with torch.no_grad():
        for img, target in test_data:
            output_box = get_bbox(model, img)
            target_box = target["boxes"].squeeze(0)
            if output_box is not None:
                box_iou += (
                    torchvision.ops.box_iou(
                        output_box.unsqueeze(0), target_box.unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
            num_samples += 1
    return box_iou / num_samples


LOSS_REPORT_FREQ = 50
TEST_ACC_FREQ = 1000


def train(
    game: str,
    outdir="models",
    total_steps=50_000,
    batch_size=8,
    max_lr=0.0003,
    test_set_size=2000,
):
    start_time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(start_time_string)

    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        print("Using CPU")

    train_set = dataset.GenerativeBboxDataset(
        game=game,
        augment_ratio=0.4,
    )
    test_set = dataset.generate_fixed_test_set(game=game, size=test_set_size)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, drop_last=True,
        num_workers=8, persistent_workers=True,
        collate_fn=dataset.collate_fn,
    )

    model = BoardBBox()
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=max_lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
    )

    test_box_iou_list = []
    best_box_iou = -1.0
    best_model = None
    num_steps = 0
    progress = tqdm(total=total_steps, desc="Training", dynamic_ncols=True, leave=True)

    for images, targets in train_loader:
        model.train()
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        num_steps += 1
        progress.update(1)
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.5f}",
        )

        if num_steps % LOSS_REPORT_FREQ == 0:
            tqdm.write(
                f"[{num_steps}/{total_steps}] "
                f"loss: {loss.item():.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.5f}"
            )

        if num_steps % TEST_ACC_FREQ == 0 or num_steps >= total_steps:
            test_box_iou = get_box_iou(test_set, model)
            test_box_iou_list.append(test_box_iou)
            tqdm.write(f"Num steps: {num_steps}, Test IOU: {test_box_iou_list[-1]:.3f}")

            if test_box_iou > best_box_iou:
                best_box_iou = test_box_iou
                best_model = model.state_dict()
                tqdm.write(f"Best model updated: Test IOU: {best_box_iou:.3f}")

        if num_steps >= total_steps:
            break
    progress.close()

    game_outdir = os.path.join(outdir, game)
    os.makedirs(game_outdir, exist_ok=True)
    file_name = f"{game_outdir}/best_model_bbox_{best_box_iou:.3f}_{start_time_string}.pth"
    print("Saving to", file_name)
    torch.save(best_model, file_name)

    plt.figure(figsize=(12, 4))
    plt.plot(test_box_iou_list, label="Test IOU")
    plt.xlabel("Step")
    plt.ylabel("IOU")
    plt.legend()
    plt.savefig(file_name + ".png", dpi=250)
