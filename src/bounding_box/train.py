# Import modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import os

from src.bounding_box import dataset
from src.bounding_box.model import ChessBoardBBox
from src.bounding_box.inference import get_bbox

import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_box_iou(loader, model):
    box_iou = 0.0
    num_samples = 0
    for imgs, target_boxes, target_masks in loader:
        for img, target_box, target_mask in zip(imgs, target_boxes, target_masks):
            output_box = get_bbox(model, img)
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
TEST_ACC_FREQ = 200


def train(
    data_root_dir="resources/generated_images/chessboards_bbox",
    outdir="models",
    batch_size=8,
    num_epochs=2,
    train_test_split=0.95,
    augment_ratio=0.4,
    max_data=None,
):
    start_time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(start_time_string)

    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        print("Using CPU")

    chess_board_set = dataset.ChessBoardBBoxDataset(
        root_dir=data_root_dir, augment_ratio=augment_ratio, max=max_data, device=device
    )
    train_set, test_set = torch.utils.data.random_split(
        chess_board_set, [train_test_split, 1.0 - train_test_split]
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=True
    )

    model = ChessBoardBBox()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs
    )

    test_box_iou_list = []
    best_box_iou = -1.0
    best_model = None

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (img, target_box, target_mask) in enumerate(train_loader):
            img = img.to(device)
            target_mask = target_mask.to(device)

            optimizer.zero_grad()

            output = model(img)
            loss = criterion(output, target_mask)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if (i + 1) % LOSS_REPORT_FREQ == 0:
                print(
                    "[%d, %4d] loss: %.4f, lr: %.5f"
                    % (
                        epoch + 1,
                        i + 1,
                        running_loss / LOSS_REPORT_FREQ,
                        optimizer.param_groups[0]["lr"],
                    )
                )
                running_loss = 0.0

            if (i + 1) % TEST_ACC_FREQ == 0 or (i + 1) >= len(train_loader):
                test_box_iou = get_box_iou(test_loader, model)
                test_box_iou_list.append(test_box_iou)
                print("Epoch %d: Test IOU: %.3f" % (epoch + 1, test_box_iou_list[-1]))

                if test_box_iou > best_box_iou:
                    best_box_iou = test_box_iou
                    best_model = model.state_dict()
                    print("Best model updated: Test IOU: %.3f" % best_box_iou)

    os.makedirs(outdir, exist_ok=True)
    file_name = outdir + "/best_model_bbox_%.3f_%s.pth" % (
        best_box_iou,
        start_time_string,
    )
    print("Saving to", file_name)
    torch.save(best_model, file_name)

    # Plot the loss and accuracy curves
    plt.figure(figsize=(12, 4))
    plt.plot(test_box_iou_list, label="Test IOU")
    plt.xlabel("Epoch")
    plt.ylabel("IOU")
    plt.legend()
    plt.savefig(file_name + ".png", dpi=250)
