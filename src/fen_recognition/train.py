import datetime
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from src import common, consts
from src.fen_recognition import dataset
from src.fen_recognition.model import AblationConfig, BoardRec
from src.games import get_game

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_accuracy_and_loss(loader, model, criterion, game: str):
    spec = get_game(game)
    num_correct = 0
    num_samples = 0
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for img, target in loader:
            img = img.to(device)
            target = target.to(device)
            output = model(img)

            loss += criterion(output, target).item() * target.size(0)

            output = output.cpu()
            target = target.cpu()

            assert output.size(0) == target.size(0)
            for i in range(0, output.size(0)):
                if common.tensor_to_position(output[i], spec).piece_placement == common.tensor_to_position(target[i], spec).piece_placement:
                    num_correct += 1

            num_samples += target.size(0)
    model.train()
    return num_correct / num_samples, loss / num_samples


LOSS_REPORT_FREQ = 200
TEST_ACC_FREQ = 2000


def train(
    game: str,
    outdir="models",
    total_steps=60_000,
    batch_size=8,
    lr=0.00004,
    test_set_size=2000,
    checkpoint=None,
    tile_size: int = consts.DEFAULT_TILE_SIZE,
    ablation_config: AblationConfig | None = None,
):
    spec = get_game(game)

    start_time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(start_time_string)
    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        print("Using CPU")
    print("Game:", spec.key)

    train_set = dataset.GenerativeBoardDataset(
        game=spec.key,
        tile_size=tile_size,
        augment_ratio=0.8,
        affine_augment_ratio=0.8,
    )
    test_set = dataset.generate_fixed_test_set(game=spec.key, size=test_set_size, tile_size=tile_size)

    def _worker_init_fn(worker_id):
        random.seed(torch.initial_seed() % 2**32)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, drop_last=True,
        num_workers=8, worker_init_fn=_worker_init_fn, persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    model = BoardRec(game=spec.key, tile_size=tile_size, ablation_config=ablation_config)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
        print("Using checkpoint:", checkpoint)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    test_loss_list = []
    test_acc_list = []
    best_acc = -1.0
    best_model = None
    num_steps = 0
    progress = tqdm(total=total_steps, desc="Training", dynamic_ncols=True, leave=True)

    for img, target in train_loader:
        img = img.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(img)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

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
            test_acc, test_loss = get_accuracy_and_loss(test_loader, model, criterion, game=spec.key)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            tqdm.write(
                f"Num steps: {num_steps}, "
                f"Test Loss: {test_loss_list[-1]:.4f}, "
                f"Test Acc: {test_acc_list[-1]:.3f}"
            )

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = model.state_dict()
                tqdm.write(f"Best model updated: Test Acc: {best_acc:.3f}")

        if num_steps >= total_steps:
            break
    progress.close()

    game_outdir = os.path.join(outdir, spec.key)
    os.makedirs(game_outdir, exist_ok=True)
    file_name = f"{game_outdir}/best_model_position_{best_acc:.3f}_{start_time_string}.pth"
    print("Saving to", file_name)
    torch.save(best_model, file_name)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(test_loss_list, label="Test Loss")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_acc_list, label="Test Acc")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(file_name + ".png", dpi=250)

    return {
        "best_acc": best_acc,
        "test_acc_list": test_acc_list,
        "test_loss_list": test_loss_list,
        "model_path": file_name,
    }
