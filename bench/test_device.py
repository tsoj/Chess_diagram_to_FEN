# bench/test_device.py
import torch
from speedkit.project_model_loader import load_dense_model, predict_board_dense
import numpy as np

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, dev = load_dense_model(device=device)
    print(f"✅ Dense-model geladen op: {dev}")

    # Dummy 512x512 board (RGB [0,1])
    x = torch.rand(1, 3, 512, 512, device=dev)

    with torch.no_grad():
        logits_grid = predict_board_dense(model, x, device=dev)  # (8,8,13)

    print("✅ Forward gelukt, output shape:", tuple(logits_grid.shape))

    # Kleine sanity check: FEN-achtige labels (alleen om vorm te checken)
    labels = logits_grid.argmax(dim=-1).cpu().numpy()  # (8,8)
    print("Labels shape:", labels.shape, "| uniekele klassen:", np.unique(labels)[:10], "...")

if __name__ == "__main__":
    main()
