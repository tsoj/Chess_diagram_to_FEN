import argparse, time, os
from pathlib import Path

import numpy as np
import torch
import cv2

from speedkit.io import imread_fast, normalize_inplace_bgr_to_rgb01
from speedkit.project_model_loader import load_dense_model, predict_board_dense
from speedkit.orient import choose_best_orientation


def process_board_bgr_dense(img_bgr, device, dense_model):
    """Dense single-shot: (512x512) -> (8,8,13) -> FEN (met oriëntatie/legality)."""
    # BGR uint8 -> RGB [0,1] float32
    rgb = normalize_inplace_bgr_to_rgb01(img_bgr)
    # resize naar 512x512 (verwacht door get_dense_model)
    rgb512 = cv2.resize(np.ascontiguousarray(rgb), (512, 512), interpolation=cv2.INTER_AREA)
    # CHW + batch
    x = torch.from_numpy(rgb512.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    # logits (8,8,13)
    logits_grid = predict_board_dense(dense_model, x, device=device)  # Tensor (8,8,13)
    labels = logits_grid.argmax(dim=-1).cpu().numpy()                 # (8,8)
    # oriëntatie + legality selectie
    _name, _labels_best, fen_rows, _score = choose_best_orientation(labels)
    return fen_rows


def main():
    ap = argparse.ArgumentParser(description="Dense batch runner (non-interactive) over directory")
    ap.add_argument("--input", required=True, help="Directory met png/jpg/jpeg schaakdiagrammen")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"], help="Inference device")
    ap.add_argument("--runs", type=int, default=1, help="Herhalingen voor timing (p50/p90)")
    ap.add_argument("--out", default="bench/out_dense_batch.txt", help="Logbestand voor FEN-uitvoer")
    args = ap.parse_args()

    root = Path(args.input)
    if not root.is_dir():
        raise SystemExit("Geef een map door aan --input")

    # Verzamel afbeeldingen
    items = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        items.extend(root.rglob(ext))
    items = sorted(items)
    if not items:
        raise SystemExit("Geen afbeeldingen gevonden in de map.")

    # Device selecteren
    device = args.device if (args.device == "mps" and torch.backends.mps.is_available()) else "cpu"

    # Dense model laden (1x)
    dense_model, device = load_dense_model(device=args.device)

    # Non-interactive omgeving (voor de zekerheid; we tonen niets)
    os.environ["NOGUI"] = "1"

    # Run(s)
    all_fens = []
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        fens = []
        for p in items:
            img_bgr = imread_fast(str(p))
            fen = process_board_bgr_dense(img_bgr, device=device, dense_model=dense_model)
            fens.append(fen)
        dt = time.perf_counter() - t0
        all_fens = fens  # laatste run bewaren
        times.append(dt / len(items))

    # Statistiek
    p50 = float(np.median(times) * 1000.0)
    p90 = float(np.percentile(times, 90) * 1000.0)
    bps = 1.0 / (np.median(times) if np.median(times) > 0 else 1e-9)

    print(f"Boards processed: {len(items)}  |  runs: {args.runs}")
    print(f"p50: {p50:.3f} ms/bord   p90: {p90:.3f} ms/bord   boards/s: {bps:.2f}")
    print("Sample FENs (eerste 5):", all_fens[:5])

    # Log schrijven
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p, fen in zip(items, all_fens):
            f.write(f"{p.name}\t{fen}\n")
    print(f"→ Volledige FEN-uitvoer gelogd in: {args.out}")


if __name__ == "__main__":
    main()
