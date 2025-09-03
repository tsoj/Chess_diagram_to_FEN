# bench/fastpath_wrapper.py
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import torch

from speedkit.io import imread_fast, normalize_inplace_bgr_to_rgb01
from speedkit.fen import fen_from_grid
from speedkit.project_model_loader import load_dense_model, predict_board_dense
from speedkit.orient import choose_best_orientation


def process_board_bgr_dense(img_bgr, device="mps", dense_model=None):
    """
    Dense single-shot pad: hele board (512x512) -> (8,8,13) -> FEN (met oriëntatiecheck).
    """
    # BGR uint8 -> RGB [0,1] float32
    rgb = normalize_inplace_bgr_to_rgb01(img_bgr)
    # Resize naar 512x512 (verwacht door get_dense_model)
    rgb512 = cv2.resize(np.ascontiguousarray(rgb), (512, 512), interpolation=cv2.INTER_AREA)
    # CHW + batch
    x = torch.from_numpy(rgb512.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # logits (8,8,13)
    logits_grid = predict_board_dense(dense_model, x, device=device)
    labels = logits_grid.argmax(dim=-1).cpu().numpy()   # (8,8)
    # oriëntatie + legaliteit
    name, labels_best, fen_rows, score = choose_best_orientation(labels)
    return fen_rows


def main():
    ap = argparse.ArgumentParser(description="Fastpath demo: dense single-shot (512x512)")
    ap.add_argument("--input", nargs="*", help="Pad(den) naar bord-afbeeldingen of mappen (png/jpg/jpeg).")
    ap.add_argument("--dummy", type=int, default=0, help="Aantal dummy-borden genereren i.p.v. echte afbeeldingen.")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"], help="Device voor inference.")
    ap.add_argument("--runs", type=int, default=1, help="Aantal herhalingen voor timing (gemiddelde p50/p90).")
    args = ap.parse_args()

    # Device
    device = args.device if (args.device == "mps" and torch.backends.mps.is_available()) else "cpu"

    # Alleen dense-pad
    dense_model, device = load_dense_model(device=args.device)

    # Input verzamelen
    boards_paths = []
    if args.input:
        for p in args.input:
            path = Path(p)
            if path.is_dir():
                boards_paths += [str(f) for f in path.glob("*.png")]
                boards_paths += [str(f) for f in path.glob("*.jpg")]
                boards_paths += [str(f) for f in path.glob("*.jpeg")]
            elif path.exists():
                boards_paths.append(str(path))
        if not boards_paths:
            raise SystemExit("Geen geldige input-afbeeldingen gevonden met --input.")
    elif args.dummy > 0:
        boards_paths = [f"DUMMY_{i}" for i in range(args.dummy)]
    else:
        raise SystemExit("Geef --input <pad> of --dummy N.")

    # Run
    fens = []
    times = []
    for r in range(args.runs):
        t0 = time.perf_counter()
        for b in boards_paths:
            if isinstance(b, str) and b.startswith("DUMMY_"):
                img_bgr = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
            else:
                img_bgr = imread_fast(b)
            fen = process_board_bgr_dense(img_bgr, device=device, dense_model=dense_model)
            fens.append(fen)
        t1 = time.perf_counter()
        times.append((t1 - t0) / len(boards_paths))

    # Statistieken
    p50 = float(np.median(times) * 1000.0)
    p90 = float(np.percentile(times, 90) * 1000.0)
    bps = 1.0 / (np.median(times) if np.median(times) > 0 else 1e-9)

    print(f"Boards processed: {len(boards_paths)}  |  runs: {args.runs}")
    print(f"p50: {p50:.3f} ms/bord   p90: {p90:.3f} ms/bord   boards/s: {bps:.2f}")
    print("Sample FENs (eerste 3):", fens[:3])


if __name__ == "__main__":
    main()
