# bench/fastpath_wrapper.py
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import torch

from speedkit.io import imread_fast, normalize_inplace_bgr_to_rgb01
from speedkit.tiles import crop_tiles_vectorized, tiles_to_batch
from speedkit.torch_fast import batch_tiles_mps
from speedkit.fen import fen_from_grid
from speedkit.project_model_loader import (
    load_project_model,        # tile-model loader
    load_dense_model,          # dense-model loader (single-shot)
    predict_board_dense,       # (1,3,512,512) -> (8,8,13) logits
)

def process_board_bgr(
    img_bgr,
    device="mps",
    use_amp=True,
    dense=False,
    model=None,
    dense_model=None,
):
    """
    Verwerkt één bordafbeelding tot FEN.

    - dense=False  → tiles-pad (snijden -> batch -> model(64,3,h,w))
    - dense=True   → single-shot dense model op 512x512
    """
    if dense:
        # 1) normaliseer (BGR->RGB [0,1] + standaardisatie)
        rgb = normalize_inplace_bgr_to_rgb01(img_bgr)  # (H,W,3) float32
        # 2) resize naar 512x512 (zoals get_dense_model werkt)
        rgb512 = cv2.resize(np.ascontiguousarray(rgb), (512, 512), interpolation=cv2.INTER_AREA)
        # 3) naar torch CHW, batch=1
        x = torch.from_numpy(rgb512.transpose(2, 0, 1)).unsqueeze(0).contiguous()
        # 4) dense predict → (8,8,13)
        logits_grid = predict_board_dense(dense_model, x, device=device)   # Tensor (8,8,13)
        labels = logits_grid.argmax(dim=-1).cpu().numpy()                  # (8,8)
        return fen_from_grid(labels)

    # Tiles-pad (bestaande flow met batching + AMP)
    board_rgb = normalize_inplace_bgr_to_rgb01(img_bgr)      # (H,W,3) float32
    tiles = crop_tiles_vectorized(board_rgb, grid=8)         # (8,8,tH,tW,3)
    batch = tiles_to_batch(tiles).astype(np.float32)         # (64,3,tH,tW)
    logits = batch_tiles_mps(model, batch, device=device, use_amp=use_amp)  # (64, n_cls)
    labels = logits.argmax(dim=-1).view(8, 8).cpu().numpy()
    return fen_from_grid(labels)


def main():
    ap = argparse.ArgumentParser(description="Fastpath demo: tiles (AMP) óf dense single-shot (512x512)")
    ap.add_argument("--input", nargs="*", help="Pad(den) naar bord-afbeeldingen of mappen (png/jpg/jpeg).")
    ap.add_argument("--dummy", type=int, default=0, help="Aantal dummy-borden genereren i.p.v. echte afbeeldingen.")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"], help="Device voor inference.")
    ap.add_argument("--runs", type=int, default=1, help="Aantal herhalingen voor timing (gemiddelde p50/p90).")
    ap.add_argument("--amp", action="store_true", help="Gebruik AMP fp16 (alleen tiles-pad op MPS).")
    ap.add_argument("--dense", action="store_true", help="Gebruik dense single-shot model (512x512) i.p.v. tiles.")
    args = ap.parse_args()

    # Device kiezen
    device = args.device if (args.device == "mps" and torch.backends.mps.is_available()) else "cpu"

    # Model(s) laden
    model = None
    dense_model = None
    if args.dense:
        dense_model, device = load_dense_model(device=args.device)
    else:
        model, device = load_project_model(device=args.device)

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
            fen = process_board_bgr(
                img_bgr,
                device=device,
                use_amp=args.amp,
                dense=args.dense,
                model=model,
                dense_model=dense_model,
            )
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
