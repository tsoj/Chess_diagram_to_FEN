import argparse, time, numpy as np, torch
from pathlib import Path

from speedkit.io import imread_fast, normalize_inplace_bgr_to_rgb01
from speedkit.tiles import crop_tiles_vectorized, tiles_to_batch
from speedkit.torch_fast import batch_tiles_mps
from speedkit.fen import fen_from_grid

# --- DummyModel die een vaste stelling voorspelt: "rnbqkbnr/pppppppp/8/8/8/8/8/8"
ROW0 = np.array([10, 8, 9, 11, 12, 9, 8, 10], dtype=np.int64)  # r n b q k b n r
ROW1 = np.full((8,), 7, dtype=np.int64)                        # p p p p p p p p
REST = np.zeros((6, 8), dtype=np.int64)
GRID_IDS = np.vstack([ROW0, ROW1, REST])                       # (8,8)

class DummyModel(torch.nn.Module):
    def __init__(self, labels_8x8):
        super().__init__()
        self.labels = torch.from_numpy(labels_8x8.reshape(-1))  # (64,)
        self.n_cls = 13
    def forward(self, x):
        B = x.shape[0]
        assert B == 64, f"Expected 64 tiles, got {B}"
        logits = torch.zeros(B, self.n_cls, device=x.device)
        logits += -10.0
        logits[torch.arange(B), self.labels.to(x.device)] = 10.0
        return logits

def process_board_bgr(img_bgr, model, device="mps", use_amp=True):
    board_rgb = normalize_inplace_bgr_to_rgb01(img_bgr)            # (H,W,3) float32 RGB normed
    tiles = crop_tiles_vectorized(board_rgb, grid=8)                # (8,8,64,64,3)
    batch = tiles_to_batch(tiles).astype(np.float32)                # (64,3,64,64)
    logits = batch_tiles_mps(model, batch, device=device, use_amp=use_amp)  # (64,13)
    labels = logits.argmax(dim=-1).view(8,8).cpu().numpy()
    fen = fen_from_grid(labels)
    return fen

def main():
    ap = argparse.ArgumentParser(description="Fastpath demo (AMP + channels_last + vectorized tiles)")
    ap.add_argument("--input", nargs="*", help="Pad(den) naar bord-afbeeldingen (png/jpg).")
    ap.add_argument("--dummy", type=int, default=0, help="Aantal dummy-borden genereren i.p.v. echte afbeeldingen.")
    ap.add_argument("--device", default="mps", choices=["mps","cpu"], help="Device voor inference.")
    ap.add_argument("--runs", type=int, default=1, help="Aantal herhalingen voor timing.")
    ap.add_argument("--amp", action="store_true", help="Gebruik AMP fp16 op MPS.")
    args = ap.parse_args()

    device = args.device if (args.device=="mps" and torch.backends.mps.is_available()) else "cpu"
    model = DummyModel(GRID_IDS).to(device)

    boards = []
    if args.input:
        for p in args.input:
            path = Path(p)
            if path.is_dir():
                boards += [str(f) for f in path.glob("*.png")] + [str(f) for f in path.glob("*.jpg")] + [str(f) for f in path.glob("*.jpeg")]
            elif path.exists():
                boards.append(str(path))
        if not boards:
            raise SystemExit("Geen geldige input-afbeeldingen gevonden.")
    elif args.dummy > 0:
        boards = [f"DUMMY_{i}" for i in range(args.dummy)]
    else:
        raise SystemExit("Geef --input <pad> of --dummy N.")

    fens = []
    times = []
    for r in range(args.runs):
        t0 = time.perf_counter()
        for b in boards:
            if isinstance(b, str) and b.startswith("DUMMY_"):
                img_bgr = (np.random.rand(512,512,3)*255).astype(np.uint8)
            else:
                img_bgr = imread_fast(b)
            fen = process_board_bgr(img_bgr, model, device=device, use_amp=args.amp)
            fens.append(fen)
        t1 = time.perf_counter()
        times.append((t1-t0)/len(boards))

    p50 = np.median(times)*1000
    p90 = np.percentile(times, 90)*1000
    bps = 1.0/np.median(times)
    print(f"Boards processed: {len(boards)}  |  runs: {args.runs}")
    print(f"p50: {p50:.3f} ms/bord   p90: {p90:.3f} ms/bord   boards/s: {bps:.2f}")
    print("Sample FENs (eerste 3):", fens[:3])

if __name__ == "__main__":
    main()
