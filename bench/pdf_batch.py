import argparse, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import torch

try:
    import pypdfium2 as pdfium
except Exception:
    pdfium = None

from speedkit.io import imread_fast, normalize_inplace_bgr_to_rgb01
from speedkit.project_model_loader import load_dense_model, predict_board_dense
from speedkit.orient import choose_best_orientation

def process_bgr(img_bgr, model, device="mps"):
    rgb = normalize_inplace_bgr_to_rgb01(img_bgr)
    rgb512 = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb512.transpose(2,0,1)).unsqueeze(0).contiguous()
    logits_grid = predict_board_dense(model, x, device=device)  # (8,8,13)
    labels = logits_grid.argmax(dim=-1).cpu().numpy()
    _, _, fen_rows, _ = choose_best_orientation(labels)
    return fen_rows

def load_images_from_dir(d: Path):
    imgs = []
    for ext in ("*.png","*.jpg","*.jpeg"):
        imgs += list(d.glob(ext))
    return imgs

def load_pages_from_pdf(pdf_path: Path, scale=2.0):
    if pdfium is None:
        raise SystemExit("pypdfium2 niet geïnstalleerd. Doe: pip install pypdfium2")
    pdf = pdfium.PdfDocument(str(pdf_path))
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=scale).to_pil()  # PIL RGB
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        yield img, f"{pdf_path.name}_page_{i+1:03d}"

def main():
    ap = argparse.ArgumentParser(description="Batch diagram->FEN over map of PDF (dense model)")
    ap.add_argument("--input", required=True, help="Pad naar map met afbeeldingen OF naar PDF")
    ap.add_argument("--workers", type=int, default=8, help="Aantal threads voor I/O + verwerking")
    ap.add_argument("--device", default="mps", choices=["mps","cpu"], help="Device")
    args = ap.parse_args()

    device = args.device if (args.device=="mps" and torch.backends.mps.is_available()) else "cpu"
    model, device = load_dense_model(device=args.device)

    src = Path(args.input)
    jobs = []
    t0 = time.perf_counter()

    if src.is_file() and src.suffix.lower() == ".pdf":
        items = list(load_pages_from_pdf(src))
        def gen():
            for img, name in items:
                yield (img, name)
        def process(item):
            img, name = item
            return name, process_bgr(img, model, device=device)
        iterator = gen()
    elif src.is_dir():
        files = load_images_from_dir(src)
        def gen():
            for f in files:
                yield (f,)
        def process(item):
            (f,) = item
            img = imread_fast(str(f))
            return f.name, process_bgr(img, model, device=device)
        iterator = gen()
    else:
        raise SystemExit("Geef een bestaande map met afbeeldingen of een PDF-bestand.")

    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, it): it for it in iterator}
        for fut in as_completed(futs):
            name, fen = fut.result()
            out.append((name, fen))

    dt = time.perf_counter() - t0
    print(f"Done: {len(out)} items in {dt:.2f}s  | avg {1000*dt/len(out):.2f} ms/item")
    out.sort(key=lambda x: x[0])
    for name, fen in out[:10]:
        print(name, "→", fen)

if __name__ == "__main__":
    main()
