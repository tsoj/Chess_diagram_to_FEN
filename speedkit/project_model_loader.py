from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # handig voor evt. ops

# --- project roots & import path ---
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# importeer dense helper uit de repo
from fen_recognition.model import get_dense_model


def _find_dense_fen_weights() -> Path | None:
    """
    Zoek een getraind FEN-checkpoint. Eerst 'models/best_model_fen*.pth',
    anders elk *fen*.pth/pt in models|weights|checkpoints.
    """
    # voorkeurs-pad
    pref = REPO_ROOT / "models"
    if pref.is_dir():
        hit = next(pref.glob("best_model_fen*.pth"), None)
        if hit:
            return hit

    # fallbacks
    for folder in ("models", "weights", "checkpoints"):
        d = REPO_ROOT / folder
        if d.is_dir():
            for pat in ("*fen*.pth", "*fen*.pt"):
                hit = next(d.glob(pat), None)
                if hit:
                    return hit
    return None


def load_dense_model(device: str = "mps"):
    """
    Bouw dense single-shot model en laad FEN-weights indien aanwezig.
    """
    dev = "mps" if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
    model = get_dense_model().eval().to(dev)

    w = _find_dense_fen_weights()
    if w is not None:
        state = torch.load(w, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"⚠️ dense load_state_dict: missing={len(missing)} unexpected={len(unexpected)} | {w.name}")
        print(f"✅ Dense-model geladen op device={dev}  | weights='{w.name}'")
    else:
        print("⚠️ Geen FEN-weights gevonden voor dense-model; je draait met random init.")

    return model, dev


@torch.no_grad()
def predict_board_dense(model: torch.nn.Module, board_rgb_norm_chw: torch.Tensor, device: str = "mps"):
    """
    Input: (1,3,512,512) RGB genormaliseerd.
    Output: logits_grid (8,8,13).
    Model-output is (1,3,512,13): mean over 3 heads, 512=8*64 → (1,8,64,13),
    vervolgens kolommen poolen in blokken van 8 → (1,8,8,13).
    """
    dev = "mps" if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
    x = board_rgb_norm_chw.to(dev, non_blocking=True)
    y = model(x)                      # (1, 3, 512, 13)
    y = y.mean(dim=1)                 # (1, 512, 13)
    y = y.view(1, 8, 64, 13)          # 512 = 8 * 64
    y = y.view(1, 8, 8, 8, 13).mean(dim=3)  # 64 -> 8 via blokgemiddelde
    return y[0]                       # (8, 8, 13)
