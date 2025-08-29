import os, sys
from pathlib import Path
import torch
import torch.nn.functional as F

# --- Zorg dat src/ importeerbaar is
REPO_ROOT = Path(__file__).resolve().parents[1]   # .../Chess_diagram_to_FEN
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- Importeer projectmodel helpers
try:
    # We willen specifiek de tile-classifier (per-tegel logits)
    from fen_recognition.model import get_tile_model
except Exception as e:
    raise ImportError(
        f"Kon 'get_tile_model' niet importeren uit {SRC_DIR}/fen_recognition/model.py.\n"
        f"Open dat bestand en controleer de naam/signature. Oorspronkelijke fout: {e}"
    )

def _pick_device(preferred: str = "mps") -> str:
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _find_weights() -> Path | None:
    """
    Probeer gewichten te vinden voor de tile-classifier.
    Als jouw repo de gewichten intern laadt in get_tile_model(), is dit niet nodig en
    retourneren we None. Anders zoeken we een paar gangbare namen/locaties.
    """
    candidates = [
        REPO_ROOT / "models"      / "tile_model.pt",
        REPO_ROOT / "models"      / "tile_model.pth",
        REPO_ROOT / "models"      / "fen_recognition.pt",
        REPO_ROOT / "models"      / "fen_recognition.pth",
        REPO_ROOT / "weights"     / "tile_model.pt",
        REPO_ROOT / "weights"     / "tile_model.pth",
        REPO_ROOT / "checkpoints" / "tile_model.pt",
        REPO_ROOT / "checkpoints" / "tile_model.pth",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None  # laat get_tile_model() het desnoods zelf regelen

def load_project_model(device: str = "mps"):
    """
    Laadt het tile-model uit de repo en zet eventueel gewichten.
    """
    dev = _pick_device(device)

    # 1) Bouw het model via de repo-functie
    model = get_tile_model()  # << als er args nodig zijn, passen we dit zo aan

    # 2) Optioneel: laad externe gewichten als die gevonden worden
    weights_path = _find_weights()
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"⚠️ load_state_dict: missing={len(missing)} unexpected={len(unexpected)} | {weights_path.name}")

    model.eval().to(dev)
    print(f"✅ Tile-model geladen op device={dev}" + (f"  | weights='{weights_path.name}'" if weights_path else ""))
    return model, dev




import torch, torch.nn.functional as F
@torch.no_grad()
def predict_board_dense(model: torch.nn.Module, board_rgb_norm_chw: torch.Tensor, device: str = "mps"):
    dev = "mps" if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
    x = board_rgb_norm_chw.to(dev, non_blocking=True)
    y = model(x)                      # (1, 3, 512, 13)
    y = y.mean(dim=1)                 # (1, 512, 13)
    y = y.view(1, 8, 64, 13)          # 512 = 8*64
    y = y.view(1, 8, 8, 8, 13).mean(dim=3)  # 64 -> 8 via blokgemiddelde
    return y[0]                       # (8, 8, 13)
