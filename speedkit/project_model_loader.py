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

@torch.no_grad()
def predict_tiles(model: torch.nn.Module, tiles_nchw, device: str = "mps"):
    """
    Uniforme predict: (64,3,h,w) -> (64, n_cls) logits.
    Pas dit aan als jouw get_tile_model iets anders verwacht/teruggeeft.
    """
    if not torch.is_tensor(tiles_nchw):
        x = torch.from_numpy(tiles_nchw)
    else:
        x = tiles_nchw
    dev = _pick_device(device)
    x = x.to(dev, non_blocking=True)
    y = model(x)
    return y



def _find_dense_fen_weights() -> Path | None:
    # prefer an explicit “fen” model in models/
    for p in (REPO_ROOT / "models").glob("best_model_fen*.pth"):
        return p
    # fallbacks if someone renamed things
    for folder in ["models", "weights", "checkpoints"]:
        d = REPO_ROOT / folder
        if d.is_dir():
            for pat in ("*fen*.pth", "*fen*.pt"):
                m = next(d.glob(pat), None)
                if m:
                    return m
    return None

def load_dense_model(device: str = "mps"):
    from fen_recognition.model import get_dense_model
    dev = "mps" if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
    m = get_dense_model().eval().to(dev)

    w = _find_dense_fen_weights()
    if w is not None:
        state = torch.load(w, map_location="cpu")
        # Some repos wrap state under "model"/"state_dict"
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"⚠️ dense load_state_dict: missing={len(missing)} unexpected={len(unexpected)}  | {w.name}")
        print(f"✅ Dense-model geladen op device={dev}  | weights='{w.name}'")
    else:
        print(f"⚠️ Geen FEN-weights gevonden voor dense-model; draai je nu met random init.")

    return m, dev


@torch.no_grad()
def predict_board_dense(model: torch.nn.Module, board_rgb_norm_chw: torch.Tensor, device: str = "mps"):
    """
    board_rgb_norm_chw: Tensor (1,3,H,W), reeds RGB genormaliseerd naar [0,1] + std/mean.
    Verwacht H=W=512 voor dit model.
    Return: logits_grid (8, 8, 13) — ruwe 13-klassen logits per cel.
    """
    dev = "mps" if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
    x = board_rgb_norm_chw.to(dev, non_blocking=True)
    y = model(x)                     # (1, 3, 512, 13)

    # 1) heads middelen
    y = y.mean(dim=1)                # (1, 512, 13)

    # 2) 512 = 8 * 64 → per rij 64 posities (kolomresolutie)
    y = y.view(1, 8, 64, 13)         # (1, 8, 64, 13)

    # 3) kolommen “tegelgemiddeld”: 64 → 8 via blokken van 8
    #    (elke schaaktegel ≈ 8 kolompixels breed op deze schaal)
    y = y.view(1, 8, 8, 8, 13).mean(dim=3)   # (1, 8, 8, 13)

    return y[0]  # (8, 8, 13)


