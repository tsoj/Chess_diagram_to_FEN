import sys, os
import torch

# Zorg dat 'src/' importeerbaar is voor: from fen_recognition import model
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Probeer het projectmodel te importeren
try:
    from fen_recognition import model as project_model
except Exception as e:
    raise ImportError(
        f"Kon projectmodel niet importeren uit {_SRC_DIR}. "
        f"Bestaat 'src/fen_recognition/model.py'? Oorspronkelijke fout: {e}"
    )

def load_project_model(device: str = "mps"):
    """
    Laadt het projectmodel en zet op het gewenste device.
    Pas zo nodig aan op basis van de echte initialisatie/gewichten in jouw repo.
    """
    # ⬇️ VUL DIT AAN ALS NODIG:
    # Als de repo een fabrieksfunctie of klassenaam heeft, gebruik die hier.
    # Voorbeeld (fictief): m = project_model.load_model(weights_path="weights.pth")
    # Of: m = project_model.BoardTileModel() ; m.load_state_dict(torch.load("..."))
    try:
        # Harde, generieke fallback: kijk of er een "build_model" of "BoardTileModel" is
        if hasattr(project_model, "build_model"):
            m = project_model.build_model()
        elif hasattr(project_model, "BoardTileModel"):
            m = project_model.BoardTileModel()
        else:
            # Laatste redmiddel: we weten de class/fabriek niet → geef duidelijke fout
            raise AttributeError("Geen build_model/BoardTileModel gevonden in projectmodel.")
    except Exception as e:
        raise RuntimeError(
            "Pas load_project_model() aan met de juiste initialisatie voor jouw repo.\n"
            f"Zie src/fen_recognition/model.py voor de correcte API. Oorspronkelijk: {e}"
        )

    # Als er gewichten nodig zijn, laad ze hier (pas pad aan):
    # weights_path = os.path.join(_REPO_ROOT, "weights", "model.pth")
    # m.load_state_dict(torch.load(weights_path, map_location="cpu"))

    dev = device if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
    m = m.to(dev).eval()
    return m, dev

@torch.no_grad()
def predict_tiles(model: torch.nn.Module, tiles_nchw, device: str = "mps"):
    """
    Uniforme voorspel-functie:
    tiles_nchw: Tensor/np.ndarray met vorm (64,3,h,w), float32.
    Return: logits Tensor (64, 13).
    Pas dit desnoods aan als jouw model een andere interface heeft.
    """
    if not torch.is_tensor(tiles_nchw):
        x = torch.from_numpy(tiles_nchw)
    else:
        x = tiles_nchw
    dev = device if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
    x = x.to(dev, non_blocking=True)

    # Probeer een rechttoe-rechtaan forward (veel modellen geven (N,13) terug):
    y = model(x)
    # Als jouw repo een tuple of dict teruggeeft, pak hier de juiste sleutel uit.
    # Voorbeeld: y = y["logits"] of y = y[0]
    return y
