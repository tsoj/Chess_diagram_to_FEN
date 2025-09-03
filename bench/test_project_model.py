import sys, os, traceback
import numpy as np
import torch

# Zorg dat src/ op de path staat, voor de zekerheid
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print("Using SRC_DIR:", SRC_DIR)

# 1) Probeer de loader
try:
    from speedkit.project_model_loader import load_project_model, predict_tiles
except Exception as e:
    print("❌ Kon loader niet importeren:", e)
    traceback.print_exc()
    raise SystemExit(1)

# 2) Probeer het model te laden
try:
    model, device = load_project_model(device="mps")
    print("✅ Model geladen op device:", device, "→", type(model).__name__)
except Exception as e:
    print("❌ Kon projectmodel niet laden:", e)
    traceback.print_exc()

    # Extra hint: toon wat er in fen_recognition zit
    try:
        import fen_recognition as fr
        print("\nInhoud van fen_recognition module (verkort):")
        names = dir(fr)
        for n in names:
            if not n.startswith("_"):
                print(" -", n)
    except Exception as e2:
        print("Hint: import fen_recognition faalde:", e2)
    raise SystemExit(1)

# 3) Dummy batch maken (64 tiles): (64, 3, h, w)
h = w = 64
dummy = np.random.randn(64, 3, h, w).astype(np.float32)

# 4) Forward pass
try:
    with torch.no_grad():
        logits = predict_tiles(model, dummy, device=device)  # Tensor verwacht
    print("✅ Forward gelukt. Logits shape:", tuple(logits.shape))
except Exception as e:
    print("❌ Forward faalde:", e)
    traceback.print_exc()
    raise SystemExit(1)

# 5) Controle op vorm en waarden
try:
    if logits.ndim != 2:
        print("⚠️ Verwachtte 2D logits (64, n_cls), kreeg", tuple(logits.shape))
    if logits.shape[0] != 64:
        print("⚠️ Eerste dimensie niet 64; model verwacht mogelijk een andere inputvorm.")
    else:
        print("OK: batch-dim is 64.")
    print("n_cls (kolommen) =", logits.shape[1], "(verwacht ~13 voor 12 stukken + leeg)")
except Exception as e:
    print("⚠️ Kon logits niet inspecteren:", e)

print("✅ Test afgerond.")
