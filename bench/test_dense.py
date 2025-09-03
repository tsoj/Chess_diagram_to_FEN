import numpy as np, torch
from speedkit.project_model_loader import load_dense_model, predict_board_dense

# Dummy 512×512 genormaliseerde tensor (gebruik dezelfde normalisatie als io.normalize_inplace_bgr_to_rgb01)
x = torch.randn(1, 3, 512, 512, dtype=torch.float32)

m, dev = load_dense_model(device="mps")
logits = predict_board_dense(m, x, device=dev)  # (8,8,13)
print("dense logits shape:", logits.shape)
print("argmax sample row:", logits[0].argmax(dim=-1).tolist())
