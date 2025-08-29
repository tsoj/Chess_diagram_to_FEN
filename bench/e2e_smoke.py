import numpy as np
import torch

from speedkit.io import normalize_inplace_bgr_to_rgb01
from speedkit.tiles import crop_tiles_vectorized, tiles_to_batch
from speedkit.torch_fast import batch_tiles_mps
from speedkit.fen import fen_from_grid

# --- Dummy invoerbord: willekeurige pixels (we testen alleen de pijplijn, niet echte herkenning)
# Maak een 512x512 "BGR" beeld zoals cv2.imread zou doen.
board_bgr = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)

# 1) Normaliseren (BGR -> RGB [0,1] -> gestandaardiseerd)
board_rgb = normalize_inplace_bgr_to_rgb01(board_bgr)

# 2) 64 tiles snijden (vectorized) en NCHW batch maken
tiles = crop_tiles_vectorized(board_rgb, grid=8)               # (8,8,64,64,3)
batch = tiles_to_batch(tiles).astype(np.float32)               # (64,3,64,64)

# 3) Dummy model dat een bekende stelling "faket", zodat FEN voorspelbaar is.
#    We willen als output: "rnbqkbnr/pppppppp/8/8/8/8/8/8"
#    Mapping die we in speedkit.fen gebruikten:
#    0=empty, 1..6 = P N B R Q K, 7..12 = p n b r q k
row0_ids = np.array([10, 8, 9, 11, 12, 9, 8, 10], dtype=np.int64)  # r n b q k b n r
row1_ids = np.full((8,), 7, dtype=np.int64)                       # p p p p p p p p
rest    = np.zeros((6, 8), dtype=np.int64)                        # empties
grid_ids = np.vstack([row0_ids, row1_ids, rest])                  # (8,8)

# We maken een model dat de logits zo teruggeeft dat argmax == grid_ids.flatten()
class DummyModel(torch.nn.Module):
    def __init__(self, labels_8x8):
        super().__init__()
        self.labels = torch.from_numpy(labels_8x8.reshape(-1))  # (64,)
        self.n_cls = 13
    def forward(self, x):
        B = x.shape[0]                # verwacht 64
        assert B == 64, f"Expected 64 tiles, got {B}"
        logits = torch.zeros(B, self.n_cls, device=x.device)
        logits += -10.0               # lage score overal
        logits[torch.arange(B), self.labels.to(x.device)] = 10.0  # hoge score op juiste klasse
        return logits

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = DummyModel(grid_ids).to(device)

# 4) Eén gebatchte forward (AMP + channels_last via helper)
logits = batch_tiles_mps(model, batch, device=device, use_amp=True)  # (64,13)

# 5) Labels -> (8,8) -> FEN
pred = logits.argmax(dim=-1).view(8,8).cpu().numpy()
fen = fen_from_grid(pred)
print("Pred FEN:", fen)
# Verwacht: rnbqkbnr/pppppppp/8/8/8/8/8/8
