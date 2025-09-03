import numpy as np
from speedkit.tiles import crop_tiles_vectorized, tiles_to_batch

# Maak een dummy bord: 512x512 met 3 kanalen (RGB), gevuld met random getallen
board = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)

tiles = crop_tiles_vectorized(board, grid=8)
print("Tiles shape:", tiles.shape)  # verwacht (8, 8, 64, 64, 3)

batch = tiles_to_batch(tiles)
print("Batch shape:", batch.shape)  # verwacht (64, 3, 64, 64)

# Extra check: reconstructie van eerste tile
tile0 = tiles[0,0]
back0 = batch[0].transpose(1,2,0)
print("Tile0 equal:", np.array_equal(tile0, back0))
