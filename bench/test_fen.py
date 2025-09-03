import numpy as np
from speedkit.fen import fen_from_grid

# Dummy bord: bovenste rij vol met zwarte stukken (kleine letters), rest leeg
# Mapping: 0=empty, 7..12 = p n b r q k (voorbeeld)
# Hier gebruiken we: 12=k, 11=q, 10=r, 9=b, 8=n, 7=p (let op: jouw uiteindelijke mapping kan anders zijn)
first_row = np.array([10, 8, 9, 11, 12, 9, 8, 10])  # r n b q k b n r (zoals in chess, als voorbeeld)
grid = np.zeros((8,8), dtype=int)
grid[0] = first_row
grid[1] = 7  # pionnenrij (p)
# rest blijft 0 (empty)

fen = fen_from_grid(grid)
print("FEN:", fen)
# Verwachting (met onze mapping): "rnbqkbnr/pppppppp/8/8/8/8/8/8"
# NB: dit klopt als jouw mapping 7..12 = p n b r q k is zoals boven gezet.
