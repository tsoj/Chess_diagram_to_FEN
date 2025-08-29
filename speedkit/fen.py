import numpy as np

# Mapping: 0 = empty, daarna hoofdletters = wit, kleine letters = zwart
PIECES = np.array(["", "P","N","B","R","Q","K","p","n","b","r","q","k"], dtype=object)

def fen_from_grid(lbl8x8: np.ndarray) -> str:
    """
    Zet een (8,8) raster met label-id's om naar een FEN-reeks (alleen het stuk/empty deel).
    lbl8x8: np.ndarray van shape (8,8), ints in 0..12 (0=empty).
    """
    rows = []
    for r in lbl8x8:
        s, cnt = "", 0
        for x in r:
            if x == 0:
                cnt += 1
            else:
                if cnt:
                    s += str(cnt)
                    cnt = 0
                s += PIECES[x]
        if cnt:
            s += str(cnt)
        rows.append(s)
    return "/".join(rows)
