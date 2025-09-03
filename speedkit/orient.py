from typing import Tuple, List
import numpy as np
import chess

PIECES = np.array(["", "P","N","B","R","Q","K","p","n","b","r","q","k"])

def fen_from_grid(labels: np.ndarray) -> str:
    rows = []
    for r in labels:
        s, cnt = "", 0
        for x in r:
            if x == 0:
                cnt += 1
            else:
                if cnt: s += str(cnt); cnt = 0
                s += PIECES[x]
        if cnt: s += str(cnt)
        rows.append(s)
    return "/".join(rows)

def transforms(labels: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    Y = []
    Y.append(("id", labels))
    Y.append(("flipV", labels[::-1, :]))
    Y.append(("flipH", labels[:, ::-1]))
    Y.append(("rot180", labels[::-1, ::-1]))
    return Y

def legality_score(fen_rows: str) -> int:
    fen = f"{fen_rows} w - - 0 1"
    try:
        board = chess.Board(fen)
    except Exception:
        return -999
    score = 0
    # 1 witte + 1 zwarte koning
    wk = sum(1 for _, p in board.piece_map().items() if p.symbol() == "K")
    bk = sum(1 for _, p in board.piece_map().items() if p.symbol() == "k")
    if wk == 1 and bk == 1:
        score += 5
    # elke rij precies 8 velden
    for row in fen_rows.split("/"):
        cnt = 0
        for ch in row:
            cnt += int(ch) if ch.isdigit() else 1
        if cnt == 8:
            score += 2
    # geen pionnen op 1e/8e rij
    rows = fen_rows.split("/")
    if not any(ch in "Pp" for ch in rows[0]):
        score += 1
    if not any(ch in "Pp" for ch in rows[-1]):
        score += 1
    return score

def choose_best_orientation(labels: np.ndarray) -> Tuple[str, np.ndarray, str, int]:
    best = ("", None, "", -10**9)
    for name, L in transforms(labels):
        fen_rows = fen_from_grid(L)
        s = legality_score(fen_rows)
        if s > best[3]:
            best = (name, L, fen_rows, s)
    return best
