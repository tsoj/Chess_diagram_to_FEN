import numpy as np

def crop_tiles_vectorized(board: np.ndarray, grid=8):
    """
    Splitst een bordafbeelding in (grid x grid) tegels met NumPy reshape/tricks.
    Verwacht dat de hoogte en breedte deelbaar zijn door 'grid'.

    Parameters
    ----------
    board : np.ndarray
        Array met vorm (H, W, C), bijvoorbeeld (512, 512, 3).
    grid : int
        Aantal velden per rij/kolom (standaard 8 voor schaak).

    Returns
    -------
    tiles : np.ndarray
        Array van vorm (grid, grid, tile_h, tile_w, C).
    """
    H, W, C = board.shape
    th, tw = H // grid, W // grid
    tmp = board.reshape(grid, th, grid, tw, C)
    tiles = tmp.swapaxes(1, 2)  # (grid, grid, th, tw, C)
    return tiles

def tiles_to_batch(tiles: np.ndarray):
    """
    Zet (8,8,th,tw,3) om naar (64,3,th,tw) voor PyTorch-inferentie.

    Parameters
    ----------
    tiles : np.ndarray
        Array uit crop_tiles_vectorized.

    Returns
    -------
    batch : np.ndarray
        Vorm (64, 3, th, tw), NCHW.
    """
    t = tiles.reshape(-1, *tiles.shape[2:])        # (64, th, tw, 3)
    t = t.transpose(0, 3, 1, 2).copy()             # (64, 3, th, tw)
    return t
