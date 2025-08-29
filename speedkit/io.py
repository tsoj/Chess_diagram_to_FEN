from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

def imread_fast(path: str):
    """
    Snelle en robuuste image loader.
    cv2.imread kan soms last hebben van pad-encoding; imdecode met fromfile is robuuster.
    Return: BGR uint8 array (H, W, 3).
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(f"Kan afbeelding niet laden: {path}")
    return img

def load_batch(paths, workers: int = 8):
    """
    Parallel afbeeldingen laden met een thread pool.
    Return: list van BGR uint8 arrays (H, W, 3).
    """
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        imgs = list(ex.map(imread_fast, paths))
    return imgs

def normalize_inplace_bgr_to_rgb01(img_bgr: np.ndarray,
                                   mean=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225)):
    """
    In-place-achtige normalisatie: BGR uint8 -> RGB float32 [0,1], gestandaardiseerd.
    Let op: we retourneren een NIEUWE float32 array met RGB, zodat de input intact blijft.
    """
    rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0  # BGR->RGB + schalen
    rgb[..., 0] = (rgb[..., 0] - mean[0]) / std[0]
    rgb[..., 1] = (rgb[..., 1] - mean[1]) / std[1]
    rgb[..., 2] = (rgb[..., 2] - mean[2]) / std[2]
    return rgb  # (H,W,3) float32 genormaliseerd
