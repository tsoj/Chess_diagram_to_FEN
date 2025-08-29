import torch

def to_channels_last(x: torch.Tensor) -> torch.Tensor:
    """Zet tensor naar channels_last layout (NHWC), ideaal voor MPS."""
    return x.contiguous(memory_format=torch.channels_last)

@torch.no_grad()
def infer_amp(model: torch.nn.Module, x: torch.Tensor, use_amp: bool = True) -> torch.Tensor:
    """
    Inference helper met channels_last + (optioneel) AMP fp16 op MPS.
    - model: PyTorch model (al op 'mps' geplaatst).
    - x: (B, C, H, W) tensor.
    """
    model.eval()
    x = to_channels_last(x)
    if use_amp:
        with torch.autocast(device_type="mps", dtype=torch.float16):
            return model(x)
    return model(x)

def batch_tiles_mps(model: torch.nn.Module, tiles_nchw, device: str = "mps", use_amp: bool = True):
    """
    Voert één gebatchte forward-pass uit over 64 tiles.
    - tiles_nchw: NumPy array of Tensor met vorm (64, 3, h, w), float32.
    Return: logits (Tensor) op device.
    """
    if not torch.is_tensor(tiles_nchw):
        x = torch.from_numpy(tiles_nchw)
    else:
        x = tiles_nchw
    x = x.to(device, non_blocking=True)
    return infer_amp(model, x, use_amp=use_amp)
