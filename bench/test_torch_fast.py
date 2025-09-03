import torch
from speedkit.torch_fast import to_channels_last, infer_amp

class Dummy(torch.nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
    def forward(self, x):
        return self.conv(x).mean(dim=(1,2,3))  # (B,) logits-achtig

m = Dummy().to("mps")
x = torch.randn(8, 3, 64, 64, device="mps")  # batchje tiles
y32 = infer_amp(m, x, use_amp=False)
y16 = infer_amp(m, x, use_amp=True)
print("OK. fp32 shape:", y32.shape, " fp16 shape:", y16.shape)
