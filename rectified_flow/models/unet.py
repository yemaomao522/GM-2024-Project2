"""
The implementation is partially based on https://github.com/TongTong313/rectified-flow
"""
from torch import nn, Tensor
from .model import VectorField


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_trans = nn.Linear(time_emb_dim, in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x: Tensor, t: Tensor):
        x = x + self.time_trans(t)[:, :, None, None]
        return self.conv(x)


class TimeConditionalUnet(VectorField):
    """Accept a batch of 1x64x64 Tensors and corresponding timestep embeddings t as inputs.
    """
    def construct_upsample(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
    
    def __init__(self, base_dim: int, time_emb_dim: int):
        super().__init__()
        # 1x64x64 -> Dx64x64
        self.in_trans = nn.Conv2d(1, base_dim, kernel_size=3, padding=1)
        # Dx64x64 -> 2Dx32x32
        self.downconv1 = DoubleConv(base_dim, 2 * base_dim, time_emb_dim)
        self.downpool1 = nn.MaxPool2d(2)
        # 2Dx32x32 -> 4Dx16x16
        self.downconv2 = DoubleConv(base_dim * 2, base_dim * 4, time_emb_dim)
        self.downpool2 = nn.MaxPool2d(2)
        # 4Dx16x16 -> 8Dx8x8
        self.downconv3 = DoubleConv(base_dim * 4, base_dim * 8, time_emb_dim)
        self.downpool3 = nn.MaxPool2d(2)
        # 8Dx8x8 -> 8Dx8x8
        self.middle = DoubleConv(base_dim * 8, base_dim * 8, time_emb_dim)
        # 8Dx8x8 -> 8Dx16x16
        self.upsample3 = self.construct_upsample(base_dim * 8, base_dim * 8)
        self.upconv3 = DoubleConv(16 * base_dim, 8 * base_dim, time_emb_dim)
        # 8Dx16x16 -> 4Dx32x32
        self.upsample2 = self.construct_upsample(base_dim * 8, base_dim * 4)
        self.upconv2 = DoubleConv(base_dim * 8, base_dim * 4, time_emb_dim)
        # 4Dx32x32 -> 2Dx64x64
        self.upsample1 = self.construct_upsample(base_dim * 4, base_dim * 2)
        self.upconv1 = DoubleConv(base_dim * 4, base_dim * 2, time_emb_dim)
        # 2Dx64x64 -> 1x64x64
        self.out_trans = nn.Conv2d(base_dim * 2, 1, kernel_size=3, padding=1)
