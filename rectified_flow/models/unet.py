"""
The implementation is partially based on https://github.com/TongTong313/rectified-flow
"""
import torch
from torch import nn, Tensor, LongTensor
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def __init__(self, base_dim: int):
        super().__init__()
        assert base_dim % 2 == 0, "base_dim should be an even number."
        self.time_emb_dim = base_dim
        # 1x64x64 -> Dx64x64
        self.in_trans = nn.Conv2d(1, base_dim, kernel_size=3, padding=1)
        # Dx64x64 -> 2Dx32x32
        self.downconv1 = DoubleConv(base_dim, 2 * base_dim, self.time_emb_dim)
        self.downpool1 = nn.MaxPool2d(2)
        # 2Dx32x32 -> 4Dx16x16
        self.downconv2 = DoubleConv(base_dim * 2, base_dim * 4, self.time_emb_dim)
        self.downpool2 = nn.MaxPool2d(2)
        # 4Dx16x16 -> 8Dx8x8
        self.downconv3 = DoubleConv(base_dim * 4, base_dim * 8, self.time_emb_dim)
        self.downpool3 = nn.MaxPool2d(2)
        # 8Dx8x8 -> 8Dx8x8
        self.middle = DoubleConv(base_dim * 8, base_dim * 8, self.time_emb_dim)
        # 8Dx8x8 -> 8Dx16x16
        self.upsample3 = self.construct_upsample(base_dim * 8, base_dim * 8)
        self.upconv3 = DoubleConv(16 * base_dim, 8 * base_dim, self.time_emb_dim)
        # 8Dx16x16 -> 4Dx32x32
        self.upsample2 = self.construct_upsample(base_dim * 8, base_dim * 4)
        self.upconv2 = DoubleConv(base_dim * 8, base_dim * 4, self.time_emb_dim)
        # 4Dx32x32 -> 2Dx64x64
        self.upsample1 = self.construct_upsample(base_dim * 4, base_dim * 2)
        self.upconv1 = DoubleConv(base_dim * 4, base_dim * 2, self.time_emb_dim)
        # 2Dx64x64 -> 1x64x64
        self.out_trans = nn.Conv2d(base_dim * 2, 1, kernel_size=3, padding=1)
    
    def sin_cos_embedding(self, x: Tensor, dim: int):
        """Generate sin-cos embedding for timesteps.
        Args:
            x (Tensor): a batch of timesteps or labels.
        """
        x = (x * 1000).float()
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(x.device)
        sin_emb = torch.sin(x[:, None] / freqs)
        cos_emb = torch.cos(x[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)
    
    def forward(self, x: Tensor, y: LongTensor, t: Tensor):
        x = self.in_trans(x)
        # contiditonal
        y_emb = self.sin_cos_embedding(y, self.time_emb_dim)
        y_emb[y==-1] = 0
        time_emb = self.sin_cos_embedding(t, self.time_emb_dim)
        time_emb += y_emb
        # down
        x1 = self.downconv1(x, time_emb)
        x2 = self.downconv2(self.downpool1(x1), time_emb)
        x3 = self.downconv3(self.downpool2(x2), time_emb)
        x = self.middle(self.downpool3(x3), time_emb)
        # up
        x = self.upconv3(torch.concat((self.upsample3(x), x3), dim=1), time_emb)
        x = self.upconv2(torch.concat((self.upsample2(x), x2), dim=1), time_emb)
        x = self.upconv1(torch.concat((self.upsample1(x), x1), dim=1), time_emb)
        return self.out_trans(x)
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
