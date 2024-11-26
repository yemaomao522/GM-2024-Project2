import torch
from .model import VectorField
from einops.layers.torch import Rearrange
from torch import nn, Tensor, LongTensor



class PatchEmbedding(nn.Module):
    """Patching and adding positional embeddings.
    """
    def __init__(self, hidden_dim: int, img_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=4, stride=4),
            Rearrange("b c h w -> b (h w) c")
        )
        # self.cls_token = nn.Parameter(torch.randn((1, 1, hidden_dim), dtype=torch.float32), requires_grad=True)
        self.pos_embed = nn.Parameter(torch.randn((1, (img_size // 4) ** 2, hidden_dim)))
        self.hidden_dim = hidden_dim

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
        z = self.proj(x)
        # cls_token = self.cls_token.expand((x.shape[0], -1, -1))
        # z = torch.concat((cls_token, z), dim=1)
        pos_embed = self.pos_embed.expand(z.shape)
        t_embed = self.sin_cos_embedding(t, self.hidden_dim)
        y_embed = self.sin_cos_embedding(y + 1, self.hidden_dim)
        return z + pos_embed + t_embed + y_embed


class DecodingLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            Rearrange("b (h w) d -> b h w d", h=4, w=4),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
    
    def forward(self, x: Tensor):
        return self.conv(x)


class TimeConditionalViT(VectorField):
    """Fixed to 4x4 patch; accept 1xHxW images.
    For convenience, H and W should be the same and divided by 4.
    """
    def __init__(self, hidden_dim: int, nhead: int, num_layers: int, img_size: int):
        super().__init__()
        self.patching = PatchEmbedding(hidden_dim, img_size)
        self.encoding = self.construct_encoder(hidden_dim, nhead, hidden_dim * 4, num_layers)
        self.decoding = DecodingLayer(hidden_dim)
    
    def construct_encoder(self, hidden_dim: int, nhead: int, dim_feedforward: int, num_layers: int):
        layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, norm_first=True, batch_first=True)
        return nn.TransformerEncoder(layer, num_layers)
     
    def forward(self, x: Tensor, y: LongTensor, t: Tensor):
        x = self.patching.forward(x, y, t)
        x = self.encoding.forward(x)
        return self.decoding(x)
    
    def initialize(self):
        pass
