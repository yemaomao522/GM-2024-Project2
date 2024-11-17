import numpy as np
import torch
from torch import nn
from abc import ABC, abstractmethod


class VectorField(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.LongTensor, t: torch.Tensor):
        """Predict v(x, t | y).
        Args:
            x (Tensor): the images X_t, shape = (batch, channel, height, width).
            y (LongTensor): the labels.
            t (Tensor): the timesteps, t in [0, 1].
        """
        pass
    
    @torch.no_grad()
    def generate(self, x: torch.Tensor, y: torch.LongTensor, num_time_steps: int) -> torch.Tensor:
        """Transfer from noises to target images.
        Args:
            x (Tensor): the noises, in the shape of (batch, channel, height, width).
            y (Tensor): the labels, in the shape of (batch,).
            num_time_steps (int): the number of forward steps.
        Returns:
            The generated images, in the same shape as x.
        """
        time_step_size = 1 / num_time_steps
        for t in np.linspace(0, 1, num_time_steps, endpoint=False):
            t = torch.ones(x.shape[0]) * t.item()
            v = self.forward(x, y, t)
            x = x + v * time_step_size
        return x


class ConditionalUNet(VectorField):
    """Parameterize the vector field as a UNet.
    It expects a batch of 1x256x256 Tensors (images) as the input.
    """    
    def __init__(self, num_classes: int):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, 256 * 256)
        self.encoder1 = self.double_conv_constructor(3, 48)
        self.encoder2 = self.double_conv_constructor(48, 96)
        self.encoder3 = self.double_conv_constructor(96, 192)
        self.encoder4 = self.double_conv_constructor(192, 384)
        self.bottleneck = self.double_conv_constructor(384, 768)
        self.upconv4 = self.upsample_constructor(768, 384)
        self.decoder4 = self.double_conv_constructor(768, 384)
        self.upconv3 = self.upsample_constructor(384, 192)
        self.decoder3 = self.double_conv_constructor(384, 192)
        self.upconv2 = self.upsample_constructor(192, 96)
        self.decoder2 = self.double_conv_constructor(192, 96)
        self.upconv1 = self.upsample_constructor(96, 48)
        self.decoder1 = self.double_conv_constructor(96, 48)
        self.pred_head = nn.Conv2d(48, 1, kernel_size=1)
    
    def upsample_constructor(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def double_conv_constructor(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, y: torch.LongTensor, t: torch.Tensor):
        class_emb = self.class_emb(y).reshape(x.shape)
        time_emb = t[:, None, None, None].expand(x.shape)
        input_tensor = torch.concat([x, class_emb, time_emb], dim=1)
        
        cache1 = self.encoder1(input_tensor)
        cache2 = self.encoder2(nn.functional.max_pool2d(cache1, 2))
        cache3 = self.encoder3(nn.functional.max_pool2d(cache2, 2))
        cache4 = self.encoder4(nn.functional.max_pool2d(cache3, 2))
        
        input_tensor = self.bottleneck(nn.functional.max_pool2d(cache4, 2))
        
        input_tensor = torch.concat([self.upconv4(input_tensor), cache4], dim=1)
        input_tensor = self.decoder4(input_tensor)
        input_tensor = torch.concat([self.upconv3(input_tensor), cache3], dim=1)
        input_tensor = self.decoder3(input_tensor)
        input_tensor = torch.concat([self.upconv2(input_tensor), cache2], dim=1)
        input_tensor = self.decoder2(input_tensor)
        input_tensor = torch.concat([self.upconv1(input_tensor), cache1], dim=1)
        input_tensor = self.decoder1(input_tensor)
        
        return self.pred_head(input_tensor)
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
