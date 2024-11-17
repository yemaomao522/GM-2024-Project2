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


class ConditionalUNet(VectorField):
    """Parameterize the vector field as a UNet.
    It expects a batch of 1x256x256 Tensors (images) as the input.
    """    
    def __init__(self, num_classes: int):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, 256 * 256)
        self.encoder1 = self.double_conv_constructor(3, 32)
        self.encoder2 = self.double_conv_constructor(32, 64)
        self.encoder3 = self.double_conv_constructor(64, 128)
        self.encoder4 = self.double_conv_constructor(128, 256)
        self.bottleneck = self.double_conv_constructor(256, 512)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self.double_conv_constructor(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self.double_conv_constructor(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self.double_conv_constructor(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = self.double_conv_constructor(64, 32)
        self.pred_head = nn.Conv2d(32, 1, kernel_size=1)
    
    def double_conv_constructor(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
