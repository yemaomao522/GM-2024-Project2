import numpy as np
import torch
import tqdm
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
        for t in tqdm.tqdm(np.linspace(0, 1, num_time_steps, endpoint=False), desc="Generate"):
            t = torch.ones(x.shape[0]) * t.item()
            t = t.cuda()
            v = self.forward(x, y, t)
            x = torch.clip(x + v * time_step_size, 0, 1)
        return x
