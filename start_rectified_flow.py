import torch
from dataclasses import dataclass, field
from rectified_flow.model import ConditionalUNet
from rectified_flow.data import get_dataloader
from rectified_flow.train import train_1_rectified
from torch.utils.data import DataLoader
from typing import Optional


@dataclass
class CustomConfig:
    num_train_epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulate_steps: int
    output_dir: str
    wandb_proj_name: Optional[str] = field(default=None)
    wandb_team_name: Optional[str] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)


config = vars(CustomConfig(
    num_train_epochs=100,
    learning_rate=1e-4,
    batch_size=4,
    gradient_accumulate_steps=8,
    output_dir='models/rectifide-flow-1.pth',
    wandb_proj_name='GM-2024-Project2',
    wandb_team_name='lumen-team',
    wandb_run_name='debug'
))
output_dir = config.pop('output_dir')
batch_size = config.pop('batch_size')


model = ConditionalUNet(12).cuda()
model.initialize()
print(sum(p.numel() for p in model.parameters()))
train_dataloader = get_dataloader('noise_cache', batch_size=batch_size, shuffle=True)
train_1_rectified(model, train_dataloader, **config)
torch.save(model.state_dict(), output_dir)
