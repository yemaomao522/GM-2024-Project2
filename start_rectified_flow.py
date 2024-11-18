import numpy as np
import torch
from dataclasses import dataclass, field
from PIL import Image
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
    resume_from_checkpoint: Optional[str] = field(default=None)


config = vars(CustomConfig(
    num_train_epochs=100,
    learning_rate=1e-4,
    batch_size=4,
    gradient_accumulate_steps=8,
    output_dir='models/rectified-flow-1.pth',
    wandb_proj_name='GM-2024-Project2',
    wandb_team_name='lumen-team',
    wandb_run_name='flow-no-act',
))
output_dir = config.pop('output_dir')
batch_size = config.pop('batch_size')
resume_from_checkpoint = config.pop('resume_from_checkpoint')


if resume_from_checkpoint:
    model = ConditionalUNet(12)
    checkpoint = torch.load(resume_from_checkpoint)
    model.load_state_dict(checkpoint)
    model.cuda()
else:
    model = ConditionalUNet(12).cuda()
    model.initialize()
    print(sum(p.numel() for p in model.parameters()))
    train_dataloader = get_dataloader('noise_cache', batch_size=batch_size, shuffle=True)
    train_1_rectified(model, train_dataloader, **config)
    torch.save(model.state_dict(), output_dir)


x = ((np.random.randn(12, 1, 256, 256).clip(-1, 1) + 1) / 2).astype(np.float32)
x = torch.from_numpy(x).cuda()
y = torch.LongTensor([i for i in range(12)]).cuda()
pred = model.generate(x, y, 100)

for i in range(12):
    image = pred[i, 0]
    image = (image.cpu().numpy() * 255).astype(np.uint8)
    image = Image.fromarray(image, mode='L')
    image.save(f'debug{i}.png')
