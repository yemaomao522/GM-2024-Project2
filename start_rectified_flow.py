import numpy as np
import os
import torch
from dataclasses import dataclass, field
from PIL import Image
from rectified_flow.models import TimeConditionalUnet
from rectified_flow.data import get_dataloader
from rectified_flow.train import train_1_rectified
from typing import Optional


@dataclass
class CustomConfig:
    num_train_epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulate_steps: int
    output_dir: str
    sampling_steps: Optional[int] = field(default=None)
    wandb_proj_name: Optional[str] = field(default=None)
    wandb_team_name: Optional[str] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)
    resume_from_checkpoint: Optional[str] = field(default=None)


config = vars(CustomConfig(
    num_train_epochs=30,
    learning_rate=1e-3,
    batch_size=16,
    gradient_accumulate_steps=8,
    sampling_steps=100,
    output_dir='models/Normal.pth',
    wandb_proj_name='GM-2024-Project2',
    wandb_team_name='lumen-team',
    wandb_run_name='Normal',
))
output_dir = config.pop('output_dir')
batch_size = config.pop('batch_size')
resume_from_checkpoint = config.pop('resume_from_checkpoint')
sampling_steps = config.pop('sampling_steps')


if resume_from_checkpoint:
    model = TimeConditionalUnet(64)
    checkpoint = torch.load(resume_from_checkpoint)
    model.load_state_dict(checkpoint)
    model.cuda()
else:
    model = TimeConditionalUnet(64).cuda()
    model.initialize()
    print(sum(p.numel() for p in model.parameters()))
    train_dataloader = get_dataloader('noise_cache', image_size=64, batch_size=batch_size, shuffle=True, sampling_steps=sampling_steps)
    train_1_rectified(model, train_dataloader, **config)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    torch.save(model.state_dict(), output_dir)


x = ((np.random.randn(12, 1, 64, 64).clip(-1, 1) + 1) / 2).astype(np.float32)
x = torch.from_numpy(x).cuda()
y = torch.LongTensor([i for i in range(12)]).cuda()
pred = model.generate(x, y, sampling_steps)

for i in range(12):
    image = pred[i, 0]
    image = (image.cpu().numpy() * 255).astype(np.uint8)
    image = Image.fromarray(image, mode='L')
    image.save(f'debug{i}.png')
