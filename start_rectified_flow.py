import argparse
import json
import numpy as np
import os
import rectified_flow.models
import torch
from dataclasses import dataclass, field
from PIL import Image
from rectified_flow.data import get_dataloader
from rectified_flow.train import train_1_rectified
from typing import Optional, Dict


@dataclass
class CustomConfig:
    num_train_epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulate_steps: int
    output_path: str
    image_size: int = field(default=64)
    discrete_training_steps: bool = field(default=False)
    sampling_steps: Optional[int] = field(default=None)
    wandb_proj_name: Optional[str] = field(default=None)
    wandb_team_name: Optional[str] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)
    resume_from_checkpoint: Optional[str] = field(default=None)
    scheduler_cls: Optional[str] = None
    scheduler_kwargs: Optional[Dict] = None
    architecture: str = field(default='TimeConditionalUnet')
    architecture_kwargs: Dict = field(default_factory=lambda: {'base_dim': 64})
    mix_unconditional: bool = field(default=False)


parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = vars(CustomConfig(json.load(f)))
output_path = config.pop('output_path')
batch_size = config.pop('batch_size')
resume_from_checkpoint = config.pop('resume_from_checkpoint')
sampling_steps = config.pop('sampling_steps')
image_size = config.pop('image_size')
architecture = config.pop('architecture')
architecture_kwargs = config.pop('architecture_kwargs')
discrete_training_steps = config.pop('discrete_training_steps')


model = getattr(rectified_flow.models, architecture)(**architecture_kwargs)
if resume_from_checkpoint:
    checkpoint = torch.load(resume_from_checkpoint)
    model.load_state_dict(checkpoint)
    model.cuda()
else:
    model.cuda()
    model.initialize()
    print(sum(p.numel() for p in model.parameters()))
    # Do not use sampling step this time
    train_dataloader = get_dataloader(f'noise_cache_{image_size}', image_size=image_size, batch_size=batch_size, shuffle=True, sampling_steps=sampling_steps)
    train_1_rectified(model, train_dataloader, **config)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)


x = np.random.randn(12, 1, image_size, image_size).astype(np.float32)
x = torch.from_numpy(x).cuda()
y = torch.LongTensor([i for i in range(12)]).cuda()
pred = model.generate(x, y, sampling_steps)

for i in range(12):
    image = pred[i, 0]
    image = (image.cpu().numpy() * 255).astype(np.uint8)
    image = Image.fromarray(image, mode='L')
    image.save(f'debug{i}.png')
