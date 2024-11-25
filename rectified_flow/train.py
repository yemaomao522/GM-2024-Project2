import tqdm
import torch
import wandb
from .models import VectorField
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from typing import Optional

def train_1_rectified(model: VectorField, train_dataloader: DataLoader, num_train_epochs: int, learning_rate: float, gradient_accumulate_steps: int, wandb_proj_name: Optional[str]=None, wandb_team_name: Optional[str]=None, wandb_run_name: Optional[str]=None):
    """Randomly matching noises with target images.
    Args:
        model (VectorField): any instance of vector field models.
        train_dataloader (DataLoader): the dataloader of the training dataset.
        num_train_epochs (int): the number of training epochs.
        learning_rate (float): the learning rate for AdamW.
        gradient_accumulate_steps (int): the steps of gradient accumulation.
        wandb_proj_name (str, Optional): the name of the project; set it to None (default) to disable reporting to wandb.
        wandb_team_name (str, Optional): the name of the wandb group.
        wandb_run_name (str, Optional): the name of the wandb run.
    """
    if wandb_proj_name:
        wandb.init(
            project=wandb_proj_name,
            entity=wandb_team_name,
            name=wandb_run_name
        )
    global_step = 0
    real_batch_size = train_dataloader.batch_size * gradient_accumulate_steps
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.8)
    temp_loss = 0
    loss_fn = torch.nn.MSELoss()
    for epoch in tqdm.tqdm(range(num_train_epochs), desc='Epoch'):
        num_steps = len(train_dataloader)
        for i, (x, t, y, v) in enumerate(tqdm.tqdm(train_dataloader, desc='Step')):
            x = x.cuda()
            t = t.cuda().float()
            y = y.cuda()
            v = v.cuda()
            if i % gradient_accumulate_steps == 0:
                optimizer.zero_grad()
                temp_loss = 0
            pred = model.forward(x, y, t)
            loss = loss_fn(pred - v) * (x.shape[0] / real_batch_size)
            temp_loss += loss.detach().cpu().item()
            loss.backward()
            if (i + 1) % gradient_accumulate_steps == 0 or i + 1 == num_steps:
                optimizer.step()
                global_step += 1
                if wandb_team_name:
                    wandb.log({
                        'global_step': global_step,
                        'loss': temp_loss
                    })
        scheduler.step()
    wandb.finish()
