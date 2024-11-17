import logging
import numpy as np
import os
import random
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from typing import Tuple


class DatasetForRectifiedFlow(ImageFolder):
    def __init__(self, noise_cache_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_samples = len(self.samples)
        if not self.detect_cache_dir(num_samples, noise_cache_dir):
            self.generate_noises(num_samples, noise_cache_dir)
        self.noise_cache_dir = noise_cache_dir
    
    def detect_cache_dir(self, num: int, cache_dir: str) -> bool:
        if not os.path.exists(cache_dir):
            return False
        return all(os.path.exists(os.path.join(cache_dir, f'noise_{i}.png')) for i in range(num))
    
    def generate_noises(self, num: int, cache_dir: str):
        logging.info(f"Generate noise images and save then into {cache_dir}.")
        os.makedirs(cache_dir, exist_ok=True)
        for i in tqdm.tqdm(range(num)):
            noise = (np.random.randn(256, 256).clip(-1, 1) + 1) / 2
            noise = (noise * 255).astype(np.uint8)
            image = Image.fromarray(noise, mode='L')
            image.save(os.path.join(cache_dir, f'noise_{i}.png'))
    
    def load_noise(self, index: int) -> torch.Tensor:
        image = Image.open(os.path.join(self.noise_cache_dir, f'noise_{index}.png'))
        image_array = np.array(image)[np.newaxis, :, :]
        return torch.from_numpy(image_array / 255)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float, int, torch.Tensor]:
        sample, target = super().__getitem__(index)
        time_step = random.random()
        noise = self.load_noise(index)
        input_image = sample * time_step + noise * (1 - time_step)
        return input_image, time_step, target, (sample - noise)


def get_dataloader():
    # The training images are black-white 3x255x255 images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.nn.functional.pad(x[:1], (0, 1, 0, 1)))
    ])
    data = DatasetForRectifiedFlow('noise_cache', 'subclass12', transform=transform)
    dataloader = DataLoader(data, batch_size=2)
