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
from typing import Tuple, Optional


class DatasetForRectifiedFlow(ImageFolder):
    def __init__(self, noise_cache_dir: str, image_size: int, sampling_steps: Optional[int]=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        num_samples = len(self.samples)
        if not self.detect_cache_dir(num_samples, noise_cache_dir):
            self.generate_noises(num_samples, noise_cache_dir)
        self.noise_cache_dir = noise_cache_dir
        self.sampling_steps = sampling_steps
    
    def detect_cache_dir(self, num: int, cache_dir: str) -> bool:
        if not os.path.exists(cache_dir):
            return False
        return all(os.path.exists(os.path.join(cache_dir, f'noise_{i}.png')) for i in range(num))
    
    def generate_noises(self, num: int, cache_dir: str):
        logging.info(f"Generate noise images and save then into {cache_dir}.")
        os.makedirs(cache_dir, exist_ok=True)
        for i in tqdm.tqdm(range(num)):
            noise = (np.random.randn(self.image_size, self.image_size).clip(-1, 1) + 1) / 2
            noise = (noise * 255).astype(np.uint8)
            image = Image.fromarray(noise, mode='L')
            image.save(os.path.join(cache_dir, f'noise_{i}.png'))
    
    def load_noise(self, index: int) -> torch.Tensor:
        image = Image.open(os.path.join(self.noise_cache_dir, f'noise_{index}.png'))
        image_array = np.array(image)[np.newaxis, :, :]
        return torch.from_numpy((image_array / 255)).float()
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float, int, torch.Tensor]:
        sample, target = super().__getitem__(index)
        time_step = random.random() if self.sampling_steps is None else random.randint(0, self.sampling_steps - 1) / self.sampling_steps
        noise = self.load_noise(index)
        input_image = sample * time_step + noise * (1 - time_step)
        return input_image, time_step, target, (sample - noise)


def get_dataloader(noise_cache_dir: str, batch_size: int, image_size: int, shuffle: bool=False, sampling_steps: Optional[int]=None) -> DataLoader:
    """Load the dataloader.
    Args:
        noise_cache_dir (str): the path to the cache of the noise images.
        batch_size (int):
        shuffle (bool, Optional): default to False.
        sampling_steps (int, Optional): default to None; when assigned with an integer, it's the number of sampling steps during generation; it controls the time steps during training.
    """
    # The training images are black-white 3x255x255 images
    assert 256 % image_size == 0, "image_size should be a factor of 256."
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.nn.functional.pad(x[:1], (0, 1, 0, 1))),
        transforms.Lambda(lambda x: torch.nn.functional.max_pool2d(x, kernel_size=256 // image_size))
    ])
    data = DatasetForRectifiedFlow(noise_cache_dir, sampling_steps, 'subclass12', transform=transform)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
