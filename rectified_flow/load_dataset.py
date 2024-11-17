import torch
import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder

def load_dataset():
    # The training images are black-white 3x255x255 images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.nn.functional.pad(x[:1], (0, 1, 0, 1)))
    ])
    data = ImageFolder('data/subclass12', transform=transforms.ToTensor())
