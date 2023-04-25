import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np


class SICEDataset(Dataset):
    def __init__(self, dataset_path, img_size):
        img_base = 'train_images/*.png'
        self.img_path = os.path.join(dataset_path, img_base)
        self.transform_image = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5]),
             transforms.Resize((img_size, img_size))
            
            ]
        )
        self.data_img = sorted(glob.glob(self.img_path))
    
    def __len__(self):
        return len(self.data_img)

    def __getitem__(self, index):
        img = PIL.Image.open(self.data_img[index])
        ############################

        img = self.transform_image(img) # -1 to 1
        return img

def get_dataset(dataset_path, img_size, batch_size=10):
    dataset = SICEDataset(dataset_path, img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return dataloader


