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
import json
import numpy as np


class SICEDataset(Dataset):
    def __init__(self, dataset_path, img_size, img_base='train_test_split.json', mode='train'):
        # img_base = 'train_test_split.json'
        self.img_path = os.path.join(dataset_path, img_base)
        self.img_size = img_size
        if img_size == -1:
            self.transform_image = transforms.Compose(
                [transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.transform_image = transforms.Compose(
                [transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
                transforms.Resize((img_size, img_size), antialias=True)
                
                ]
            )
        with open(self.img_path, 'r') as openfile:
            json_object = json.load(openfile)
        
        self.data_img = json_object[mode]
    
    def __len__(self):
        return len(self.data_img)

    def __getitem__(self, index):
        img = PIL.Image.open(self.data_img[index])
        img = img.resize((self.img_size, self.img_size), PIL.Image.ANTIALIAS)
        img = np.asarray(img)/255.0
        img = torch.from_numpy(img).float()
        ############################
        # print(img)
        # img = self.transform_image(img) # -1 to 1

        return img.permute(2, 0, 1)

def get_dataset(dataset_path, img_size, batch_size=10, img_base='train_test_split.json', mode='train'):
    dataset = SICEDataset(dataset_path, img_size, img_base, mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return dataloader


