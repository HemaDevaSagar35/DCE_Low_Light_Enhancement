import argparse
import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from network import EnhancerModel
from losses import *
import datasets
import glob 
import PIL
import json

from torchvision.utils import save_image

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.transforms as transforms
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def load(dir):
    model = EnhancerModel.DceNet().to(DEVICE)
    model.load_state_dict(torch.load(dir))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--infer_dataset", type=str, default='data/train_test_split.json')
    parser.add_argument("--model_dir", type=str, default = 'models/enhancer_model_v_10.pth')
    parser.add_argument("--inference_output", type=str, default='inference_outputs/full_run/')

    opt = parser.parse_args()

    infer_dataset = opt.infer_dataset
    model_dir = opt.model_dir
    inference_output = opt.inference_output
    # files = glob.glob(infer_dataset + '/*.jpg')
    # dataloader = datasets.get_dataset(infer_dataset, 512, 1, 'train_test_split.json', 'test')
    model = load(model_dir)
    with open(infer_dataset, 'r') as openfile:
        json_object = json.load(openfile)
        
    all_file = json_object['test']
    with torch.no_grad():
        for img_file in tqdm(all_file):
            img = PIL.Image.open(img_file)
            img = img.resize((512, 512), PIL.Image.ANTIALIAS)
            img = np.asarray(img)/255.0
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1).unsqueeze(0)

            img = img.to(DEVICE)
            _, enhanced_img = model(img)
            # print(torch.max(enhanced_img))
            # print(torch.min(enhanced_img))
            print('_'.join(img_file.split('\\')[-2:]))
            save_image(enhanced_img[0], os.path.join(inference_output, '_'.join(img_file.split('\\')[-2:])),normalize=True, range=(-1, 1))
            # i += 1
