import argparse
import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import EnhancerModel
from losses import *
import datasets

from tqdm import tqdm

def train(opt):
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = EnhancerModel.DceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = 1e-4)

    dataloader = datasets.get_dataset(
        dataset_path = opt.dataset_path, 
        img_size = opt.img_size, 
        batch_size=opt.batch_size
    )

    total_progress_bar = tqdm(
        total = opt.n_epochs,
        desc = 'Total progress',
        dynamic_ncols = True
    )
    for ep in range(opt.n_epochs):
        total_progress_bar.update(1)
        losses = []
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device).float()
            maps, enhanced_image = model(imgs)

            loss_spt = spatial_consistency_loss(imgs, enhanced_image, local_region=opt.spt_local_region)
            loss_exp = exposure_control_loss(enhanced_image, E = opt.E, local_region = opt.exp_local_region)
            loss_col = color_consistency_loss(enhanced_image)
            loss_ill = illumination_smoothness_loss(maps)

            total_loss = loss_spt + loss_exp + opt.color_weight*loss_col + opt.ill_weight*loss_ill
            losses.append(total_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        tqdm.write(f"[Epoch : {ep+1}/{opt.n_epochs}][total_loss : {np.mean(losses)}]")

    torch.save(model.state_dict(), opt.output_dir + '/enhancer_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--npochs", type=int, default=60)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--spt_local_region", type=int, default=4)
    parser.add_argument("--exp_local_region", type=int, default=16)
    parser.add_argument("--color_weight", type=int, default=0.5)
    parser.add_argument("--ill_weight", type=int, default=20)
    parser.add_argument("--output_dir", type=str)

    opt = parser.parse_args()
    print(opt)
    train(opt)
    




    



