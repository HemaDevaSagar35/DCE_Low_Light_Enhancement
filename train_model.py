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
import PIL

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.utils import save_image
from datetime import datetime

def train(opt):
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = SummaryWriter(os.path.join(opt.output_dir, 'logs'))
    if device == 'cpu':
        print("cpu is the device")
        return 
    # device='cpu'
    model = EnhancerModel.DceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0.0001,
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
    steps = 0
    
                
    for ep in range(1, opt.n_epochs+1):
        total_progress_bar.update(1)
        losses = []
        losses_spa = []
        losses_exp = []
        losses_col = []
        losses_ill = []

        # start_s = datetime.now()
        for imgs in tqdm(dataloader):
        # for i, imgs in enumerate(dataloader):
            end_s = datetime.now()
            # print("time take is {}".format((end_s - start_s).total_seconds()))

            start = datetime.now()
            imgs = imgs.to(device).float()
            maps, enhanced_image = model(imgs)
            end = datetime.now()
            # print("time take for forward prediction {}".format((end - start).total_seconds()))
            
            start = datetime.now()
            loss_spt = spatial_consistency_loss(imgs, enhanced_image, local_region=opt.spt_local_region)
            end = datetime.now()
            # print("time take for spatial_consistency_loss {}".format((end - start).total_seconds()))

            start = datetime.now()
            loss_exp = exposure_control_loss(enhanced_image, E = opt.E, local_region = opt.exp_local_region)
            end = datetime.now()
            # print("time take for exposure_control_loss {}".format((end - start).total_seconds()))

            start = datetime.now()
            loss_col = color_consistency_loss(enhanced_image)
            end = datetime.now()
            # print("time take for color_consistency_loss {}".format((end - start).total_seconds()))

            start = datetime.now()
            loss_ill = illumination_smoothness_loss(maps)
            end = datetime.now()
            # print("time take for illumination_smoothness_loss {}".format((end - start).total_seconds()))

            start = datetime.now()
            total_loss = opt.spa_weight*loss_spt + opt.exp_weight*loss_exp + opt.color_weight*loss_col + opt.ill_weight*loss_ill

            logger.add_scalar('spatial consistency loss', loss_spt.item(), steps)
            logger.add_scalar('exposure control loss', loss_exp.item(), steps)
            logger.add_scalar('color consistency loss', loss_col.item(), steps)
            logger.add_scalar('illumination smoothness loss', loss_ill.item(), steps)
            logger.add_scalar('total loss', total_loss.item(), steps)

            losses.append(total_loss.item())

            losses_spa.append(loss_spt.item())
            losses_exp.append(loss_exp.item())
            losses_col.append(loss_col.item())
            losses_ill.append(loss_ill.item())
            end = datetime.now()
            # print("time take for logging {}".format((end - start).total_seconds()))
            
            start = datetime.now()
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            end = datetime.now()
            # print("time take for grad backward {}".format((end - start).total_seconds()))

            steps += 1
            start_s = datetime.now()
        
        tqdm.write(f"[Epoch : {ep+1}/{opt.n_epochs}][total_loss : {np.mean(losses)}] [losses_spa : {np.mean(losses_spa)}] [losses_exp : {np.mean(losses_exp)}] [losses_col : {np.mean(losses_col)}] [losses_ill : {np.mean(losses_ill)}]")
        # print("saving test image")
        # torch.cuda.empty_cache()
        # test_image = PIL.Image.open('D:\\spring2023\\748\\project\\data\\Dataset_Part1\\Dataset_Part1\\318\\1.JPG')
        # test_image = transforms.ToTensor()(test_image).to(device)
        # with torch.no_grad():
        #     _, en_img = model(test_image)
        #     save_image(en_img, 'output_{}.jpg'.format(ep))
        if ep % 5 == 0:
            torch.save(model.state_dict(), opt.output_dir + '/enhancer_model_v_{}.pth'.format(ep))
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), opt.output_dir + '/enhancer_model_final.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--dataset_path", type=str, default = 'data/')
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--spt_local_region", type=int, default=4)
    parser.add_argument("--exp_local_region", type=int, default=16)
    parser.add_argument("--exp_weight", type=int, default=10)
    parser.add_argument("--color_weight", type=int, default=5)
    parser.add_argument("--ill_weight", type=int, default=200)
    parser.add_argument("--spa_weight", type=int, default=1)
    parser.add_argument("--E", type=int, default=0.6)
    parser.add_argument("--output_dir", type=str, default = 'models/')

    opt = parser.parse_args()
    print(opt)
    start = datetime.now()
    train(opt)
    end = datetime.now()
    print("total time take is {}".format((end - start).total_seconds()/60.0))
    




    



