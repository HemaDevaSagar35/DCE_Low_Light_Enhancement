
import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class DceNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='same')  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same')
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same')
        self.conv7 = nn.Conv2d(64, 24, kernel_size=3, stride=1, padding='same')
    
    def enhance(self, alphas, x):
        output = x 
        for i in range(alphas.shape[1] // 3):
            output = output + alphas[:,(3*i):(3*(i+1)),:,:]*output*(1 - output)
        return output

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))

        sym_cat_one = torch.cat([x4, x3], axis=1)
        x5 = F.relu(self.conv5(sym_cat_one))

        sym_cat_two = torch.cat([x5, x2], axis=1)
        x6 = F.relu(self.conv6(sym_cat_two))

        sym_cat_three = torch.cat([x6, x1], axis=1)
        x7 = F.tanh(self.conv7(sym_cat_three))

        enhanced_image = self.enhance(x7, x)
        return x7, enhanced_image

    

       
        