import math
import torch
from torch import autograd
from torch.nn import functional as F

def spatial_consistency_loss(real_image, enhanced_image, local_region = 4):
   #TODO: need to be tested.
    with torch.cuda.amp.autocast():
        
        real_image_avg = F.avg_pool2d(real_image, local_region, stride=(local_region, local_region), padding=0, ceil_mode=True)
        enhanced_image_avg = F.avg_pool2d(enhanced_image, local_region, stride=(local_region, local_region), padding=0, ceil_mode=True)

        #i - (i+1) center - right
        # # x N x c x img x 2
        right_kernel = torch.ones(1, 2, device=real_image.device)
        right_kernel[:,1] = -1.0
        right_kernel = right_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        real_x = real_image_avg.unfold(-1, 2, 1)
        enhance_x = enhanced_image_avg.unfold(-1, 2, 1)
        # print("real avg is ", real_image_avg)
        # print("enhance avg is ", enhanced_image_avg)

        real_right_diff = torch.abs((real_x * right_kernel).sum(axis=-1))
        enhance_right_diff = torch.abs((enhance_x * right_kernel).sum(axis=-1))
        # print("real right diff is \n ", real_right_diff)
        # print("enhance right diff is \n ", enhance_right_diff)
        right_part = ((enhance_right_diff - real_right_diff)**2).sum()

        #i - (i - 1) center - left
        left_kernel = torch.ones(1, 2, device=real_image.device)
        left_kernel[:,0] = -1.0
        left_kernel = left_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        real_left_diff = torch.abs((real_x * left_kernel).sum(axis=-1))
        enhance_left_diff = torch.abs((enhance_x * left_kernel).sum(axis=-1))

        left_part = ((enhance_left_diff - real_left_diff)**2).sum()
        print((enhance_left_diff - real_left_diff)**2)
        print(left_part)
        #j - j + 1 center - down
        down_kernel = torch.ones(2, 1, device=real_image.device)
        down_kernel[1,:] = -1.0
        down_kernel = down_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        real_y = real_image_avg.unfold(-2, 2, 1)
        enhance_y = enhanced_image_avg.unfold(-2, 2, 1)

        real_down_diff = torch.abs((real_y * down_kernel).sum(axis=-2))
        enhance_down_diff = torch.abs((enhance_y*down_kernel).sum(axis=-2))

        down_part = ((enhance_down_diff - real_down_diff)**2).sum()

        #j - j - 1 center - top
        top_kernel = torch.ones(2, real_image_avg.shape[-1], device=real_image.device)
        top_kernel[0,:] = -1.0
        top_kernel = top_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        real_top_diff = torch.abs((real_y * top_kernel).sum(axis=-2))
        enhance_top_diff = torch.abs((real_y * top_kernel).sum(axis=-2))

        top_part = ((enhance_top_diff - real_top_diff)**2).sum()

        loss_spa = (left_part + right_part + down_part + top_part)/(real_image_avg.shape[-2]*real_image_avg.shape[-1])
    
    return loss_spa


if __name__ == '__main__':
    real_image = torch.tensor([[0.5, -0.3, -0.9, 1.0],[-0.22, 0.35, 0.51, 0.88],[1.0, 0.77, 0.44, -0.11],[0.22, -0.97, 0.23, 0.55]])
    enhanced_image = torch.tensor([[0.25, -0.13, -.9, 0.45],[-0.16, 0.15, 0.21, -0.48],[0.85, 0.71, 0.54, -0.21],[0.32, 0.07, -0.33, 0.15]])

    real_image = real_image.unsqueeze(0)
    real_image = torch.cat([real_image, real_image, real_image], axis=0)
    real_image = real_image.unsqueeze(0)

    enhanced_image = enhanced_image.unsqueeze(0)
    enhanced_image = torch.cat([enhanced_image, enhanced_image, enhanced_image], axis=0)
    enhanced_image = enhanced_image.unsqueeze(0)

    c = spatial_consistency_loss(real_image, enhanced_image, local_region = 2)
    print(c)
