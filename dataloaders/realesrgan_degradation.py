'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
import sys
sys.path.append(os.getcwd())
import cv2

import torch
from torchvision.transforms import Normalize, Compose
import torch.nn.functional as F

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian
import random

class RealesrganDegradation(torch.nn.Module):
    def __init__(self, args_degradation, use_usm=True, sf=4, resize_lq=True):
        super(RealesrganDegradation, self).__init__()
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.args_degradation = args_degradation
        self.use_usm = use_usm
        self.sf = sf
        self.resize_lq = resize_lq

    def forward(self, batch):
        im_gt = batch['gt'].cuda()
        if self.use_usm:
            im_gt = self.usm_sharpener(im_gt)
        im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
        kernel1 = batch['kernel1'].cuda()
        kernel2 = batch['kernel2'].cuda()
        sinc_kernel = batch['sinc_kernel'].cuda()

        ori_h, ori_w = im_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.args_degradation['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.args_degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.args_degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.args_degradation['gray_noise_prob']
        if random.random() < self.args_degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.args_degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.args_degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.args_degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < self.args_degradation['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.args_degradation['resize_prob2'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.args_degradation['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.args_degradation['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(int(ori_h / self.sf * scale),
                        int(ori_w / self.sf * scale)),
                mode=mode,
                )
        # add noise
        gray_noise_prob = self.args_degradation['gray_noise_prob2']
        if random.random() < self.args_degradation['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.args_degradation['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.args_degradation['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
                )
                
        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // self.sf,
                            ori_w // self.sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.args_degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.args_degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // self.sf,
                            ori_w // self.sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
        
        # clamp and round
        im_lq = torch.clamp(out, 0, 1.0)
        
        # random crop
        # gt_size = self.args_degradation['gt_size']
        # im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, self.sf)
        lq, gt = im_lq, im_gt

        gt = torch.clamp(gt, 0, 1)
        lq = torch.clamp(lq, 0, 1)

        return lq, gt