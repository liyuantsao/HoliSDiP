import cv2
import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
import math

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor

from PIL import Image



class DataLoader(Dataset):
    def __init__(self, opt, fix_size=512): 
        
        self.opt = opt['kernel_info']
        self.image_root = opt['gt_path']
        self.fix_size = fix_size
        exts = ['*.jpg', '*.png']
        
        self.image_list = []
        for image_root in self.image_root:
            for ext in exts:
                image_list = sorted(glob.glob(os.path.join(image_root, ext)))
                self.image_list += image_list
                # if add lsdir dataset
                image_list = sorted(glob.glob(os.path.join(image_root, '00*', ext)))
                self.image_list += image_list

        self.img_preproc = transforms.Compose([
            transforms.RandomCrop(fix_size),
            transforms.ToTensor(),
        ])

        # blur settings for the first degradation
        self.blur_kernel_size = self.opt['blur_kernel_size']
        self.kernel_list = self.opt['kernel_list']
        self.kernel_prob = self.opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = self.opt['blur_sigma']
        self.betag_range = self.opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = self.opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = self.opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = self.opt['blur_kernel_size2']
        self.kernel_list2 = self.opt['kernel_list2']
        self.kernel_prob2 = self.opt['kernel_prob2']
        self.blur_sigma2 = self.opt['blur_sigma2']
        self.betag_range2 = self.opt['betag_range2']
        self.betap_range2 = self.opt['betap_range2']
        self.sinc_prob2 = self.opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = self.opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        print(f'The dataset length: {len(self.image_list)}')


    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        image = self.img_preproc(image)
        
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': image, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'lq_path': self.image_list[index]}
        return return_d
        

    def __len__(self):
        return len(self.image_list)
        
