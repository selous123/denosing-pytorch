import data
from option import args
import time
import glob
import imageio
import torch
import os
"""
We need this function to predict single test files.
We are supposed to save the PSNR results for each frames,

and plot results.

Author: Tao Zhang (Selous)
Data  : 2019.12.26(Start)~
"""


def load_video_data(data_dir):
    noised_paths = glob.glob(os.path.join(data_dir, "noised", "*.png"))
    target_paths = glob.glob(os.path.join(data_dir, "target", "*.png"))
    noised_images = []
    target_images = []
    for noised_path, target_path in zip(noised_paths, target_paths):
        noised_images.append(torch.tensor(imageio.imread(noised_path)))
        target_images.append(torch.tensor(imageio.imread(target_path)))

    ##frames, batch, CHW
    noised_images_torch = torch.stack(noised_images, dim=0).unsqueeze(1).permute(0,1,4,2,3)
    noised_images_torch = torch.stack(target_images, dim=0).unsqueeze(1).permute(0,1,4,2,3)

    return noised_images_torch, noised_images_torch


dirdata = '/home/lrh/git/data/v1'

noise_input, target = load_video_data(dirdata)

print(noise_input.shape)
print(target.shape)
