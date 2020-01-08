import data
from option import args
import time
import glob
import imageio
import torch
import os
from tqdm import tqdm
"""
We need this function to predict single test files.
We are supposed to save the PSNR results for each frames,

and plot results.

Author: Tao Zhang (Selous)
Data  : 2019.12.26(Start)~
"""
import model.frvd
import utility
from option import args
#checkpoint = utility.checkpoint(args)
_model = model.frvd.make_model(args)
_model.load_state_dict(torch.load(args.pre_train))

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

    return noised_images_torch.type(torch.FloatTensor), noised_images_torch.type(torch.FloatTensor)


#dirdata = '/home/lrh/git/data/v2'
dirdata = '/store/dataset/vdenoising/vimeo/v1/'
noise_input, target = load_video_data(dirdata)

device = torch.device('cpu' if args.cpu else 'cuda')
noise_input, target = noise_input.to(device), target.to(device)

_model = _model.to(device)
_model.eval()

h, w = noise_input.shape[3:]
_model.init_hidden(h,w)

print(noise_input.shape)
print(h,w)
import time
import numpy as np
psnrs = []
for idx_frame in tqdm(range(len(noise_input)), ncols=80):
    nseq, tseq = noise_input[idx_frame], target[idx_frame]
    ## fakeTarget for t'th denoised frame
    ## fakeNoise for (t-1)'th noised frame alignmented
    ## after optical-flow
    with torch.no_grad():
        fakeTarget, _ = _model(nseq)
    psnrs.append(utility.calc_psnr(fakeTarget, tseq, args.rgb_range))

print(psnrs)
np.save('/home/lrh/git/FRVD-pytorch/experiment/vimeo_denoise_test.npy', np.array(psnrs))
print('saved!')


#_model(noise_input)
