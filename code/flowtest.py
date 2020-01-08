import data
from option import args
import time
import glob
import imageio
import torch
import os
import utility
from tqdm import tqdm
"""
We need this function to predict single test files.
We are supposed to save the PSNR results for each frames,

and plot results.

Author: Tao Zhang (Selous)
Data  : 2019.12.26(Start)~
"""
import model.flow.flownets
import utility
from option import args



ckp = utility.checkpoint(args)
_model = model.flow.flownets.make_model(args)
_model.load_state_dict(torch.load(args.pre_train))

ckp.add_log(torch.zeros(args.n_frames, 1))

def load_video_data(data_dir):
    noised_paths = glob.glob(os.path.join(data_dir, "noised", "*.png"))
    target_paths = glob.glob(os.path.join(data_dir, "target", "*.png"))
    noised_paths.sort()
    target_paths.sort()
    #print([os.path.basename(target_path) for target_path in target_paths])
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

print(noise_input.shape)
device = torch.device('cpu' if args.cpu else 'cuda')
noise_input, target = noise_input.to(device), target.to(device)

_model = _model.to(device)
_model.eval()


print(noise_input.shape)

if args.save_results: ckp.begin_background()
save_list = {}
for idx_frame in tqdm(range(len(target)-1), ncols=80):
    ## after optical-flow
    input = torch.cat((target[idx_frame], target[idx_frame+1]), dim=1)
    flowmap = _model(input)

    save_list['flow'] = utility.vis_opticalflow(flowmap)
    ## warped result
    warped_image = utility.warpfunc(target[idx_frame], flowmap)
    warped_image = utility.quantize(warped_image, args.rgb_range)

    save_list['warped'] = warped_image

    ckp.log[idx_frame, 0] = utility.calc_psnr(warped_image, target[idx_frame+1], args.rgb_range)

    if args.save_gt:
        save_list['source'] = target[idx_frame]
        save_list['Target'] = target[idx_frame+1]
        #save_list.extend([nseq, tseq])

    if args.save_results:
        ckp.save_results('vimeo', 'v1/test', save_list, idx_frame)

if args.save_results:
    ckp.end_background()

ckp.plot_psnr(args.n_frames, name = 'idx_frame')
