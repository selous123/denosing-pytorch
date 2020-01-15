import os
import torch
import glob
from tqdm import tqdm
import imageio
import numpy as np


import utility
from option import args

import model
checkpoint = utility.checkpoint(args)
"""
Test Code for testing real dataset with no target.

Author: Tao Zhang (Selous)
Data  : 2020.01.15
"""



def torch2img(img_torch):
    normalized = img_torch.mul(255 / args.rgb_range)
    target_cpu = normalized.byte().permute(1, 2, 0).cpu()
    return target_cpu

"""
Load video frames from vimeo dataset splited by ffpmeg software.
    example video: https://vimeo.com/142480565
    video frame is in 'vimeo/v1'+[noised, target]
    and pivot points for the frame which transforms scenario..
"""
def load_video_data(data_dir, points = None):
    prepoint = 0
    if points is None:
        pointpath = os.path.join(data_dir, 'point.txt')
        f = open(pointpath)
        points = [int(line) for line in f.readlines()]

    noised_paths = glob.glob(os.path.join(data_dir, "noised", "*.png"))
    #target_paths = glob.glob(os.path.join(data_dir, "target", "*.png"))
    noised_paths.sort()
    #print(noised_paths)
    #target_paths.sort()
    for point in points:
        n_paths = noised_paths[prepoint:point]
        #t_paths = target_paths[prepoint:point-1]
        noised_images = []
        #target_images = []
        for noised_path in n_paths:
            noised_images.append(torch.tensor(imageio.imread(noised_path)).mul_(args.rgb_range / 255.0))
            #target_images.append(torch.tensor(imageio.imread(target_path)).mul_(args.rgb_range / 255.0))
        ##frames, batch, CHW
        noised_images_torch = torch.stack(noised_images, dim=0).unsqueeze(1).permute(0,1,4,2,3).type(torch.FloatTensor)
        #target_images_torch = torch.stack(target_images, dim=0).unsqueeze(1).permute(0,1,4,2,3).type(torch.FloatTensor)

        yield noised_images_torch, [prepoint, point-1]

        prepoint = point


def inference():
    device = torch.device('cpu' if args.cpu else 'cuda')

    #_model = model.frvd.make_model(args)
    _model = model.Model(args, checkpoint)
    #_model.load_state_dict(torch.load(args.pre_train))
    _model = _model.to(device)
    _model.eval()
    #dirdata = '/home/lrh/git/data/v2'
    #dirdata = '/store/dataset/vdenoising/vimeo/v1/'
    dirdata = args.dir_data
    data_generator = load_video_data(dirdata)

    desdir = os.path.join(dirdata, 'denoised', _model.name)
    os.makedirs(desdir, exist_ok=True)

    for video_idx, (noise_input, seq_ids) in tqdm(enumerate(data_generator), ncols=80):
        noise_input = noise_input.to(device)
        h, w = noise_input.shape[3:]
        _model.model.init_hidden(h,w)
        start_seqid = seq_ids[0]
        end_seqid = seq_ids[1]

        ## video sequence.
        ## Recurrent feed data into pre-trained model
        for idx_frame in range(len(noise_input)):
            curr_seqid = idx_frame + start_seqid + 1
            nseq = noise_input[idx_frame]
            with torch.no_grad():
                ## Inference time.
                fakeTarget = _model(nseq)
                if type(fakeTarget) in [tuple, list]:
                    fakeTarget = fakeTarget[0]
                fakeTarget = utility.quantize(fakeTarget, args.rgb_range)
                ## store the image files.
                target_cpu = torch2img(fakeTarget[0])
                filename = os.path.join(desdir, 'frame_'+str(curr_seqid).zfill(3)+'.png')
                imageio.imwrite(filename, target_cpu.numpy())

                flow = _model.model.get_opticalflow_map()
                flowmap =  utility.vis_opticalflow(flow)
                ## store the image files.
                target_cpu = torch2img(flowmap[0])
                filename = os.path.join(desdir, 'frame_'+str(curr_seqid).zfill(3)+'_flowmap.png')
                imageio.imwrite(filename, target_cpu.numpy())



    print('Done!!')


if __name__=='__main__':
    inference()
