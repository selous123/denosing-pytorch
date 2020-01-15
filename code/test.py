import data
from option import args
import time
import glob
import imageio
import torch
import os
from tqdm import tqdm
import numpy as np
"""
We need this function to predict single test files.
We are supposed to save the PSNR results for each frames,
and plot results.

Author: Tao Zhang (Selous)
Data  : 2019.12.26(Start)
    ~2020.01.13(Rewrite)
    ~2020.01.15(Debug)
"""
import utility
from option import args
import model
checkpoint = utility.checkpoint(args)
#checkpoint = utility.checkpoint(args)

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
    target_paths = glob.glob(os.path.join(data_dir, "target", "*.png"))
    noised_paths.sort()
    target_paths.sort()
    for point in points:
        n_paths = noised_paths[prepoint:point]
        t_paths = target_paths[prepoint:point]
        noised_images = []
        target_images = []
        for noised_path, target_path in zip(n_paths, t_paths):
            noised_images.append(torch.tensor(imageio.imread(noised_path)).mul_(args.rgb_range / 255.0))
            target_images.append(torch.tensor(imageio.imread(target_path)).mul_(args.rgb_range / 255.0))
        ##frames, batch, CHW
        noised_images_torch = torch.stack(noised_images, dim=0).unsqueeze(1).permute(0,1,4,2,3).type(torch.FloatTensor)
        target_images_torch = torch.stack(target_images, dim=0).unsqueeze(1).permute(0,1,4,2,3).type(torch.FloatTensor)

        yield noised_images_torch, target_images_torch, [prepoint, point-1]

        prepoint = point


def plot_psnr(filepath = None):
    #filepath = '/store/dataset/vdenoising/vimeo/v1/denoised/frvd/xx.npy'
    if filepath is None:
        raise ValueError('filepath should be not None')
    else:
        #data with shape [n_frames]
        data = np.load(filepath)
    # savefilename = os.path.join(
    #     os.path.dirname(filepath),
    #     os.path.basename(filepath).split('.')[0]+'.pdf'
    # )
    savefilename = filepath.replace('.npy', '.pdf')

    n_frames = data.shape[0]
    axis = np.linspace(1, n_frames, n_frames)

    label = 'denoised on vimdeo test dataset via frame index'
    fig = plt.figure()
    plt.title(label)
    ## the 0'th dimension should be same with epoch
    plt.plot(axis, data, label = 'PSNR')
    plt.legend()
    plt.xlabel('frame_index')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(savefilename)
    plt.close(fig)


def plot_psnrs_filenames(dirdata, model_name):
    desdir = os.path.join(dirdata, 'denoised', model_name)

    pointpath = os.path.join(data_dir, 'point.txt')
    f = open(pointpath)
    points = [int(line) for line in f.readlines()]
    start_seqid = 0
    for point in points:
        end_seqid = point - 1
        filename = 'psnr_'+str(start_seqid)+'-'+str(end_seqid)+'.npy'
        filepath = os.path.join(desdir, filename)
        plot_psnr(filepath = filepath)

        start_seqid = point
    print('Plot, Done')



def torch2img(img_torch):
    normalized = img_torch.mul(255 / args.rgb_range)
    target_cpu = normalized.byte().permute(1, 2, 0).cpu()
    return target_cpu


def torch2save(dic_torch, desdir, curr_seqid = 0):
    for p, v in dic_torch.items():
        filename = os.path.join(desdir, "frame_"+str(curr_seqid).zfill(3) + "_" + p +".png")
        normalized = v[0].mul(255 / args.rgb_range)
        tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
        imageio.imwrite(filename, tensor_cpu.numpy())

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np

def addnote2img(img_np, note):
    font = ImageFont.truetype("FreeMonoBold.ttf", 30)
    img = Image.fromarray(img_np.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0),note,'red', font=font)
    return np.array(img)
"""
Return splice images:
    ---------------------------------
    |Noised Images, Denoised Images.|
    |Optical-flow ,   Target Images.|
    ---------------------------------
    |          Plot PSNR            |
    ---------------------------------
Arguments:
    imgs : list type with four images. numpy array.
        with shape [256, 448, 3] HWC
"""
def splice_image(imgs):
    assert len(imgs) == 4
    denoised_img   = addnote2img(imgs[0], 'Denoised')
    noised_img = addnote2img(imgs[1], 'Noised')
    c1 = np.concatenate((noised_img, denoised_img), axis=1)

    flowmap    = addnote2img(imgs[2], 'Flowmap')
    target_img = addnote2img(imgs[3], 'Target')
    c2 = np.concatenate((flowmap, target_img), axis = 1)

    c3 = np.concatenate((c1, c2), axis = 0)
    return c3

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.measure
def splice_save_image(img_seqs, desroot='.', startid=0):
    os.makedirs(desroot, exist_ok=True)
    psnrs = []
    psnrs_noise = []
    for idx_batch, imgs in enumerate(img_seqs):
        curr_seqid = startid + idx_batch + 1
        length = idx_batch + 1
        #print(len(imgs))
        c_img = splice_image(imgs)
        fig, ax = plt.subplots()
        ax.imshow(c_img)
        ax.axis('off')

        axisxlength = c_img.shape[1]
        axisx = np.linspace(1, axisxlength, length)
        #axisylength = c_img.shape[0]
        psnrs.append(skimage.measure.compare_psnr(imgs[3], imgs[0], args.rgb_range))
        psnrs_noise.append(skimage.measure.compare_psnr(imgs[3], imgs[1], args.rgb_range))

        divider = make_axes_locatable(ax)
        axbottom = divider.append_axes("bottom", size=1.6, pad=0.1, sharex=ax)
        axbottom.plot(axisx, psnrs, color='red', marker="x", label='denoise')
        axbottom.plot(axisx, psnrs_noise, color='blue', marker="o", label='noise')
        axbottom.set_yticks([])
        axbottom.set_xticks([])
        axbottom.legend()
        #axbottom.set_xticks(axisx)
        #axbottom.set_xticklabels(np.linspace(1, length, length))
        axbottom.margins(x=0)
        plt.tight_layout()
        plt.savefig(os.path.join(desroot, 'frame_'+str(curr_seqid).zfill(3)+'.png'), bbox_inches = 'tight',pad_inches = 0)
        plt.close()





def inference():
    device = torch.device('cpu' if args.cpu else 'cuda')

    _model = model.Model(args, checkpoint)
    _model = _model.to(device)
    _model.eval()
    #dirdata = '/home/lrh/git/data/v2'
    dirdata = args.dir_data
    data_generator = load_video_data(dirdata)

    desdir = os.path.join(dirdata, 'denoised', _model.name)
    os.makedirs(desdir, exist_ok=True)

    for video_idx, (noise_input, target, seq_ids) in tqdm(enumerate(data_generator), ncols=80):
        noise_input, target = noise_input.to(device), target.to(device)
        h, w = noise_input.shape[3:]
        _model.model.init_hidden(h,w)
        start_seqid = seq_ids[0]
        end_seqid = seq_ids[1]

        ## store PSNRs for each sequence.
        psnrs = []
        ## video sequence.
        ## Recurrent feed data into pre-trained model
        save_list = {}
        for idx_frame in range(len(noise_input)):
            curr_seqid = idx_frame + start_seqid + 1
            nseq, tseq = noise_input[idx_frame], target[idx_frame]

            with torch.no_grad():
                ## Inference time.
                fakeTarget = _model(nseq)
                if type(fakeTarget) in [tuple, list]:
                    fakeTarget = fakeTarget[0]

                fakeTarget = utility.quantize(fakeTarget, args.rgb_range)
                ## calculate PSNR value.
                psnrs.append(utility.calc_psnr(fakeTarget, tseq, args.rgb_range))

                ## store the image files.
                # normalized = fakeTarget[0].mul(255 / args.rgb_range)
                # target_cpu = normalized.byte().permute(1, 2, 0).cpu()
                # filename = os.path.join(desdir, 'frame_'+str(curr_seqid).zfill(3)+'.png')
                # imageio.imwrite(filename, target_cpu.numpy())
                save_list['est'] = fakeTarget

                if args.save_gt:
                    save_list['noise'] = nseq
                    save_list['target'] = tseq

                if args.save_of:
                    flow = _model.model.get_opticalflow_map()
                    flowmap =  utility.vis_opticalflow(flow)
                    save_list['zflowmap'] = flowmap

                    prest_warped = _model.model.get_warpresult()
                    save_list['prest-warped'] = utility.quantize(prest_warped, args.rgb_range)
                ## store the image files.
                # target_cpu = torch2img(flowmap[0])
                # filename = os.path.join(desdir, 'frame_'+str(curr_seqid).zfill(3)+'_flowmap.png')
                # imageio.imwrite(filename, target_cpu.numpy())
                torch2save(save_list, desdir, curr_seqid)



        ## save PSNR values.
        #print(psnrs)
        filename = os.path.join(desdir, 'psnr_'+str(start_seqid + 1)+'-'+str(end_seqid + 1)+'.npy')
        np.save(filename, np.array(psnrs))
        plot_psnr(filename)
        print('Saved PSNR! Frames: {}~{}'.format(start_seqid + 1, end_seqid + 1))

    print('Done!!')

from matplotlib import pyplot as plt
if __name__=='__main__':

    ## Inference
    if False:
        inference()
    else:
        pointpath = os.path.join(args.dir_data, 'point.txt')
        f = open(pointpath)
        points = [int(line) for line in f.readlines()]


        suffixes = ["est", "noise", "zflowmap", "target"]
        desdir = os.path.join(args.dir_data, 'denoised', args.model)
        filepathseqs = []

        for suffix in suffixes:
            filepathes = glob.glob(os.path.join(desdir, '*' + suffix + '.png'))
            filepathes.sort()
            filepathseqs.append(filepathes)


        img_seqs = []
        for index in range(len(filepathseqs[0])):
            imgs = []
            for jndex in range(len(filepathseqs)):
                imgs.append(imageio.imread(filepathseqs[jndex][index]))
            img_seqs.append(imgs)

        des_root = os.path.join(desdir, 'spliced')
        start_point = 0
        for point in points:
            splice_save_image(img_seqs[start_point:point], des_root, startid=start_point)
            start_point = point

    # length = 3
    #
    # dataroot = '/home/lrh/git/FRVD-pytorch/experiment/frvdwof-v0.2/results-ToFlow/'
    # filename0 = ['00010_0144_frame2_Noise.png','00010_0144_frame2_Est.png',\
    #             '00010_0144_frame2_flow.png','00010_0144_frame2_Target.png']
    # filename1 = ['00010_0144_frame3_Noise.png','00010_0144_frame3_Est.png',\
    #             '00010_0144_frame3_flow.png','00010_0144_frame3_Target.png']
    # filename2 = ['00010_0144_frame4_Noise.png','00010_0144_frame4_Est.png',\
    #             '00010_0144_frame4_flow.png','00010_0144_frame4_Target.png']
    # filenameseqs = (filename0, filename1, filename2)
    # img_seqs = []
    #
    # for filenames in filenameseqs:
    #     imgs = []
    #     for filename in filenames:
    #         imgs.append(imageio.imread(os.path.join(dataroot, filename)))
    #     img_seqs.append(imgs)
    #
    #
