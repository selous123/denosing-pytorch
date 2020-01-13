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
Data  : 2019.12.26(Start)~2020.01.13(rewrite)
"""
import model.frvd
import utility
from option import args
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
        n_paths = noised_paths[prepoint:point-1]
        t_paths = target_paths[prepoint:point-1]
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
    savefilename = savefilename.replace('.npy', '.pdf')

    n_frames = data.shape[0]
    axis = np.linspace(1, n_frames, n_frames).

    label = 'denoised on vimdeo test dataset via frame index'
    fig = plt.figure()
    plt.title(label)
    ## the 0'th dimension should be same with epoch
    plt.plot(axis, data, label = 'PSNR')
    plt.legend()
    plt.xlabel(frame_index)
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



if __name__=='__main__':
    device = torch.device('cpu' if args.cpu else 'cuda')

    _model = model.frvd.make_model(args)
    _model.load_state_dict(torch.load(args.pre_train))
    _model = _model.to(device)
    _model.eval()
    #dirdata = '/home/lrh/git/data/v2'
    dirdata = '/store/dataset/vdenoising/vimeo/v1/'
    data_generator = load_video_data(dirdata)

    desdir = os.path.join(dirdata, 'denoised', _model.name)
    os.makedirs(desdir, exist_ok=True)

    for video_idx, (noise_input, target, seq_ids) in tqdm(enumerate(data_generator), ncols=80):
        noise_input, target = noise_input.to(device), target.to(device)
        h, w = noise_input.shape[3:]
        _model.init_hidden(h,w)
        start_seqid = seq_ids[0]
        end_seqid = seq_ids[1]

        ## store PSNRs for each sequence.
        psnrs = []
        ## video sequence.
        ## Recurrent feed data into pre-trained model
        for idx_frame in range(len(noise_input)):
            curr_seqid = idx_frame + start_seqid + 1
            nseq, tseq = noise_input[idx_frame], target[idx_frame]

            with torch.no_grad():
                ## Inference time.
                fakeTarget, _ = _model(nseq)
                fakeTarget = utility.quantize(fakeTarget, args.rgb_range)
                ## calculate PSNR value.
                psnrs.append(utility.calc_psnr(fakeTarget, tseq, args.rgb_range))

                ## store the image files.
                normalized = fakeTarget[0].mul(255 / args.rgb_range)
                target_cpu = normalized.byte().permute(1, 2, 0).cpu()
                filename = os.path.join(desdir, 'frame_'+str(curr_seqid).zfill(3)+'.png')
                imageio.imwrite(filename, target_cpu.numpy())

        ## save PSNR values.
        print(psnrs)
        np.save(os.path.join(desdir, 'psnr_'+str(start_seqid)+'-'+str(end_seqid)+'.npy'), np.array(psnrs))
        print('Saved PSNR! Frames: {}~{}'.format(start_seqid, end_seqid))

    print('Done!!')
