## python
import imageio
import os
import pickle
from PIL import Image
import numpy as np
import time
## deep learning
import torch.utils.data as data


## local define
from data import dedata
from data import common

# import dedata
# import common

class DeData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark

        # args.dir_data : /store/dataset/vdenoising/
        # self.name : TOFlow
        # self.apath = os.path.join(args.dir_data, self.name)
        self._set_filesystem(args.dir_data)

        ## args.ext : dataset file extension, "sep or img, reset"
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_tseqs, list_nseqs = self._scan()

        if args.ext.find('img') >= 0 or benchmark:
            self.dir_tseqs, self.dir_nseqs = list_tseqs, list_nseqs
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_target.replace(self.apath, path_bin),
                exist_ok=True
            )
            os.makedirs(
                self.dir_noise.replace(self.apath, path_bin),
                exist_ok=True
            )
            ## list_tseqs has the same property with self.dir_tseqs
            ## they all represent directory names
            self.dir_tseqs, self.dir_nseqs = [], []
            ## target images
            for h in list_tseqs:
                b = h.replace(self.apath, path_bin)
                self.dir_tseqs.append(b)

                ## transfer png images to bin files.
                os.makedirs(b, exist_ok=True)
                for filename in os.listdir(h):
                    imgpath = os.path.join(h, filename)
                    bpath = os.path.join(b, filename.replace(self.ext[0],'.pt'))
                    self._check_and_load(args.ext, imgpath, bpath, verbose=True)
            ## noise images
            for l in list_nseqs:
                b = l.replace(self.apath, path_bin)
                self.dir_nseqs.append(b)
                ## transfer png images to bin files.
                os.makedirs(b, exist_ok=True)
                for filename in os.listdir(l):
                    imgpath = os.path.join(l, filename)
                    bpath = os.path.join(b, filename.replace(self.ext[1],'.pt'))
                    self._check_and_load(args.ext, imgpath, bpath, verbose=True)

        ## Data Augmentation
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.dir_tseqs)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)
        #super(self, DeData).__init__()

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_target = os.path.join(self.apath, 'target')
        self.dir_noise = os.path.join(self.apath, 'input')
        #if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

    def _scan(self):
        ## the dir names of target sequences
        dir_tseqs = []
        dir_nseqs = []

        dir_videos = os.listdir(self.dir_target)
        dir_videos.sort()
        ## loop for video dir
        for dir_video in dir_videos:
            dir_seqs = os.listdir(os.path.join(self.dir_target, dir_video))
            dir_seqs.sort()
            ## loop for seq dir
            for dir_seq in dir_seqs:
                # self.dir_target: /store/dataset/vdenoising/ToFlow/target/
                # dir_video      :00001
                # dir_seq        :0266
                dir_tseqs.append(os.path.join(
                    self.dir_target, dir_video, dir_seq
                ))
                ## assert that we can find the corresponding target
                ## with the same directory tree
                dir_nseqs.append(os.path.join(
                    self.dir_noise, dir_video, dir_seq
                ))

        return dir_tseqs, dir_nseqs

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    """
    Return sequence data with shape [t, c, h, w]
    t is the number of frames in each sequence.
    """
    def __getitem__(self, idx):
        idx = self._get_index(idx)
        nseqs, tseqs, dirname = self._load_file(idx)
        #print("load file time:", etime-stime, dirname)
        pair = self.get_patch(nseqs, tseqs)
        # print(pair[0].shape)
        # set color channel to 3 if not
        # print(self.args.n_colors)
        pair = common.set_channel(pair, n_channel=self.args.n_colors)
        # to tensor
        pair_t = common.np2Tensor(pair, rgb_range=self.args.rgb_range)
        #print("process data time:", e2time-etime)
        return pair_t[1], pair_t[0], dirname

    def __len__(self):
        if self.train:
            return len(self.dir_tseqs) * self.repeat
        else:
            return len(self.dir_tseqs)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.dir_tseqs)
        else:
            return idx


    """
    Return a tensor accoring to the all images in a directory path
    根据 文件夹路径 返回文件夹中所有图像
    Input :
        target and noise dirs, index
    Output:
        nseqs & tseqs : Tensor with shape [t,h,w,c], t means frame number, which is 7 in TOFlow dataset
        dirname       : Name of directory.
    """
    def _load_file(self, idx):
        dir_tseq = self.dir_tseqs[idx]
        dir_nseq = self.dir_nseqs[idx]

        frame_tseqs = []
        frame_nseqs = []

        filenames = os.listdir(dir_tseq)
        filenames.sort()
        for filename in filenames:
            path_tseq = os.path.join(dir_tseq, filename)
            path_nseq = os.path.join(dir_nseq, filename)
            ## whether path_nseq existed or not
            ## raise pathnotfind error
            if not os.path.exists(path_nseq):
                raise ValueError("The file path of the corresponding noise image does not exist")
            ## load image files
            if self.args.ext == 'img':
                frame_tseqs.append(imageio.imread(path_tseq))
                frame_nseqs.append(imageio.imread(path_nseq))
            ## load bin files
            elif self.args.ext.find('sep') >= 0:
                with open(path_tseq, 'rb') as _f:
                    frame_tseqs.append(pickle.load(_f))
                with open(path_nseq, 'rb') as _f:
                    frame_nseqs.append(pickle.load(_f))
        ## stack from list to array
        return np.stack(frame_tseqs, axis=0), np.stack(frame_nseqs, axis=0), dir_tseq

    """
    Return random cropped patch from sequence files
    Input :
        nseqs&tseqs : Tensor with shape [t,h,w,c]
    Output:
        pair        : list type with cropped patches
                    *[nseqs, tseqs] with shape [t, ph, pw, c]
    """
    def get_patch(self, frame_tseqs, frame_nseqs):
        if self.train:
            patch_tseqs, patch_nseqs = common.get_patch(
                            frame_tseqs, frame_nseqs,
                            patch_size = self.args.patch_size
                            )
            if not self.args.no_augment:
                patch_tseqs, patch_nseqs = common.augment([patch_tseqs, patch_nseqs])
        else:
            patch_tseqs, patch_nseqs = common.get_center_patch(
                            frame_tseqs, frame_nseqs,
                            patch_size = self.args.test_patch_size
                            )

        return patch_tseqs, patch_nseqs

if __name__=='__main__':
    print("hello world")
    DeData()
