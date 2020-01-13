import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as func

from option import args

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        ## make directory for saving experimental results.
        ## init
        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        ## load models
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch,  is_best=False, plot_title=None):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        if isinstance(trainer.loss, list):
            # names = ["loss_denoised", "loss_flow"]
            # if self.args.loss_rel is not None:
            #     names.append("loss_rel")
            for i,l in enumerate(trainer.loss):
                l.save(self.dir, l.name)
                l.plot_loss(self.dir, epoch, l.name)
        else:
            trainer.loss.save(self.dir, l.name)
            trainer.loss.plot_loss(self.dir, epoch, l.name)
        # trainer.loss.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch, label=plot_title)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def show_test(self):
        #1. save log result
        # self.log with shape [frames, idx_data]
        torch.save(self.log, self.get_path('psnr_log.pt'))
        self.plot_psnr(self.args.n_frames, name = 'idx_frame')
        #2. plot PSNR according to frames

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    ## epoch can be see as the axis length
    def plot_psnr(self, epoch, mean = True, dimension=1, label=None, name=None):
        assert epoch == self.log.shape[1 - dimension]

        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            ## 定制化需求
            if label is None:
                label = 'denoised on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            ## the 0'th dimension should be same with epoch
            if len(self.log.shape) == 3:
                if mean:
                    plt.plot(
                        axis,
                        self.log[:, :, idx_data].mean(dimension).numpy(),
                        label = 'PSNR'
                    )
                else:
                    #print("plot epoch'th PSNR")
                    plt.plot(
                        axis,
                        self.log[-1, :, idx_data].numpy(),
                        label = 'PSNR'
                    )
            elif len(self.log.shape) == 2:
                plt.plot(
                    axis,
                    self.log[:, idx_data].numpy(),
                    label = 'PSNR'
                )
            else:
                raise ValueError("Dimension Error, dimension of self.log should be 2 or 3")
            plt.legend()
            plt.xlabel(name)
            plt.ylabel('PSNR')
            plt.grid(True)
            if name is None:
                plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            else:
                plt.savefig(self.get_path(name+'_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]

        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, idx_frame = None):
        if self.args.save_results:
            if idx_frame is None:
                filename = self.get_path(
                    'results-{}'.format(dataset if isinstance(dataset, str) else dataset.dataset.name),
                    '{}_{}_'.format(*filename.split('/'))
                )
            else:
                assert type(idx_frame) is int
                filename = self.get_path(
                    'results-{}'.format(dataset if isinstance(dataset, str) else dataset.dataset.name),
                    '{}_{}_{}_'.format(*filename.split('/'), 'frame'+str(idx_frame))
                )

            #postfix = ('Est', 'Noise', 'Target')
            #for v, p in zip(save_list, postfix):
            for p,v in save_list.items():
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(fakeTarget, target, rgb_range, dataset=None):

    diff = (fakeTarget - target) / rgb_range
    valid = diff
    mse = valid.pow(2).mean()
    return -10 * math.log10(mse)

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

import cv2
import numpy as np


def vis_opticalflow(flow):
    bgrs = torch.ones([0, flow.shape[2],flow.shape[3],3], dtype=torch.uint8)
    for img_idx in range(len(flow)):
        flow = flow[img_idx].detach().cpu().numpy().transpose([1,2,0])
        hsv = np.zeros([flow.shape[0],flow.shape[1],3], dtype=np.uint8)
        hsv[..., 2] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        #print(hsv.shape)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #print(bgrs.shape)
        #print(torch.tensor(bgr).unsqueeze(0).shape)
        bgrs = torch.cat((bgrs, torch.tensor(bgr).unsqueeze(0)), dim = 0)
    return bgrs.permute(0,3,1,2)


def warpfunc(ori_image, flow_map, sparse=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow.
    Code heavily inspired by
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = ori_image.size()
    if sparse:
        flow_map = sparse_max_pool(flow_map, (H, W))
    else:
        flow_map = func.interpolate(flow_map, (H, W), mode='area')
    device = torch.device('cpu' if args.cpu else 'cuda')
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    #print(grid)
    grid = grid
    vgrid = grid + flow_map

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :]/max(W-1, 1)-1.0
    vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :]/max(H-1, 1)-1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = func.grid_sample(ori_image, vgrid)
    # Define a first mask containing pixels that wasn't properly interpolated
    mask = torch.autograd.Variable(torch.ones(ori_image.size())).to(device)
    mask = func.grid_sample(mask, vgrid)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output


## adopt flow map to warp original image
def warpfunc_tanh(ori_image, flow_map, sparse=False):

    b, _, h, w = ori_image.size()
    if sparse:
        flow_map = sparse_max_pool(flow_map, (h, w))
    else:
        flow_map = func.interpolate(flow_map, (h, w), mode='area')

    height, width = flow_map.shape[2:4]
    assert height == h and width == w

    height_gap = 2 / (height - 1)
    width_gap = 2 / (width - 1)
    #print("::::::", height, width, height_gap, width_gap)
    height, width = torch.meshgrid([torch.range(-1, 1, height_gap), torch.range(-1, 1, width_gap)])

    device = torch.device('cpu' if args.cpu else 'cuda')
    identity = torch.stack([width, height]).to(device)

    relative_place = flow_map + identity

    relative_placeNWHC = relative_place.permute(0, 2, 3, 1)
    warped_image = func.grid_sample(ori_image, relative_placeNWHC)
    return warped_image

## https://github.com/ClementPinard/FlowNetPytorch/blob/03c17be89ee0f613d3ac8d6f37d02830034a424e/util.py#L35
## have not been invoked.
def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)
