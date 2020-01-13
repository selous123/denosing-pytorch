import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp, ls=None, task=None):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        assert task is not None
        assert ls is not None

        self.name = task

        self.n_frames = args.n_frames
        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in ls.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('TVL1')>=0:
                module = import_module('loss.tvloss')
                loss_function = getattr(module, 'TVL1')()
            elif loss_type.find('TVL2')>=0:
                module = import_module('loss.tvloss')
                loss_function = getattr(module, 'TVL2')()

            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                #print(l['function'])
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)
        #print(self.loss_module)
        #print(self.loss)
    ## idx represents idx'th recurrent loss value
    ## calculate loss function
    def forward(self, data, idx = 0):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                ## TV Loss only need 1 parameter
                if l['type'].find('TV')>=0:
                    loss = l['function'](data['flowmap'])
                else:
                    loss = l['function'](data['est'], data['target'])

                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i, idx] += effective_loss.item()
            ## save the discriminator loss
            elif l['type'] == 'DIS':
                self.log[-1, i, idx] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1, idx] += loss_sum.item()
        #print(self.log)
        return loss_sum

    def batch_sum(self):
        for idx in range(len(self.loss)):
            self.log[-1, idx, -1] = self.log[-1, idx, :-1].sum()

        #print(self.log)
    ## whether this loss function consists of scheduler
    # for adjust learning rate values.
    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                print("l:", l)
                l.scheduler.step()

    def start_log(self):
        ## self.n_frames for frame loss
        ## +1 for sum loss
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss), self.n_frames + 1)))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1,:,-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], (c / (n_samples * self.n_frames))))

        return ''.join(log)

    def plot_loss(self, apath, epoch, name=None):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i, :-1].mean(dim = 1).numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            if name is None:
                plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            else:
                plt.savefig(os.path.join(apath, 'loss_{}_{}.pdf'.format(name,l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath, name=None):
        if name is None:
            torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
            torch.save(self.log, os.path.join(apath, 'loss_log.pt'))
        else:
            torch.save(self.state_dict(), os.path.join(apath, name + '.pt'))
            torch.save(self.log, os.path.join(apath, name + '_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()
