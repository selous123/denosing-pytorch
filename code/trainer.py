import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        ## ?? self.loss.step for what
        ## Loss STEP 1
        [l.step() for l in self.loss]

        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        ## Loss STEP 2
        [l.start_log() for l in self.loss]

        self.model.train()

        #self.loss_denoised = self.loss

        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (nseqs, tseqs, _,) in enumerate(self.loader_train):
            ## show dataset
            # from matplotlib import pyplot as plt
            # for idx in range(1, len(nseqs)):
            #     plt.subplot(3,4,idx)
            #     plt.imshow(nseqs[idx-1, 2].numpy().transpose(1,2,0))
            #     plt.subplot(3,4,idx+6)
            #     plt.imshow(tseqs[idx-1, 2].numpy().transpose(1,2,0))
            # plt.savefig('show.pdf')
            # exit(0)
            #print(len(nseqs))
            nseqs, tseqs = self.prepare(nseqs, tseqs)
            #print(nseqs.shape)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            self.model.init_hidden()
            loss = 0
            for idx_frame, (nseq, tseq) in enumerate(zip(nseqs, tseqs)):
                ## fakeTarget for t'th denoised frame
                ## fakeNoise for (t-1)'th noised frame alignmented
                ## after optical-flow
                fakeTarget, fakeNoise = self.model(nseq)

                ## loss for denoised
                ld = self.loss[0](fakeTarget, tseq, idx_frame)
                #exit(0)
                ## loss for optical-flow
                lf = self.loss[1](fakeNoise, nseq, idx_frame)

                loss += (ld + lf)

            #print(loss)
            ## Loss STEP 3
            [l.batch_sum() for l in self.loss]
            #exit(0)
            loss /= len(nseqs)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            ## Loss STEP 4
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss[0].display_loss(batch),
                    self.loss[1].display_loss(batch),
                    loss,
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        ## Loss STEP 5
        [l.end_log(len(self.loader_train)) for l in self.loss]
        self.error_last = self.loss[0].log[-1, -1, -1].mean()
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')

        ## Add Log to loss_log matrix
        ## with shape [idx_epoch, number_dataset]
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for nseqs, tseqs, pathname in tqdm(d, ncols=80):
                filename = [fname[-10:] for fname in pathname]
                nseqs, tseqs = self.prepare(nseqs, tseqs)

                self.model.init_hidden()
                #save_list = []
                save_list = {}
                for idx_frame, (nseq, tseq) in enumerate(zip(nseqs, tseqs)):
                    ## fakeTarget for t'th denoised frame
                    ## fakeNoise for (t-1)'th noised frame alignmented
                    ## after optical-flow
                    fakeTarget, fakeNoise = self.model(nseq)


                fakeTarget = utility.quantize(fakeTarget, self.args.rgb_range)

                save_list['Est'] = fakeTarget
                self.ckp.log[-1, idx_data] += utility.calc_psnr(
                    fakeTarget, tseq, self.args.rgb_range, dataset=d
                )
                if self.args.save_gt:
                    save_list['Noise'] = nseq
                    save_list['Target'] = tseq
                    #save_list.extend([nseq, tseq])

                if self.args.save_results:
                    self.ckp.save_results(d, filename[0], save_list)

            self.ckp.log[-1, idx_data] /= len(d)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    d.dataset.name,
                    self.ckp.log[-1, idx_data],
                    best[0][idx_data],
                    best[1][idx_data] + 1
                )
            )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
