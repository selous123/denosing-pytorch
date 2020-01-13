import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
from model.filters import GaussianSmoothing
pre_denoise = GaussianSmoothing(channels=3, kernel_size=7, padding=3)
pre_denoise = pre_denoise.cuda()

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
            self.model.model.init_hidden()
            loss_d = 0
            loss_f = 0
            loss_input_data_flow = {}
            loss_input_data_denoise = {}
            for idx_frame, (nseq, tseq) in enumerate(zip(nseqs, tseqs)):
                ## fakeTarget for t'th denoised frame
                ## fakeNoise for (t-1)'th noised frame alignmented
                ## after optical-flow
                fakeTarget = self.model(nseq)
                ## loss for denoised
                loss_input_data_denoise['est'] = fakeTarget
                loss_input_data_denoise['target'] = tseq
                ld = self.loss[0](loss_input_data_denoise, idx_frame)

                loss_d += ld
                #exit(0)

                ## loss for optical-flow
                # if idx_frame == 0:
                #     ## we should not compute the optical-flow loss
                #     ## for 0'th frame with full-zero matrix.
                #     lf = 0
                # else:
                ## Get The pre-denoised output for calculate
                ## The Loss for optical-flow task
                ## t1: for t-1'th frame and t2: for t'th frame
                lf = 0
                if idx_frame != 0:
                    t1, t2 = self.model.model.get_optical_input().chunk(2, dim=1)
                    flowmaps = self.model.model.get_opticalflow_map()
                    if type(flowmaps) in [tuple, list]:
                        weights = [0.005, 0.01, 0.02, 0.08, 0.32]
                        #weights = [1.0, 1.0, 1.0, 1.0, 1.0]
                        assert len(flowmaps) == len(weights)
                        for flowmap, weight in zip(flowmaps, weights):
                        ## warped result
                            warped_image = utility.warpfunc(t1, flowmap)
                            #print(flowmap.max(), flowmap.min())
                            ## L1 (est, target)
                            loss_input_data_flow['est'] = warped_image
                            loss_input_data_flow['target'] = t2
                            ##TVL1(flowmap)
                            loss_input_data_flow['flowmap'] = flowmap

                            #l_data += weight * self.loss[0](warped_image, tseqs[idx_frame+1], idx_frame)
                            lf += weight * self.loss[1](loss_input_data_flow, idx_frame)

                    else:
                        warped_image = utility.warpfunc(t1, flowmaps)
                        ## loss for optical-flow
                        lf = self.loss[1](warped_image, t2, idx_frame)
                    # print("l0:", self.loss[0].display_loss(batch))
                    # print("l1:", self.loss[1].display_loss(batch))
                loss_f += lf
            ## Loss STEP 3
            [l.batch_sum() for l in self.loss]
            #exit(0)
            loss_d = loss_d / len(nseqs)
            loss = loss_d + loss_f
            #loss /= len(nseqs)
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
                self.ckp.write_log('[{}/{}]\t{}\t{}\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss[0].display_loss(batch),
                    self.loss[1].display_loss(batch),
                    loss_d,
                    loss_f,
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
            torch.zeros(1, self.args.n_frames, len(self.loader_test))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for nseqs, tseqs, pathname in tqdm(d, ncols=80):
                filename = [fname[-10:] for fname in pathname]
                nseqs, tseqs = self.prepare(nseqs, tseqs)

                self.model.model.init_hidden()
                #save_list = []
                save_list = {}
                for idx_frame, (nseq, tseq) in enumerate(zip(nseqs, tseqs)):
                    ## fakeTarget for t'th denoised frame
                    ## fakeNoise for (t-1)'th noised frame alignmented
                    ## after optical-flow
                    fakeTarget = self.model(nseq)
                    fakeTarget = utility.quantize(fakeTarget, self.args.rgb_range)

                    save_list['Est'] = fakeTarget ## t estimate frame
                    self.ckp.log[-1, idx_frame, idx_data] += utility.calc_psnr(
                        fakeTarget, tseq, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list['Noise'] = nseq ## t noised frame
                        save_list['Target'] = tseq ## t target frame
                        #save_list.extend([nseq, tseq])

                    if self.args.save_of:
                        ## optical-flow
                        t1, t2 = self.model.model.get_optical_input().chunk(2, dim=1)
                        flowmap = self.model.model.get_opticalflow_map()
                        save_list['flow'] = utility.vis_opticalflow(flowmap) ## t-1 with t
                        save_list['source'] = t1 ## pre-denoised t-1 frame noise input
                        save_list['noise-warped'] = utility.warpfunc(t1, flowmap) ## t-1 noised warped
                        save_list['FTarget'] = t2 ## pre-denoised noise input
                        prest_warped = self.model.model.get_warpresult()
                        save_list['prest-warped'] = utility.quantize(prest_warped, self.args.rgb_range) ## t-1 estimated warped

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, idx_frame)

            self.ckp.log[-1, : , idx_data] /= len(d)
            self.ckp.write_log('{}\'th epoch, PSNR via frame: {}'.format(epoch, self.ckp.log[-1, :, idx_data]))
            epoch_best, epoch_idx = self.ckp.log.max(0)
            best, frame_idx = epoch_best.max(0)
            best_frame_idx = frame_idx[idx_data]
            best_epoch_idx = epoch_idx[best_frame_idx, idx_data]
            self.ckp.write_log(
                '[{}]\t PSNR: {:.3f} (Best: {:.3f} @frame {} @epoch {})'.format(
                    d.dataset.name,
                    self.ckp.log[-1, :, idx_data].mean(),
                    best[0],
                    best_frame_idx + 1,
                    best_epoch_idx + 1
                )
            )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best_epoch_idx + 1 == epoch))
            self.ckp.plot_psnr(self.args.n_frames, mean = False, dimension = 0, name = 'idx_frame')

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            tensor = tensor.to(device)
            return tensor



        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
