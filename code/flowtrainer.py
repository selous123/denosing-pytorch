
import trainer
import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class FlowTrainer(trainer.Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(FlowTrainer, self).__init__(args, loader, my_model, my_loss, ckp)


    def train(self):
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
            nseqs, tseqs = self.prepare(nseqs, tseqs)
            #print(nseqs.shape)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            loss = 0
            loss_data = 0
            loss_reg = 0

            for idx_frame in range(len(tseqs)-1):

                ## after optical-flow
                input = torch.cat((tseqs[idx_frame], tseqs[idx_frame+1]), dim=1)

                flowmaps = self.model(input)

                if type(flowmaps) in [tuple, list]:
                    l_data = 0
                    l_reg = 0
                    weights = [0.005, 0.01, 0.02, 0.08, 0.32]
                    #weights = [1.0, 1.0, 1.0, 1.0, 1.0]
                    assert len(flowmaps) == len(weights)
                    for flowmap, weight in zip(flowmaps, weights):
                    ## warped result
                        warped_image = utility.warpfunc(tseqs[idx_frame], flowmap)
                        #print(flowmap.max(), flowmap.min())
                        l_data += weight * self.loss[0](warped_image, tseqs[idx_frame+1], idx_frame)
                        if self.args.loss_freg is not None:
                            #loss_reg += weight * self.loss[1](flowmap, None, idx_frame)
                            l_reg += weight * self.loss[1](flowmap, None, idx_frame)
                            #l += loss_reg

                    loss_data += l_data
                    loss_reg += l_reg
                    loss += (l_data + l_reg)

                else:
                    ## loss for optical-flow
                    loss += self.loss[0](warped_image, tseqs[idx_frame+1], idx_frame)

            ## Loss STEP 3


            [l.batch_sum() for l in self.loss]
            # print(loss_data)
            # print(loss_reg)
            # print(self.loss[0].log)
            # print(self.loss[1].log)
            # print(loss,loss_data)
            # exit(0)
            #print(self.loss[0].display_loss(batch))
            #loss /= (len(tseqs) - 1)
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
                    loss,
                    loss_data,
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
        ## with shape [n_frames, 1]
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

                #self.model.init_hidden()
                #save_list = []
                save_list = {}
                for idx_frame in range(len(tseqs)-1):

                    ## after optical-flow
                    input = torch.cat((tseqs[idx_frame], tseqs[idx_frame+1]), dim=1)
                    flowmap = self.model(input)

                    ## save flow map
                    save_list['flow'] = utility.vis_opticalflow(flowmap)
                    ## warped result
                    warped_image = utility.warpfunc(tseqs[idx_frame], flowmap)
                    warped_image = utility.quantize(warped_image, self.args.rgb_range)

                    save_list['source'] = tseqs[idx_frame]
                    save_list['warped'] = warped_image

                    self.ckp.log[-1, idx_frame, idx_data] += utility.calc_psnr(
                        warped_image, tseqs[idx_frame+1], self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list['Target'] = tseqs[idx_frame+1]
                        #save_list.extend([nseq, tseq])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, idx_frame)

            self.ckp.log[-1, :, idx_data] /= len(d)
            print(self.ckp.log[-1, :, idx_data])
            #self.ckp.write_log((self.ckp.log[-1, :, idx_data]))
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
            self.ckp.save(self, epoch, is_best=(best_epoch_idx + 1 == epoch), plot_title='optical-flow')

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
