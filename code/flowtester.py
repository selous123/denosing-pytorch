import torch
import numpy as np

import utility
from tqdm import tqdm
"""
Modify from tester.py

Plan  :   We hope test function can save and report more results
than the test function (validation) in the Trainer
(test in training process)process


Author:   Selous(Tao Zhang)
Data  :   2020.01.06

Problem: save PSNR for all images in test set or only the mean PSNR?

We calculate the mean PSNR for all images in test set in v0.2.
"""
class FlowTester():
    def __init__(self, args, loader, my_model, ckp):
        self.args = args

        self.ckp = ckp
        self.loader_test = loader.loader_test
        self.model = my_model

        self.error_last = 1e8


    def test(self):
        torch.set_grad_enabled(False)

        self.ckp.write_log('\nTest:')

        ## Add Log to loss_log matrix
        ## with shape [number_frames, number_test_dataset]
        self.ckp.add_log(
            torch.zeros(self.args.n_frames - 1, len(self.loader_test))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for nseqs, tseqs, pathname in tqdm(d, ncols=80):
                filename = [fname[-10:] for fname in pathname]
                nseqs, tseqs = self.prepare(nseqs, tseqs)

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

                    save_list['warped'] = warped_image

                    self.ckp.log[idx_frame, idx_data] += utility.calc_psnr(
                        warped_image, tseqs[idx_frame+1], self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list['Target'] = tseqs[idx_frame+1]
                        #save_list.extend([nseq, tseq])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, idx_frame)

            self.ckp.log[:, idx_data] /= len(d)

            ## output the best PSNR result of the i^dx_data dataset
            best = self.ckp.log.max(0)
            print(self.ckp.log)
            self.ckp.write_log(
                '[{}]\tLast Frame PSNR: {:.3f} (Best: {:.3f} @frame {})'.format(
                    d.dataset.name,
                    self.ckp.log[-1, idx_data],
                    best[0][idx_data],
                    best[1][idx_data]
                )
            )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        #np.save(os.path.join(os.path.save, "test_psnr.npy"), self.ckp.log.detach().cpu().numpy())
        ## 1. save PSNR results.
        ## 2. draw the PSNR plot according to frames.
        # self.ckp.show_test()

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]
