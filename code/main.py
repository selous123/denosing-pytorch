import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from frtrainer import FRTrainer
from tester import Tester

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = []
        if not args.test_only:
            ## only for optical flow
            if int(args.model_label) == 0:
                _loss.append(loss.Loss(args, checkpoint, ls = args.loss_flow))
                t = FlowTrainer(args, loader, _model, _loss, checkpoint)
            ## only for frame-recurrent
            elif int(args.model_label) == 1:
                    _loss.append(loss.Loss(args, checkpoint))
                    t = FRTrainer(args, loader, _model, _loss, checkpoint)
            ## for frame-recurrent with optical-flow
            elif int(args.model_label) == 2:
                _loss.append(loss.Loss(args, checkpoint))
                _loss.append(loss.Loss(args, checkpoint, ls = args.loss_flow))
                t = Trainer(args, loader, _model, _loss, checkpoint)
            else:
                raise ValueError("args model_label can only equal to 0,1,2")

        else:
            _loss = None
        #exit(0)

        if not args.test_only:
            while not t.terminate():
                t.train()
                t.test()
        else:
            t = Tester(args, loader, _model, checkpoint)
            t.test()

        checkpoint.done()

if __name__ == '__main__':
    main()
