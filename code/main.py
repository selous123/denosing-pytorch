import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
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
            _loss.append(loss.Loss(args, checkpoint))
            _loss.append(loss.Loss(args, checkpoint, ls = args.loss_flow))
        else:
            _loss = None
        #exit(0)
        if not args.test_only:
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()
        else:
            t = Tester(args, loader, _model, checkpoint)
            t.test()

        checkpoint.done()

if __name__ == '__main__':
    main()
