import torch

import utility
import data
import model
import loss
from option import args
from flowtrainer import FlowTrainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = []
        if not args.test_only:
            _loss.append(loss.Loss(args, checkpoint, ls = args.loss_flow))
        else:
            _loss = None
        #exit(0)

        t = FlowTrainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

if __name__ == '__main__':
    main()
