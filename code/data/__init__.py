from importlib import import_module
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
## concat dataset for multi training dataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

## permute data
## from B * n_frames * CHW
## to n_frames * B * CHW
class loader_wrapper(object):
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset

    def __iter__(self):
        for data in self.loader:
            yield data[0].permute(1, 0, 2, 3, 4), data[1].permute(1, 0, 2, 3, 4), data[2]

    def __len__(self):
        return len(self.loader)

class Data:
    def __init__(self, args):
        self.loader_train = None
        ## only return one concated dataset if multi-training dataset
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = loader_wrapper(dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            ))
        ## Return a list of test dataset, if multi-test dataset
        self.loader_test = []
        for d in args.data_test:
            ## These are the standard testset for super-resolution
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                loader_wrapper(dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                ))
            )
