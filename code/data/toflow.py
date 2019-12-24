from data import dedata

class ToFlow(dedata.DeData):
    def __init__(self, args, name='ToFlow', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(ToFlow, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        dir_tseqs, dir_nseqs = super(ToFlow, self)._scan()

        dir_tseqs = dir_tseqs[self.begin - 1:self.end]
        dir_nseqs = dir_nseqs[self.begin - 1:self.end]
        return dir_tseqs, dir_nseqs
