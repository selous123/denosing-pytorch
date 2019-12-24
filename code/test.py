import data
from option import args
import time
print(args.data_train)
# trainset = dedata.DeData(args, name=args.data_train[0])
loader = data.Data(args)

print(len(loader.loader_train.dataset))
print(len(loader.loader_train))
# print(len(trainset))
# input, target, filename  =  trainset[0]
# from matplotlib import pyplot as plt
# plt.subplot(121)
# plt.imshow(input[0].permute(1,2,0) / 255.0)
# plt.subplot(122)
# plt.imshow(target[0].permute(1,2,0)/ 255.0)
# plt.show()
# loader.loader_train[0]

stime = time.time()
for batch_idx, data in enumerate(loader.loader_train):
    #print(batch_idx)
    input, target, filenames = data
    #print(filenames)
    print(input.shape)
    print(target.shape)
    break

etime = time.time()

print(etime-stime)
