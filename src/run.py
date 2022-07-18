import os
from tqdm import tqdm
import time
from utils import getAvaliableDevice
import random

os.chdir('.')

# #dataset='Office-31'
datasets = ["kinetics"]  # ntu120_100 ntu120_30

left = False

seeds = [1997, 1998, 1999]

backbones = ["st_gcn"]


py = "train.py "
lr= 0.001
regs = [0, 0.1]
use_attentions = [0, 1]
dtws = [1]
shot = 1
# epss = [1, 10, 100]
eps = 1
# xis = [10, 100]
xi = 20
# alphas = [1, 10]
alpha = 1
vats = [0, 1]
nums = [30, 60]
# leave_frames = [0, 1]

nowtime = time.strftime("%Y_%m_%d", time.localtime())

# nowtime="2021_08_28"+dataset
total = len(backbones) * len(seeds) * len(vats) * len(vats) * len(datasets)
cnt = 0

for backbone in backbones:
    for num in nums:
        for dataset in datasets:
            for seed in seeds:
                for dtw in dtws:
                    for vat in vats:
                        for reg in regs:
                            for use_attention in use_attentions:
                                gpu = getAvaliableDevice(gpu=[3, 4], left=left, min_mem=24000)
                                cnt += 1
                                command = "python " + py + "--backbone {} --experiment_root {} --device {} -seed {} -lr {} --dtw {} --vat {} --alpha {} --xi {} --eps {} --dataset {} --num_support_tr {} --reg {} --use_attention {} --num {}&".format(
                                    backbone, os.path.join(dataset + 'num_{}'.format(num) + '_' + backbone, "dtw_{}".format(dtw), 'vat_{}'.format(vat), 'reg_{}'.format(reg), 'attention_{}'.format(use_attention), 'seed_{}'.format(seed)), gpu
                                    , seed, lr, dtw, vat, alpha, xi, eps, dataset, shot, reg, use_attention, num)
                                print("***************************************************training on {}/{}:".format(cnt, total) + command)
                                os.system(command)

                                time.sleep(100)



