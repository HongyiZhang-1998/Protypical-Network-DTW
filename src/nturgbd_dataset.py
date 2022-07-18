# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import errno
import torch
import os
import pickle
import random
import gl

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}

class NTU_RGBD_Dataset(data.Dataset):

    def __init__(self, mode='train', data_list=None, debug=False, extract_frame=1, transform=None, target_transform=None):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        '''
        super(NTU_RGBD_Dataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform

        CS_PR_path = '/mnt/data1/zhanghongyi/CS_PR/CS_PR_val_test/'

        if gl.dataset == 'ntu120_30':
            path = '/mnt/data1/nturgbd_raw/ntu120/random_data/train_val_test_30'
            segment = 30
        elif gl.dataset == 'ntu120_60':
            path = '/mnt/data1/nturgbd_raw/ntu120/random_data/train_val_test_60'
            segment = 30
        elif gl.dataset == 'ntu120_100':
            path = '/mnt/data1/nturgbd_raw/ntu120/random_data/train_val_test_100'
            segment = 30
        elif gl.dataset == 'kinetics':
            path = '/data/zhanghongyi/data/kinetics-skeleton/select_motions/class_60_num_{}'.format(gl.num)
            segment = 30
        elif gl.dataset == 'erbao':
            path = ''
            segment = 30
        else:
            path = None
            segment = 0

        if mode == 'train':
            data_path = os.path.join(path, 'tr_data.npy')
            label_path = os.path.join(path, 'tr_label.npy')
            num_frame = os.path.join(path, 'tr_frame.npy')
        elif mode == 'val':
            data_path = os.path.join(path, 'val_data.npy')
            label_path = os.path.join(path, 'val_label.npy')
            num_frame = os.path.join(path, 'val_frame.npy')
        else:
            data_path = os.path.join(path, 'test_data.npy')
            label_path = os.path.join(path, 'test_label.npy')
            num_frame = os.path.join(path, 'test_frame.npy')

        self.data, self.label, self.num_frame = np.load(data_path), np.load(label_path), np.load(num_frame)
        # self.sample_name = np.load(sample_name)

        self.sample_name = np.zeros(len(self.data))

        if debug:
            data_len = len(self.label)
            data_len = int(0.1 * data_len)
            self.label = self.label[0:data_len]
            self.data = self.data[0:data_len]
            self.num_frame = self.num_frame[0:data_len]

        if extract_frame == 1:
            self.data, data_idxs = self.extract_frame(self.data, self.num_frame, segment, leave_all_frame=gl.leave_all_frame)
            data_idxs = np.array(data_idxs)
            # self.save_sample(mode, data_idxs)

        # class_path = os.path.join(path, 'train_val_test_class')
        # if not os.path.exists(class_path):
        #     os.makedirs(class_path)
        # file = os.path.join(class_path, '{}.txt'.format(mode))
        # classes = np.unique(self.label)
        # print(classes)
        # np.savetxt(file, classes)
        # print('sample_num in {}'.format(mode), len(self.label))
        # print('sample label:', np.unique(self.label))
        #
        # print('n_class', n_classes)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.label[idx], self.sample_name[idx]

    def __len__(self):
        return len(self.label)

    def extract_frame(self, x, num_frame, segment, leave_all_frame=False):
        n, c, t, v, m = x.shape
        assert n == len(num_frame)

        num_frame = np.array(num_frame)
        step = num_frame // segment
        new_x = []
        idxs = []
        for i in range(n):
            if num_frame[i] < segment:
                new_x.append(np.expand_dims(x[i, :, 0:segment, :, :], 0).reshape(1, c, segment, v, m))
                continue
            idx = [random.randint(j * step[i], (j + 1) * step[i] - 1) for j in range(segment)]
            idxs.append(idx)
            new_x.append(np.expand_dims(x[i, :, idx, :, :], 0).reshape(1, c, segment, v, m))

        new_x = np.concatenate(new_x, 0)
        return new_x, idxs

    def save_sample(self, mode, data_idxs):
        save_path = os.path.join(gl.experiment_root, 'extract_sample')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if mode == 'test':
            for i in range(len(self.sample_name)):
                file_name = self.sample_name[i].split('.')[0]
                dir = os.path.join(save_path, file_name)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                for j in data_idxs[i]:
                    os.system("cp {} {}".format(os.path.join("/home/zhanghongyi/ntu_video", file_name + "_rgb",
                                                             file_name + "_{}.jpg".format(j)), dir))
