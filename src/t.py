
import random
import numpy as np
import torch

def get_train_val_test_classes():
    classes = [i for i in range(60)]
    random.shuffle(classes)

    train_class, val_class, test_class = 40, 10, 10
    train_class_name, val_class_name, test_class_name = np.array(classes[:train_class]), np.array(classes[train_class: train_class + val_class]),\
                                                        np.array(classes[train_class + val_class:])
    train_class_name, val_class_name, test_class_name = classes[:train_class], classes[train_class: train_class + val_class], classes[train_class + val_class:]
    arr = []
    arr.append(train_class_name)
    arr.append(val_class_name)
    arr.append(test_class_name)

    arr = np.array(classes)
    print(arr)
    np.save('/home/zhanghongyi/Prototypical-Networks-for-Few-shot-Learning-PyTorch/src/train_val_test_classes.npy', arr)

# get_train_val_test_classes()
def test_split_train_val():
    class_path = '/home/zhanghongyi/Prototypical-Networks-for-Few-shot-Learning-PyTorch/src/train_val_test_classes.npy'
    x = np.load(class_path)
    print(x)
# test_split_train_val()

def test_expand_repeat():
    x = torch.ones((2, 2))
    y = x.unsqueeze(1).repeat(1, 3, 1)
    print(y.shape)
    x[1,0] = 5
    y[0, 1, 0] =10
    print(x)
    print(y)
    print(y[1, 1, 0])
#
# test_expand_repeat()

from torch.nn import functional as F

def test_split_T():
    x = torch.randn(50, 2, 256, 25)
    n_segment = 2
    # bottleneck = self.conv1(x)  # nt, c//r, h, w
    # bottleneck = self.bn1(bottleneck)  # nt, c//r, h, w
    bottleneck = x
    # t feature
    # reshape_bottleneck = bottleneck.view((-1, n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
    reshape_bottleneck = x
    t_fea, __ = reshape_bottleneck.split([n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w

    # apply transformation conv to t+1 feature
    # conv_bottleneck = conv2(bottleneck)  # nt, c//r, h, w
    conv_bottleneck = bottleneck
    # reshape fea: n, t, c//r, h, w
    # reshape_conv_bottleneck = conv_bottleneck.view((-1, n_segment) + conv_bottleneck.size()[1:])
    reshape_conv_bottleneck = x
    __, tPlusone_fea = reshape_conv_bottleneck.split([1, n_segment - 1], dim=1)  # n, t-1, c//r, h, w
    print('tPlusone_fea', tPlusone_fea.size())
    # motion fea = t+1_fea - t_fea
    # pad the last timestamp
    diff_fea = tPlusone_fea - t_fea  # n, t-1, c//r, h, w
    print('diff_fea.size ', diff_fea.size())
    pad = (0, 0, 0, 0, 0, 1)
    diff_fea_pluszero = F.pad(diff_fea, pad, mode="constant", value=0)  # n, t, c//r, h, w
    print('diff_fea_pluszero.size ', diff_fea_pluszero.size())

    diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])
    print('last.size ', diff_fea_pluszero.size())

# test_split_T()

def test_idx():
    a = torch.randn(5, 5)
    idx = [1,2,3]
    x = a[idx]
    y = x[:, idx]
    print(a, '\n', y)
# test_idx()

def sim(x, y):
    """Compute the dot product similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise dot product similarity.
    """
    return torch.mm(x, torch.transpose(y, 1, 0))

def get_local_sim(a, la, ra, ll, rl, body):
    '''

    :param a: all skeleton sim
    :param la: left arm skeleton idx
    :param ra: right arm
    :param ll: left leg
    :param rl: right leg
    :param body: body
    :return: a_la, a_ra, a_ll, a_rl, a_body
    '''
    a_la_t = a[la]
    a_la = a_la_t[:, la]
    a_ra_t = a[ra]
    a_ra = a_ra_t[:, ra]
    a_ll_t = a[ll]
    a_ll = a_ll_t[:, ll]
    a_rl_t = a[rl]
    a_rl = a_rl_t[:, rl]
    a_body_t = a[body]
    a_body = a_body_t[:, body]
    return a_la, a_ra, a_ll, a_rl, a_body

def compute_local_cross_attention(a, x, y):

    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i

    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y

def comput_cross_attention():
    x = torch.randn(25, 256)
    y = torch.randn(25, 256)
    v, c = x.size()

    x_norm = x / (x.norm(dim=0, keepdim=True) + 1e-8)
    y_norm = y / (y.norm(dim=0, keepdim=True) + 1e-8)

    left_arm_node = torch.tensor([9, 10, 11, 23, 24])
    right_arm_node = torch.tensor([5, 6, 7, 21, 22])
    left_leg_node = torch.tensor([17, 18, 19])
    right_leg_node = torch.tensor([13, 14, 15])
    body = torch.tensor([0, 1, 2, 3, 4, 8, 12, 16, 20])

    a = sim(x_norm, y_norm)

    a_la, a_ra, a_ll, a_rl, a_body = get_local_sim(a, left_arm_node, right_arm_node, left_leg_node, right_leg_node, body)

    attention_x = torch.zeros(v, c)
    attention_y = torch.zeros(v, c)
    attention_x[left_arm_node], attention_y[left_arm_node] = compute_local_cross_attention(a_la, x[left_arm_node], y[left_arm_node])
    attention_x[right_arm_node], attention_y[right_arm_node] = compute_local_cross_attention(a_ra, x[right_arm_node], y[right_arm_node])
    attention_x[left_leg_node], attention_y[left_leg_node] = compute_local_cross_attention(a_ll, x[left_leg_node],y[left_leg_node])
    attention_x[right_leg_node], attention_y[right_leg_node] = compute_local_cross_attention(a_rl, x[right_leg_node],y[right_leg_node])
    attention_x[body], attention_y[body] = compute_local_cross_attention(a_body, x[body], y[body])
    return a, attention_x, attention_y

# comput_cross_attention()

def read_file():
    with open('../gmn_not_aggregate_edge/p5_norm_attention_wsq5510_k1_xsub/trace.txt', 'r', encoding='UTF-8') as f:

        string = f.readline()
        while string:
            print(string)
            string = f.readline()
# read_file()

import os
import numpy as np
def write_tensor():
    for i in range(12500):
        if  random.randint(1, 100) <= 1:
            a, ax, ay = comput_cross_attention()
            np.savetxt(os.path.join('./a', 'a_{}.txt'.format(i)), a.numpy(), fmt='%.2f')
#
# write_tensor()

def cosine_similarity(x, y):
    """Compute the cosine similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise cosine similarity.
    """
    print(torch.sum(x ** 2))
    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), torch.tensor(1e-12))))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), torch.tensor(1e-12))))
    return torch.mm(x, torch.transpose(y, 1, 0))


# path = '/mnt/data1/nturgbd_raw/normalize_RGB_XY_ntu/xsub/train_data_joint.npy'
# import numpy as np
# data = np.load(path)
# print(data)

# path = '/mnt/data1/zhanghongyi/CS_PR/CS'
# list = os.listdir(path)
# for i in range(len(list)):
#     list[i] = list[i].split('utput')[-1].split('_S')[0]
# print(list)
# print(len(np.unique(list)))
# import numpy as np
# arr = [1, 2, 3, 3, 4, 1, 2, 2]
# classes, counts = np.unique(arr, return_counts=True)
# print(classes, counts)
#
# classes , count= np.unique([0, 1, 0,0,1,1], return_counts=True)
# c_idxs = torch.randperm(len(classes))[:2]
# print(c_idxs)
# for i, c in enumerate(classes[c_idxs]):
#     label_idx = torch.arange(len(classes)).long()[classes == c].item()
#     print(label_idx, '--', i)
#     sample_idxs = torch.randperm(10)[:5]
#     print(sample_idxs)
#
# x = torch.tensor([1, 2, 3])
# y = torch.tensor([2, 3])
#
#
# for i in x :
#     print(i)
#
# log = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(log.size())
# label = torch.tensor([0, 0, 1]).unsqueeze(1)
# x = log.gather(1, label)
# print(x)

#norm and cosine
# x = torch.ones(2, 3)
# y = torch.randn(2, 3)
# print(x)
# print(y)
# print('x * y', torch.mm(x, torch.transpose(y, 1, 0)))
# x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
# print('x_norm:\n',x_norm)
# y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-8)
# print('y_norm:\n', y_norm)
#
# print('x * y norm:', torch.mm(x_norm, torch.transpose(y_norm, 1, 0)))

#图为邻接矩阵形式
def get_distance_matrix(threshold):
    import math
    NUMBER = 25
    INF = 999
    SIZE = 26
    Graph_Matrix = [[0] * SIZE for row in range(SIZE)]
    distance = [[0]*SIZE for row in range(SIZE)]
    def BuildGraph_Matrix(Path_Cost):
        for i in range(1, SIZE):
            for j in range(1, SIZE):
                if i == j:
                    Graph_Matrix[i][j]=0
                else:
                    Graph_Matrix[i][j]=INF
        i = 0
        while i < len(Path_Cost):
            Start_Point = Path_Cost[i][0]
            End_Point = Path_Cost[i][1]
            Graph_Matrix[Start_Point][End_Point] = 1
            Graph_Matrix[End_Point][Start_Point] = 1
            i += 1

    def Dijkstra(v0, vertex_total, INF=999):
        book = set()
        minv = v0   # 源顶点到其余各顶点的初始路程
        dis = dict((k,INF) for k in range(1,vertex_total+1))
        dis[v0] = 0
        while len(book) < vertex_total:
            book.add(minv)                                  # 确定当期顶点的距离
            for w in range(1, vertex_total + 1):                               # 以当前点的中心向外扩散
                if dis[minv] + Graph_Matrix[minv][w] < dis[w]:         # 如果从当前点扩展到某一点的距离小与已知最短距离
                    dis[w] = dis[minv] + Graph_Matrix[minv][w]         # 对已知距离进行更新
            new = INF                                       # 从剩下的未确定点中选择最小距离点作为新的扩散点
            for v in dis.keys():
                if v in book: continue
                if dis[v] <= new:
                    new = dis[v]
                    minv = v
        for j in dis.keys():
            if dis[j] == 0:
                distance[v0][j] = dis[j]  # v0 = j
            elif dis[j] <= threshold:
                distance[v0][j] = -1 / (dis[j] ** 2 + 1)
            elif dis[j] > threshold:
                distance[v0][j] = dis[j] ** 2 + 1
        del (dis)


    path = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),(10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                                  (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),(20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
    BuildGraph_Matrix(path)

    for i in range(1, SIZE):
        Dijkstra(i, NUMBER)

    distance = np.array(distance)
    distance = distance[1:, 1:]

    x = np.savetxt('threshold_{}_reg_weight.txt'.format(threshold), distance, fmt='%.6f')

#get_distance_matrix(threshold=3)

# x = torch.zeros(25, 64)
# y = torch.ones(25, 64)
# z = y + 1
# print(torch.mul(z, y))
# n = x.size(0)
# m = y.size(0)
# d = x.size(1)
# assert d == y.size(1)
#
# x = x.unsqueeze(1).expand(n, m, d)
# y = y.unsqueeze(0).expand(n, m, d)
#
# dist = torch.pow(x - y, 2)
#
# print(dist.sum(2).size())
#
# x = torch.tensor(np.loadtxt('reg_weight.txt'))
#
# dist = torch.ones(25, 25) * 10
# from torch.nn import functional as F
# nums = torch.mul(x, dist).sum()
# x = torch.tensor([ torch.tensor(123.456)for i in range(250)]).view(50, 5)
# print(x)
# y = F.log_softmax(x, dim=1)
# print(y)
#
# x = torch.randn(5, 64)
# y = x
#
# x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
# y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-8)
# cos = torch.mm(x_norm, torch.transpose(y_norm, 1, 0))
# print(cos)
#
# def sdtw():
#     from soft_dtw import SoftDTW
#     import gl
#
#     batch_size, len_x, len_y, dims = 8, 15, 12, 5
#     x = torch.ones((batch_size, len_x, dims), requires_grad=True).to(gl.device)
#     y = torch.ones((batch_size, len_y, dims)).to(gl.device)
#
#     # Create the "criterion" object
#     sdtw = SoftDTW(gamma=0.1, normalize=True)
#
#     # Compute the loss value
#     loss = sdtw(x, y)
#     print(loss)

# x = torch.randn(1, 2, 3, 4)
# print(x)
# y = x
# x = x.reshape(1, 3, 8)
# print(x)
# y = y.permute(0, 2, 1, 3).contiguous()
# y = y.reshape(1, 3, 8)
# print(y)

# x = torch.zeros(13, 64)
# print(min(x.shape[0], x.shape[1]))

# import pickle
# import random
#
# label_path = '/mnt/data1/kinetics-skeleton/train_label.pkl'
# data_path = '/mnt/data1/kinetics-skeleton/train_data.npy'
#
# try:
#     with open(label_path) as f:
#         sample_name, label = pickle.load(f)
# except:
#     # for pickle file from python2
#     with open(label_path, 'rb') as f:
#         sample_name, label = pickle.load(f, encoding='latin1')
# data = np.load(data_path)

# print(data.shape)
# print(label)

# n_classes = len(np.unique(label))

#
# classes = [i for i in range(120)]
# random.shuffle(classes)
# import numpy as np
# import torch
# x = torch.arange(3)
# y = []
# y.append(torch.zeros(5))
# y.append(torch.ones(5))
# y.append(torch.ones(5) * 2)
# y = torch.cat(y)
# print(y)
# x = x.unsqueeze(1).expand(3, 5).reshape(15)
# y = y.unsqueeze(0).expand(3, 15).reshape(45)


# np.save('train_val_test120.npy', classes)
x = torch.randn(2, 3, 64)
y = torch.randn(2, 3, 64)
def calc_distance_matrix(x, y):
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)

    # x = self.attention(x)
    # y = self.attention(y)

    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    print('x - y ', x - y)

    # x = x.reshape(-1, d)
    # y = y.reshape(-1, d)
    # x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    # y = y / (y.norm(dim=1, keepdim=True) + 1e-8)
    # cos = (x * y)
    # sum_cos = cos.sum(1)
    # e_cos = torch.exp(sum_cos)
    # e_cos = e_cos.view(-1, n, m)
    # dist = e_cos

    dist = torch.pow(x - y, 2).sum(3)

    return dist

# x = torch.randn(2, 2, 2)
# print(x)
# for i in range(2):
#     print(torch.diagonal(x[i]))
#
# print(torch.diagonal(x, dim1=1, dim2=2).sum(1))
# print(torch.diagonal(x, dim1=1, dim2=2).sum())

# x = torch.tensor([66, 94, 96, 98, 114])
# y = x.unsqueeze(1).expand(5, 5).reshape(25)
# print(y)
# x1 = x.unsqueeze(1).expand(5, 25).reshape(125)
# y1 = y.unsqueeze(0).expand(5, 25).reshape(125)
# print(x1)
# print(y1)

def  save_sample_name():
    import pickle
    path = '/mnt/data1/nturgbd_raw/ntu120/random_data/train_val_test_100/train_val_test_class/'

    train = np.loadtxt(os.path.join(path, 'train.txt'))
    train = train.astype(dtype=int)
    val = np.loadtxt(os.path.join(path, 'val.txt'))
    val = val.astype(dtype=int)
    test = np.loadtxt(os.path.join(path, 'test.txt'))
    test = test.astype(dtype=int)

    origin_data_path = '/mnt/data1/nturgbd_raw/ntu120/xsub'
    data_path = os.path.join(origin_data_path, 'train_data_joint.npy')
    label_path = os.path.join(origin_data_path, 'train_label.pkl')

    try:
        with open(label_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')

    sample_name_train, sample_name_val, sample_name_test = [], [], []
    num_class = np.zeros(120)
    num = 100

    for i in range(len(label)):

        num_class[label[i]] += 1

        if label[i] in train:
            if num_class[label[i]] >= num:
                continue
            sample_name_train.append(sample_name[i])
        elif label[i] in val:
            if num_class[label[i]] >= num:
                continue
            sample_name_val.append(sample_name[i])
        elif label[i] in test:
            if num_class[label[i]] >= num:
                continue
            sample_name_test.append(sample_name[i])

    print((sample_name_test))
    save_path = '/mnt/data1/nturgbd_raw/ntu120/random_data/train_val_test_{}'.format(num)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    np.save(os.path.join(save_path, 'sample_name_train.npy'), sample_name_train)
    np.save(os.path.join(save_path, 'sample_name_val.npy'), sample_name_val)
    np.save(os.path.join(save_path, 'sample_name_test.npy'), sample_name_test)

import torch

import random
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
import os

def plot_matrix(save_path, matrix):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    p = sns.heatmap(matrix, annot=False, fmt='.0f', ax=ax, vmin=0, vmax=1)
    ax.set_title('heat map')
    ax.set_xlabel('')
    ax.set_ylabel('')
    s = p.get_figure()
    s.savefig(save_path, dpi=600, bbox_inches='tight')

# x = torch.randn(5, 5)
# print(x)
# softmax_x =torch.nn.Softmax(dim=-1)(x)
# print(softmax_x)
# file_path = '/mnt/data1/zhanghongyi/CS_PR/CS'
# file_path2 = '/mnt/data1/zhanghongyi/CS_PR/PR'
# lists = os.listdir(file_path)
# print('cs len', len(lists))
# lists = os.listdir(file_path2)
# print('pr len', len(lists))
# path = "/mnt/data1/zhanghongyi/CS_PR/test_equal_train/CS+Output2019-07-1210.19.18_S0.json"
# import json
# with open(path, 'r') as f:
#     x = json.load(f)
#     print(x['category'])
#     print(np.array(x['data']).shape)

# x = torch.arange(2)
# print(x.unsqueeze(0).expand(2, 2).reshape(-1))
# print(x.unsqueeze(1).expand(2, 2).reshape(-1).view(2, 2))


# path = '/mnt/data1/nturgbd_raw/ntu120/random_data/train_val_test_30'
#
# import numpy as np
#
# trlabel = np.load(os.path.join(path, 'train_label.npy'))
# vlabel = np.load(os.path.join(path, 'val_label.npy'))
# tlabel = np.load(os.path.join(path, 'test_label.npy'))
#
# tr_class = np.unique(trlabel)
# np.savetxt(os.path.join(path, 'tr_label.txt'), np.unique(trlabel), fmt='%d')
# np.savetxt(os.path.join(path, 'val_label.txt'), np.unique(vlabel), fmt='%d')
# np.savetxt(os.path.join(path, 'test_label.txt'), np.unique(tlabel), fmt='%d')

# tr_path = os.path.join(path, 'tr_label.txt')
# tr = np.loadtxt(tr_path).astype(int)
# print((tr[0]))
#
# data_path = '/mnt/data1/nturgbd_raw/ntu120/xsub'
# import pickle
# label_path = os.path.join(data_path, 'train_label.pkl')
#
# try:
#     with open(label_path) as f:
#         sample_name, label = pickle.load(f)
# except:
#     # for pickle file from python2
#     with open(label_path, 'rb') as f:
#         sample_name, label = pickle.load(f, encoding='latin1')
#
#
# print(label[0])
# print((label[0] - tr[0]))
# import json
# import numpy as np
# path = '/mnt/data1/zhanghongyi/CS_PR/new_split'
# train_path = os.path.join(path, 'train_1')
# test_path = os.path.join(path, 'test_1')
# output_path = os.path.join(path, 'train_test_1')
#
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
#
# for i in [train_path, test_path]:
#     lists = os.listdir(i)
#     if i == train_path :
#         data_name = os.path.join(output_path, 'train_data.npy')
#         label_name = os.path.join(output_path, 'train_label.npy')
#         num_frame = os.path.join(output_path, 'train_num_frame.npy')
#     else:
#         data_name = os.path.join(output_path, 'test_data.npy')
#         label_name = os.path.join(output_path, 'test_label.npy')
#         num_frame = os.path.join(output_path, 'test_num_frame.npy')
#     data, label = [], []
#     for j in lists:
#         with open(os.path.join(i, j), 'r') as f:
#             jsons = json.load(f)
#             x = np.expand_dims(np.array(jsons['data']),0)
#             # 1 300 25 3 1
#             _, t, v, c, m = x.shape
#             x = x.reshape(1, t, c, v, m)
#             x = x.reshape(1, c, t, v, m)
#
#             data.append(x)
#             label.append(jsons['category'])
#
#     data = np.concatenate(data)
#     label = np.array(label)
#     np.save((data_name), data)
#     np.save((label_name), label)
#     np.save(num_frame, np.ones(len(label)) * 300)

import numpy as np
path = '/mnt/data1/nturgbd_raw/ntu120/random_data/train_val_test_60/'
import os
# paths = []
#
# paths.append(os.path.join(path, 'train_label.npy'))
# paths.append(os.path.join(path, 'val_label.npy'))
# paths.append(os.path.join(path, 'test_label.npy'))
#
# paths2 = []
path2 = '/mnt/data1/kinetics-skeleton/random_data/train_val_test_100/'
# paths2.append(os.path.join(path2, 'train_label.npy'))
# paths2.append(os.path.join(path2, 'val_label.npy'))
# paths2.append(os.path.join(path2, 'test_label.npy'))
#
# for i in paths:
#     label = np.unique(np.load(i))
#     np.savetxt(i.split('.')[0] + '.txt', label, fmt='%d')
#     print(label)
#
# for i in paths2:
#     label = np.unique(np.load(i))
#     np.savetxt(i.split('.')[0] + '.txt', label, fmt='%d')
#     print(label)

# ntu_tr = np.loadtxt(os.path.join(path, 'train_label.txt'))
# ntu_val = np.loadtxt(os.path.join(path, 'val_label.txt'))
# ntu_test = np.loadtxt(os.path.join(path, 'test_label.txt'))
#
# print('ntu tr ', ntu_tr)
# print('ntu val', ntu_val)
# print('ntu test', ntu_test)
#
# ntu_tr = np.loadtxt(os.path.join(path2, 'train_label.txt'))
# ntu_val = np.loadtxt(os.path.join(path2, 'val_label.txt'))
# ntu_test = np.loadtxt(os.path.join(path2, 'test_label.txt'))
#
# print('kinetics tr ', ntu_tr)
# print('kinetics val', ntu_val)
# print('kinetics test', ntu_test)


# label_path = '/mnt/data1/nturgbd_raw/ntu120/random_data/train_val_test_100/train_label.npy'
# label = np.load(label_path)
#
# print(label, len(label), type(label))

# path = '/mnt/data1/zhanghongyi/random_30_except_test/pickle_file/data_joint.npy'
#
# data = np.load(path)
# print(data.shape)

# path = '/home/zhanghongyi/ntu_video'
# sk_path = "/mnt/data1/zhanghongyi/random_30_except_test/"
# import os
# video_list = os.listdir(path)
# file = []
# import numpy as np
#
# import torch
#
# print(torch.randn(5, 5))

# for i in video_list:
#     if i.find('avi') < 0:
#         continue
#     file.append(i.split('.avi')[0].split('_rgb')[0] + '.skeleton')
#
# sample_name = np.load('/home/zhanghongyi/sample_name.npy')
# data = np.load('/mnt/data1/zhanghongyi/random_30_except_test/pickle_file/data_joint.npy')
# label = np.load('/mnt/data1/zhanghongyi/random_30_except_test/pickle_file/label.npy')
# num_frame = np.load('/mnt/data1/zhanghongyi/random_30_except_test/pickle_file/num_frame.npy')
#
# print(data.shape)
# print(len(sample_name))
# idx = []
# for i in file:
#     for j in range(len(sample_name)):
#         if i == sample_name[j]:
#             idx.append(j)
#             break
#
# print(idx)
#
# data2 = data[idx]
# label2 = label[idx]
# num_frame2 = num_frame[idx]
# sample_name2 = sample_name[idx]
#
# np.save(os.path.join(path, 'data_joint.npy'), data2)
# np.save(os.path.join(path, 'label.npy'), label2)
# np.save(os.path.join(path, 'num_frame.npy'), num_frame2)
# np.save(os.path.join(path, 'sample_name.npy'), sample_name2)
import os
import numpy as np

body_hands = [2, 4, 11, 21, 58, 77, 82, 98, 107, 128, 150, 151, 257, 383, 289, 290, 305, 334, 335, 344]

athletics_jumping = [152, 161, 183, 208, 254, 368]
some_dance = [35, 85, 278, 343]


extract_label = []
extract_label.append(body_hands)
extract_label.append(athletics_jumping)
extract_label.append(some_dance)
extract_label = np.concatenate(extract_label)
np.random.shuffle(extract_label)

hand = [82, 98, 383]
head_mouse = [13, 17, 27, 38, 317, 320, 393]
eat_drink = [92, 101, 111, 115, 204, 354]
makeup = [5, 109, 127, 10]
arts_crafts = [25, 36, 99, 391]
athletics_throw_launch = [6, 299]
cloth = [97, 132, 187, 371]

x = [i + 1 for i in range(400)]
rest_set = set(x) - set(extract_label)
rest_set = np.array(list(rest_set))
np.random.shuffle(rest_set)
rest_set = rest_set[:30]

tr_class = extract_label[:20]
val_class = extract_label[20:25]
test_class = extract_label[25:]

def load_data(save_path, train_class_name, val_class_name, test_class_name, num):
    path = '/mnt/data1/kinetics-skeleton'
    data = np.load(os.path.join(path, 'train_data.npy'))
    import pickle
    label_path = os.path.join(path, 'train_label.pkl')
    try:
        with open(label_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')
    num_class = np.zeros(425)
    num_frame = np.ones(len(label)) * 300

    train_data, val_data, test_data = [], [], []
    train_label, val_label, test_label = [], [], []
    train_num_frame, val_num_frame, test_num_frame = [], [], []
    for i in range(len(label)):

        num_class[label[i]] += 1

        if label[i] in train_class_name:
            if num_class[label[i]] > num:
                continue
            train_data.append(np.expand_dims(data[i], axis=0))
            train_label.append(label[i])
            train_num_frame.append(num_frame[i])
        elif label[i] in val_class_name:
            if num_class[label[i]] > num:
                continue
            val_data.append(np.expand_dims(data[i], axis=0))
            val_label.append(label[i])
            val_num_frame.append(num_frame[i])
        elif label[i] in test_class_name:
            if num_class[label[i]] > num:
                continue
            test_data.append(np.expand_dims(data[i], axis=0))
            test_label.append(label[i])
            test_num_frame.append(num_frame[i])
    train_data, val_data, test_data = np.concatenate(train_data, 0), np.concatenate(val_data, 0), np.concatenate(test_data, 0)



    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, 'train_data.npy'), train_data)
    np.save(os.path.join(save_path, 'train_label.npy'), train_label)
    np.save(os.path.join(save_path, 'train_frame.npy'), train_num_frame)
    np.save(os.path.join(save_path, 'val_data.npy'), val_data)
    np.save(os.path.join(save_path, 'val_label.npy'), val_label)
    np.save(os.path.join(save_path, 'val_frame.npy'), val_num_frame)
    np.save(os.path.join(save_path, 'test_data.npy'), test_data)
    np.save(os.path.join(save_path, 'test_label.npy'), test_label)
    np.save(os.path.join(save_path, 'test_frame.npy'), test_num_frame)

    data_list = [train_data, train_label, np.array(train_num_frame), val_data, val_label, np.array(val_num_frame), test_data, test_label, np.array(test_num_frame)]

    return data_list

save_path = '/mnt/data1/kinetics-skeleton/body_relate/class_30_num_30'
# load_data(save_path, tr_class, val_class, test_class, 100)

import numpy as np
import torch
class MaxPooling2D:
    def __init__(self, kernel_size=(2, 2), stride=2):
        self.kernel_size = kernel_size
        self.w_height = kernel_size[0]
        self.w_width = kernel_size[1]

        self.stride = stride

        self.x = None
        self.in_height = None
        self.in_width = None

        self.out_height = None
        self.out_width = None

        self.arg_max = None

    def __call__(self, x):
        self.x = x
        self.in_height = np.shape(x)[0]
        self.in_width = np.shape(x)[1]

        self.out_height = int((self.in_height - self.w_height) / self.stride) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride) + 1

        out = np.zeros((self.out_height, self.out_width))
        self.arg_max = np.zeros_like(out, dtype=np.int32)

        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out[i, j] = np.max(x[start_i: end_i, start_j: end_j])
                self.arg_max[i, j] = np.argmax(x[start_i: end_i, start_j: end_j])
        self.arg_max = self.arg_max
        return out

    def backward(self, d_loss):
        dx = np.zeros_like(self.x)
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                index = np.unravel_index(self.arg_max[i, j], self.kernel_size)
                dx[start_i:end_i, start_j:end_j][index] = d_loss[i, j] #
        return dx



# np.set_printoptions(precision=8, suppress=True, linewidth=120)
# x_numpy = np.random.random((1, 1, 6, 9))
# x_tensor = torch.tensor(x_numpy, requires_grad=True)
#
# max_pool_tensor = torch.nn.MaxPool2d((2, 2), 2)
# max_pool_numpy = MaxPooling2D((2, 2), stride=2)
#
# out_numpy = max_pool_numpy(x_numpy[0, 0])
# out_tensor = max_pool_tensor(x_tensor)
#
# d_loss_numpy = np.random.random(out_tensor.shape)
# d_loss_tensor = torch.tensor(d_loss_numpy, requires_grad=True)
# out_tensor.backward(d_loss_tensor)
#
# dx_numpy = max_pool_numpy.backward(d_loss_numpy[0, 0])
# dx_tensor = x_tensor.grad
# # print('input \n', x_numpy)
# print("out_numpy \n", out_numpy)
# print("out_tensor \n", out_tensor.data.numpy())
#
# print("dx_numpy \n", dx_numpy)
# print("dx_tensor \n", dx_tensor.data.numpy())

import numpy as np
import os

path = '/mnt/data5/PaddleVideo/data/ucf101/split_file/ucf101_train_split_1_rawframes.txt'
# data_list = os.listdir(path)
# for i in data_list:
#     file_name = os.path.join(path, i)
#     data = np.load(file_name)
#     print(i, data.shape)

tmp = [x.strip().split(' ') for x in open(path)]
print(tmp)