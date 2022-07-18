import numpy as np
import os
import random
from utils import plot_matrix

t = 8
n_class = 5
n_query = 5
exp = '../stgcn_attention/save_attention_probs/'
path_list = os.listdir(exp)
for exp_root_path in path_list:
    exp_root = os.path.join(exp, exp_root_path)
    save_plot_path = os.path.join('../stgcn_attention/plot_cross_attention', exp_root_path)
    classes_path = os.path.join(exp_root, 'classes.txt')
    classes = np.loadtxt(classes_path)
    x_path = os.path.join(exp_root, 'attention_y_probs.npy')

    data = np.load(x_path)
    data = data.squeeze()
    n, v, v = data.shape
    data = data.reshape(-1, t, v, v)

    idxs = [idx for idx in range(n_class * n_query * n_class)]

    for i in range(n_class):
        action_class = classes[i]
        base_idx = i * n_class * n_query

        true_idx = [idxs[base_idx + i * n_class + idx] for idx in range(n_query)]
        # print('true_idx', true_idx)

        path = os.path.join(save_plot_path, 'class_{}'.format(action_class))

        if not os.path.exists(path):
            os.makedirs(path)
        for j in [0, t // 2, t - 1]:
            true_file = os.path.join(path, 'same_class_{}_t_{}.jpg'.format(action_class, j))
            plot_matrix(true_file, data[true_idx[random.randint(0, n_query - 1)], j, :, :])
            for k in range(n_class):
                if k == i :
                    continue
                false_idx = [idxs[base_idx + k * n_class + idx] for idx in range(n_query)]
                # print('false_idx', false_idx)
                false_file = os.path.join(path, 'false_class_{}_t_{}.jpg'.format(classes[k], j))
                plot_matrix(false_file, data[false_idx[random.randint(0, n_query - 1)], j, :, :])

