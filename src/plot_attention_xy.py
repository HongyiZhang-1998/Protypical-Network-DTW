from utils import plot_matrix
import numpy as np
import os
import random

def plot_x(exp_root, t, x_path, n_class, n_query):

    x = np.load(x_path)
    n, v, c = x.shape
    new_x = x.reshape(-1, t, v, c)

    plot_path = os.path.join(exp_root, 'plot_cross_attention_support')

    for i in range(n_class):
        dir = os.path.join(plot_path, 'class_{}'.format(i))
        if not os.path.exists(dir):
            os.makedirs(dir)

        base_idx = i * n_class * n_query
        true_idx = [idx for idx in range(base_idx + i * n_query, base_idx + (i + 1) * n_query)]
        random.shuffle(true_idx)
        true_random_idx = true_idx[0]
        false_idx = [idx for idx in range(base_idx, base_idx + n_query * n_class)]
        false_idx = list(set(false_idx) - set(true_idx))
        random.shuffle(false_idx)
        false_random_idx = false_idx[0]

        for j in [0, t // 2, t - 1]:
            file_path = os.path.join(dir, 'true_t_{}.jpg'.format(j))
            print(file_path)
            plot_matrix(file_path, new_x[true_random_idx, j, :, :])
            file_path = os.path.join(dir, 'false_t_{}.jpg'.format(j))
            plot_matrix(file_path, new_x[false_random_idx, j, :, :])

if __name__ == '__main__':
    t = 8
    n_class = 5
    n_query = 5
    exp_root = '../stgcn_attention/save_attention_xy/epoch_0_iter_0'
    x_path = os.path.join(exp_root, 'attention_y.npy')
    plot_x(exp_root, t, x_path, n_class, n_query)