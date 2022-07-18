# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn.manifold import TSNE

def get_data():
    t = 8
    n_class = 5
    n_query = 5
    exp_root = '../stgcn_attention/save_attention_xy/epoch_0_iter_0'
    x_path = os.path.join(exp_root, 'attention_y.npy')
    data = np.load(x_path)
    n, v, c = data.shape
    label = np.zeros(n // t)
    for i in range(n_class):
        base_idx = i * n_class * n_query
        true_idx = [idx for idx in range(base_idx + i * n_query, base_idx + (i + 1) * n_query)]
        label[true_idx] = 1

    # label = np.tile(label, t)

    data = data.reshape(-1, t, v, c)
    data = data[:, 0, :, :]
    data = data.reshape(-1, v * c)
    n_samples, n_features = data.shape
    print(data.shape)
    print(label.shape)
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        if label[i] == 0:
            plt.scatter(data[i, 0], data[i, 1], c='b')
        elif label[i] == 1:
            plt.scatter(data[i, 0], data[i, 1], c='r')
        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=plt.cm.Set1(label[i] / 10.),
        #          fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    t0 = time()
    result = tsne.fit_transform(data)

    fig = plot_embedding(result, label, 't-SNE embedding of the attention_x (time %.2fs)'% (time() - t0))
    plt.savefig('./t-sne.jpg')
    #plt.show(fig)


if __name__ == '__main__':
    main()