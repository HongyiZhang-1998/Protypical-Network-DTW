import os
import numpy as np
from utils import plot_matrix
path = "/home/zhanghongyi/dtw-prototypical-network/log/test_noisy/seed_2021/_datasetntu120/_backbonest_gcn/_bias0/_reg0.0/_att0/"

paths = [os.path.join(path, 'D_'), os.path.join(path, "R_")]

for i in paths:
    save_path = os.path.join(path, i[-2] + "_heatmap")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    l = os.listdir(i)
    for file in l:
        file_name = os.path.join(i, file)
        matrix = np.load(file_name)

        if i[-2] == 'D':
            matrix = matrix[2]
        else:
            matrix = matrix[2, 1:-1, 1:-1]

        save_file_name = os.path.join(save_path, file.split('.')[0] + '_batch_2.jpg')
        plot_matrix(save_file_name, matrix)


