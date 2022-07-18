import csv, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('ggplot')
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="darkgrid")


def draw(file_paths, dir, draw="acc"):
    scoremarkers = ["v", "o", "*", "s"]
    accmarkers = ["v", "o", "*", "s"]

    if draw in "acc":
        print(file_paths)
        for i, path in enumerate(file_paths):
            i = 2
            fmri = pd.read_csv(path)
            fmri['weight'] = [0, 1, 2, 3, 4, 5]
            # print(fmri)

            ax = sns.lineplot(x="weight", y="acc", err_style="band", ci="sd", marker=accmarkers[2], markersize=15, color='blue', linewidth=3,
                              # hue="region", style="event",
                              data=fmri)

        ax.figure.set_size_inches(6, 3)
        # plt.xticks(np.arange(0, 6, 1), [0.01, 0.1, 0.3, 0.5, 0.7, 0.9])
        plt.xticks(np.arange(0, 6, 1), ['0.01', '0.1', "0.3", "0.5", "0.7", "0.9"])
        # plt.xticks(np.arange(0, 5, 1), ['0.1', '1', "10", "50", "100"], fontsize=12)
        # plt.xticks(np.arange(0, 6, 1), ['0', '0.1', "0.2", "0.3", "0.4", "0.5"], fontsize=12)
        plt.xlabel("regularization parameter " + r'$\lambda$', fontsize=15)
        # plt.xlabel("Drop Rate", fontsize=15)
        plt.ylabel('Accuracy (%)', fontsize=15)
        #plt.yticks(np.arange(65, 76, 2))
        # plt.yticks(np.arange(65, 76, 2), fontsize=12)
        plt.yticks(np.arange(67, 76, 2), fontsize=12)

        plt.legend(["RankMax"], loc="upper right", fontsize=12)
        plt.savefig(os.path.join(dir, "st-gcn.pdf"), format="pdf", bbox_inches="tight", dpi=400)

    else:
        for i, path in enumerate(file_paths):
            fmri = pd.read_csv(path)
            fmri["score"] = fmri["score"] * 100
            ax = sns.lineplot(x="weight", y="score", err_style="band", ci="sd", marker=scoremarkers[i], linewidth=3,
                              # hue="region", style="event",
                              data=fmri)
            plt.xlabel("weight", fontsize=20)
            plt.ylabel('$\mathcal{S}_{\mu=1}$,%', fontsize=20)
            plt.yticks(np.arange(0, 60, 10))
            plt.legend(["EM", r'+BNM', "+Div", "+DE"], loc="upper left", fontsize=12)
            plt.savefig(os.path.join(dir + "show_SCORE.pdf"), format="pdf", bbox_inches="tight", dpi=400)
    # plt.show()

if __name__ == '__main__':

    draw([
        # '/home/zhanghongyi/hyperparam/test.csv',
        #"/home/zhanghongyi/hyperparam/dropedge.csv",
        "/home/zhanghongyi/hyperparam/avg_st_gcn.csv"
         ], "/home/zhanghongyi/hyperparam",
     "acc")

    # import matplotlib.pyplot as plt
    #
    # data = [5, 20, 15, 25, 10]
    #
    # plt.bar(range(len(data)), data)
    # plt.savefig(os.path.join(dir, "pairnorm.pdf"), format="pdf", bbox_inches="tight", dpi=400)
