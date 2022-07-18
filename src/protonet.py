import os
import time

import torch.nn as nn
import torch
import numpy as np
from GMN.utils import getGMNmodel, getidx
from mmskl.st_gcn_aaai18 import ST_GCN_18
from utils import get_support_query_data, extract_k_segement, compute_similarity, euclidean_dist, euclidean_distance
from torch.nn import functional as F
import gl
from soft_dtw import SoftDTW
from model.msg3d import MS_G3D_Model
from AGCN.agcn import AGCN_Model
from cross_attention import CrossAttention, PositionEncoding
from self_attention import SelfAttention
from utils import plot_matrix
import os

# torch.set_printoptions(
#                 precision=20,  # 精度，保留小数点后几位，默认4
#                 threshold=1000,
#                 edgeitems=3,
#                 linewidth=150,  # 每行最多显示的字符数，默认80，超过则换行显示
#                 profile=None,
#                 sci_mode=False  # 用科学技术法显示数据，默认True
#             )

class MLP(nn.Module):
    def __init__(self, input_size, out_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, out_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

class ProtoNet(nn.Module):

    def __init__(self, opt):
        super(ProtoNet, self).__init__()

        if 'ntu120' in gl.dataset:
            node = 25
            ms_graph = 'ms_g3d_graph.ntu_rgb_d.AdjMatrixGraph'
            sh_grpah = 'shift_gcn_graph.ntu_rgb_d.Graph'
            st_graph = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        elif gl.dataset == 'kinetics':
            node = 18
            ms_graph = 'ms_g3d_graph.kinetics.AdjMatrixGraph'
            sh_grpah = 'shift_gcn_graph.kinetics.Graph'
            st_graph = {'layout': 'openpose', 'strategy': 'spatial'}
        else:
            ms_graph = None
            sh_grpah = None
            st_graph = None
            node = 0

        if gl.backbone == 'ms_g3d':
            self.model = MS_G3D_Model(
                num_class=0,
                num_point=node,
                num_person=2,
                num_gcn_scales=13,
                num_g3d_scales=6,
                graph=ms_graph,
            )
            self.in_channel = node * 192
            self.out_channel = node
        elif gl.backbone == '2s_AGCN':
            self.model = AGCN_Model(
                num_class=60,
                num_point=node,
                num_person=2,
                graph=sh_grpah,
                graph_args={'labeling_mode': 'spatial'}
            )
            self.in_channel = node * 256
            self.out_channel = node
        else:
            self.model = ST_GCN_18(
                in_channels=3,
                num_class=60,
                dropout=0.5,
                edge_importance_weighting=True,
                graph_cfg=st_graph
            )
            self.in_channel = node * 64
            self.out_channel = node

        self.conv = nn.Conv1d(
            in_channels=self.in_channel,
            out_channels=self.out_channel,
            kernel_size=1,
            bias=False)
        self.data_bn = nn.BatchNorm1d(self.in_channel)

        if gl.use_attention == 1:
            # self.attention = SelfAttention(num_attention_heads=1, input_size=256, hidden_size=256, hidden_dropout_prob=0.2)
            self.attention_x = CrossAttention(num_attention_heads=1, input_size=256, hidden_size=256, hidden_dropout_prob=0.2)
            self.attention_y = CrossAttention(num_attention_heads=1, input_size=256, hidden_size=256, hidden_dropout_prob=0.2)
        else:
            self.attention_x = None
            self.attention_y = None

        if gl.use_bias == 1:
            if gl.dataset == 'kinetics':
                T_size = 13
            else:
                T_size = 8
            self.bias = nn.Parameter(torch.zeros(T_size, T_size))
        else:
            self.bias = None

    def plot_attention(self, classes):

        gl.iter += 1
        if gl.epoch % 5 == 0 and gl.iter % 100 == 0:
            classes_path = os.path.join(gl.experiment_root, 'save_attention_probs', 'epoch_{}_iter_{}'.format(gl.epoch, gl.iter))
            if not os.path.exists(classes_path):
                os.makedirs(classes_path)
            np.savetxt(os.path.join(classes_path, 'classes.txt'), classes.cpu().detach().numpy(), fmt='%.0f')

    def loss(self, input, target, n_support, dtw, d=None):
        # input is encoder by ST_GCN
        n, c, t, v = input.size()

        def supp_idxs(cc):
            # FIXME when torch will support where as np
            return torch.nonzero(target.eq(cc), as_tuple=False)[:n_support].squeeze(1)\

        # FIXME when torch.unique will be available on cuda too
        classes = torch.unique(target)
        # self.plot_attention(classes)
        n_class = len(classes)

        # FIXME when torch will support where as np
        # assuming n_query, n_target constants
        n_query = target.eq(classes[0].item()).sum().item() - n_support
        support_idxs = list(map(supp_idxs, classes))
        z_proto = torch.stack([input[idx_list] for idx_list in support_idxs]).view(-1, c, t, v)
        # FIXME when torch will support where as np
        query_idxs = torch.stack(list(map(lambda c: torch.nonzero(target.eq(c), as_tuple=False)[n_support:], classes))).view(-1)
        zq = input[query_idxs.long()]
        z_proto = z_proto.view(n_class, n_support, c, t, v).mean(1)  # n, c, t, v


        if dtw > 0:
            dist, reg_loss = self.dtw_loss(zq, z_proto, d)
        else:
            zq, z_proto = F.avg_pool2d(zq, zq.size()[2:]).view(n_class * n_query, c), F.avg_pool2d(z_proto, z_proto.size()[2:]).view(n_class, c)
            dist = euclidean_dist(zq, z_proto)
            reg_loss = torch.tensor(0).float().to(gl.device)

        log_p_y = F.log_softmax(-dist, dim=1).view(n_class, n_query, -1)

        target_inds = torch.arange(0, n_class).to(gl.device)
        target_inds = target_inds.view(n_class, 1, 1)
        target_inds = target_inds.expand(n_class, n_query, 1).long()

        batch_loss = -log_p_y.gather(2, target_inds).squeeze().view(-1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)

        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

        if gl.reg_rate > 0:
            loss_val += reg_loss

        return loss_val, acc_val, reg_loss, batch_loss

    def dtw_loss(self, zq, z_proto, d):
        if self.attention_x != None:
            zq = zq.permute(0, 2, 3, 1).contiguous()  # n, t, v, c
            z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
            dist = self.attention_dtw_dist(zq, z_proto)
        else:
            z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
            zq = zq.permute(0, 2, 3, 1).contiguous()
            dist = self.dtw_dist(zq, z_proto, d)

        reg_loss = torch.tensor(0).float().to(gl.device)

        if gl.reg_rate > 0:
            reg_loss = self.svd_reg_spatial(z_proto) + self.svd_reg_spatial(zq)
            rate = gl.reg_rate
            reg_loss = reg_loss * rate

        return dist, reg_loss

    def dtw_dist(self, x, y, d):
        '''
            :param x: [n, t, c] z_query
            :param y: [m, t, c] z_proto
            :return: [n, m]
        '''

        if len(x.size()) == 4:
            n, t, v, c = x.size()
            x = x.view(n, t, v * c)
            y = y.view(-1, t, v * c)

        n, t, c = x.size()
        m, _, _ = y.size()

        x = x.unsqueeze(1).expand(n, m, t, c).reshape(n * m, t, c)
        y = y.unsqueeze(0).expand(n, m, t, c).reshape(n * m, t, c)

        sdtw = SoftDTW(gamma=gl.gamma, normalize=True, attention=self.attention_x, attention_y=self.attention_y, bias=self.bias, d=d)
        loss = sdtw(x, y)

        return loss.view(n, m)

    def attention_dtw_dist(self, x, y):
        '''
            :param x: [n, t, c] z_query
            :param y: [m, t, c] z_proto
            :return: [n, m]
        '''

        n, t, v, c = x.size()
        m, _, _, _ = y.size()

        x = x.unsqueeze(1).expand(n, m, t, v, c).reshape(n * m, t, v, c)
        y = y.unsqueeze(0).expand(n, m, t, v, c).reshape(n * m, t, v, c)

        pos_enc = PositionEncoding(x.view(-1, v, c), c)

        sdtw = SoftDTW(gamma=gl.gamma, normalize=True, attention=self.attention_x, attention_y=self.attention_y, pos_enc=pos_enc)
        loss = sdtw(x, y)

        return loss.view(n, m)

    def svd_reg_spatial(self, x):

        if len(x.size()) == 4:
            n, t, v, c = x.size()
            x = x.view(-1, v, c)

        loss = torch.tensor(0).float().to(gl.device)

        for i in range(x.size()[0]):
            # BNM
            # _, s_tgt, _ = torch.svd(x[i])
            # method_loss = -torch.mean(s_tgt)

            # transpose_X = torch.transpose(x[i], 1, 0) # c, t
            transpose_X = x[i]

            # fast version
            softmax_tgt = torch.softmax((transpose_X - torch.max(transpose_X)), dim=1)
            list_svd, _ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_tgt, 2), dim=0)), descending=True)
            method_loss = -torch.mean(list_svd[:min(softmax_tgt.shape[0], softmax_tgt.shape[1])])

            loss += method_loss

        return loss / x.size()[0]

    def svd_reg_temporal(self, x):
        n, t, c = x.size()

        loss = torch.tensor(0).float().to(gl.device)
        for i in range(n):
            # BNM
            # _, s_tgt, _ = torch.svd(x[i])
            # method_loss = -torch.mean(s_tgt)
            # FBNM
            # transpose_X = torch.transpose(x[i], 1, 0) # c, t
            transpose_X = x[i]

            softmax_tgt = torch.softmax((transpose_X - torch.max(transpose_X)), dim=1)

            list_svd, _ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_tgt, 2), dim=0)), descending=True)
            method_loss = torch.mean(list_svd[:min(softmax_tgt.shape[0], softmax_tgt.shape[1])])

            loss += method_loss

        return loss

    def plot_node_similarity(self, x):
        '''
        :param x: (n, t, v, c)
        :return: (v, v)
        '''
        n, t, v, c = x.size()
        x = x.view(-1, v, c)

        x1 = x.unsqueeze(2).expand(-1, v, v, c)
        x2 = x.unsqueeze(1).expand(-1, v, v, c)
        dist = torch.pow(x1 - x2, 2).sum(2)

        dist = dist.view(n, t, v, v)
        if gl.iter % 100 == 0:
            plot_path = os.path.join(gl.experiment_root, 'plot_BNM', 'epoch_{}_iter_{}'.format(gl.epoch, gl.iter))
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)

            for i in range(n):
                file_name_t0 = os.path.join(plot_path, 'batch_{}_t_{}.jpg'.format(i, 0))
                file_name_half_t = os.path.join(plot_path, 'batch_{}_t_{}.jpg'.format(i, t // 2))
                file_name_t = os.path.join(plot_path, 'batch_{}_t_{}.jpg'.format(i, t - 1))

                plot_dist_t0 = dist[i, 0, :, :]
                plot_dist_half_t = dist[i, t // 2, :, :]
                plot_dist_t = dist[i, -1, :, :]

                plot_matrix(file_name_t0, plot_dist_t0)
                plot_matrix(file_name_half_t, plot_dist_half_t)
                plot_matrix(file_name_t, plot_dist_t)

        return None

    def angle_reg_loss(self, x, y, n_class, n_query, target_inds):
        n, t, c = x.size()
        m, _, _ = y.size()
        x = F.normalize(x, dim=2)
        y = F.normalize(y, dim=2)

        x = x.unsqueeze(1).expand(n, m, t, c).reshape(n * m, t, c)
        y = y.unsqueeze(0).expand(n, m, t, c).reshape(n * m, t, c)

        matrix = torch.pow(x - y, 2).sum(2).sum(1).view(n, m)

        angle_reg_loss = matrix.view(n_class, n_query, -1)
        angle_reg_loss = -angle_reg_loss.gather(2, target_inds).squeeze().view(-1).mean()

        return angle_reg_loss

    def idm_reg(self, x):
        n, t, c = x.size()
        reg_loss = torch.tensor(0).float().to(gl.device)
        thred = 5
        margin = 2
        weight, inverse_weight = self.get_W(x, thred)
        for i in range(n):
            dist = euclidean_dist(x[i, :, :], x[i, :, :]) # t * t
            inverse_dist = torch.max(torch.zeros(t, t).to(gl.device), margin - dist).to(gl.device)
            reg_loss += (inverse_dist * inverse_weight + dist * weight).sum()

        return reg_loss / n

    def get_W(self, x, thred):
        n, t, c = x.size()
        weight, inverse_weight = torch.zeros(t, t).float().to(gl.device), torch.zeros(t, t).float().to(gl.device)
        import math
        for i in range(t):
            for j in range(t):
                if math.fabs(i - j) <= thred:
                    weight[i, j] = 1 / ((i - j) ** 2 + 1)
                else:
                    inverse_weight[i, j] = (i - j) ** 2 + 1
        return weight, inverse_weight

    def extract_channel(self, x):
        return 0
        # n, c, t, v = x.size()
        # x = x.permute(0, 1, 3, 2).contiguous() # n, t, c, v
        # x = x.view(n, c * v, t) # n, c, t
        # x = self.data_bn(x)
        # x = self.conv(x)
        # x = x.view(n, t, -1)
        # return x

    def forward(self, x):
        x = self.model(x)

        return x

