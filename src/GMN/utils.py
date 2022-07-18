import numpy as np
import torch
from GMN.graphembeddingnetwork import GraphEncoder, GraphAggregator
from GMN.graphmatchingnetwork import GraphMatchingNet

'''
    生成GMN模型
'''
def getGMNmodel(in_channels, out_channels, opt):

    node_state_dim = in_channels
    graph_rep_dim = out_channels
    node_feature_dim = in_channels
    edge_feature_dim = 1

    encoder = dict(
        node_hidden_sizes=[node_state_dim],
        node_feature_dim=node_feature_dim,
        edge_hidden_sizes=None,
        edge_feature_dim=edge_feature_dim)
    aggregator = dict(
        node_hidden_sizes=[graph_rep_dim],
        graph_transform_sizes=[graph_rep_dim],
        input_size=[node_state_dim],
        gated=True,
        aggregation_type='sum')
    graph_matching_net = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],
        n_prop_layers=opt.n_prop,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here. gru
        node_update_type='residual',
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=False,
        # set to True if your graph is directed
        reverse_dir_param_different=False,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
        # set to `embedding` to use the graph embedding net.
        prop_type='matching')

    Encoder = GraphEncoder(**encoder)
    Aggregator = GraphAggregator(**aggregator)
    encoder = 0
    import json, os
    with open(os.path.join(opt.experiment_root, 'opt.json'), 'w') as f:
        json.dump(aggregator, f)
        f.write('\n')
        json.dump(graph_matching_net, f)
        f.write('\n')
    return GraphMatchingNet(encoder, Aggregator, **graph_matching_net)


def getidx(N, V):
    # fromm = [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 12, 13, 13, \
    #          14, 14, 15, 16, 16, 17, 17, 18, 18, 19, 20, 20, 20, 20, 21, 22, 23, 24]
    # to = [1, 12, 16, 0, 20, 3, 20, 2, 5, 20, 4, 6, 5, 7, 22, 6, 21, 9, 20, 8, 10, 9, 11, 24, 10, 23, 0, 13, 12, 14, \
    #       13, 15, 14, 0, 17, 16, 18, 17, 19, 18, 1, 2, 4, 8, 7, 6, 11, 10]

    froms = []
    to = []
    neighbor_base = [(1, 2), (2, 21), (3, 21),
                     (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                     (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
    for (i, j) in neighbor_link:
        froms.append(i)
        to.append(j)
        froms.append(j)
        to.append(i)

    edge_length = len(to)
    edge_feature = np.ones(edge_length * N)

    edge_feature = edge_feature.reshape(edge_length * N, 1)
    from_idx = []
    to_idx = []
    graph_idx = []

    for i in range(N):
        from_idx.extend(froms)
        to_idx.extend(to)

    for i in range(N):
        x = np.ones(V) * i
        graph_idx.append(x)
    graph_idx = np.concatenate(graph_idx, 0).astype(int)

    edge_feature, from_idx, to_idx, graph_idx = torch.tensor(edge_feature, dtype=torch.float32), torch.tensor(from_idx), \
                                                torch.tensor(to_idx), torch.tensor(graph_idx)

    l = list()
    l.append(edge_feature)
    l.append(from_idx)
    l.append(to_idx)
    l.append(graph_idx)
    return l, N
