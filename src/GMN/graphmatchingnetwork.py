from GMN.graphembeddingnetwork import GraphEmbeddingNet
from GMN.graphembeddingnetwork import GraphPropLayer
import torch
import torch.nn.functional as F
import gl
import random
import numpy as np

import os

def pairwise_euclidean_similarity(x, y):
    """Compute the pairwise Euclidean similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = -|x_i - y_j|^2.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise euclidean similarity.
    """
    s = 2 * torch.mm(x, torch.transpose(y, 1, 0))
    diag_x = torch.sum(x * x, dim=-1)
    diag_x = torch.unsqueeze(diag_x, 0)
    diag_y = torch.reshape(torch.sum(y * y, dim=-1), (1, -1))

    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
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


def pairwise_cosine_similarity(x, y):
    """Compute the cosine similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise cosine similarity.
    """
    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), torch.tensor(1e-12).to(gl.device))))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), torch.tensor(1e-12).to(gl.device))))
    return torch.mm(x, torch.transpose(y, 1, 0))


PAIRWISE_SIMILARITY_FUNCTION = {
    'euclidean': pairwise_euclidean_similarity,
    'dotproduct': pairwise_dot_product_similarity,
    'cosine': pairwise_cosine_similarity,
}


def get_pairwise_similarity(name):
    """Get pairwise similarity metric by name.

    Args:
      name: string, name of the similarity metric, one of {dot-product, cosine,
        euclidean}.

    Returns:
      similarity: a (x, y) -> sim function.

    Raises:
      ValueError: if name is not supported.
    """
    if name not in PAIRWISE_SIMILARITY_FUNCTION:
        raise ValueError('Similarity metric name "%s" not supported.' % name)
    else:
        return PAIRWISE_SIMILARITY_FUNCTION[name]

def get_local_sim(a, arm, leg, body):
    '''

    :param a: all skeleton sim
    :param la: left arm skeleton idx
    :param ra: right arm
    :param ll: left leg
    :param rl: right leg
    :param body: body
    :return: a_la, a_ra, a_ll, a_rl, a_body
    '''
    a_arm_t = a[arm]
    a_arm = a_arm_t[:, arm]
    a_leg_t = a[leg]
    a_leg = a_leg_t[:, leg]
    a_body_t = a[body]
    a_body = a_body_t[:, body]
    return a_arm, a_leg, a_body

def compute_local_cross_attention(a, x, y):

    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i

    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y

def compute_cross_attention(x, y, sim, i):
    """Compute cross attention.

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i

    Args:
      x: NxD float tensor.
      y: MxD float tensor.
      sim: a (x, y) -> similarity function.

    Returns:
      attention_x: NxD float tensor.
      attention_y: NxD float tensor.
    """

    v, c = x.size()

    x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-8)
    a = sim(x_norm, y_norm)

    if gl.local_match != 1:
        a_x = torch.softmax(a, dim=1)  # i->j
        a_y = torch.softmax(a, dim=0)  # j->i
        attention_x = torch.mm(a_x, y)
        attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    else:
        arm_skeleton = torch.tensor([5, 6, 7, 9, 10, 11, 23, 24, 21, 22]).to(gl.device)
        leg_skeleton = torch.tensor([13, 14, 15, 17, 18, 19]).to(gl.device)
        body = torch.tensor([0, 1, 2, 3, 4, 8, 12, 16, 20]).to(gl.device)

        a_arm, a_leg, a_body = get_local_sim(a, arm_skeleton, leg_skeleton, body)

        attention_x = torch.zeros(v, c).to(gl.device)
        attention_y = torch.zeros(v, c).to(gl.device)
        attention_x[arm_skeleton], attention_y[arm_skeleton] = compute_local_cross_attention(a_arm, x[arm_skeleton], y[arm_skeleton])
        attention_x[leg_skeleton], attention_y[leg_skeleton] = compute_local_cross_attention(a_leg, x[leg_skeleton], y[leg_skeleton])
        attention_x[body], attention_y[body] = compute_local_cross_attention(a_body, x[body], y[body])

        if gl.debug == True and gl.epoch > 60 and random.randint(1, 100) <= 1:
            a_path = os.path.join(gl.experiment_root, 'a')
            if not os.path.exists(a_path):
                os.mkdir(a_path)
            a_path = os.path.join(a_path, '{}_{}'.format(gl.epoch, i))
            if not os.path.exists(a_path):
                os.mkdir(a_path)
            np.savetxt(os.path.join(a_path, 'a.txt'), a.cpu().detach().numpy(), fmt='%.2f')
            np.savetxt(os.path.join(a_path, 'a_arm.txt'), a_arm.cpu().detach().numpy(), fmt='%.2f')
            np.savetxt(os.path.join(a_path, 'a_leg.txt'), a_leg.cpu().detach().numpy(), fmt='%.2f')
            np.savetxt(os.path.join(a_path, 'a_body.txt'), a_body.cpu().detach().numpy(), fmt='%.2f')
            np.savetxt(os.path.join(a_path, 'atten_x.txt'), attention_x.cpu().detach().numpy(), fmt='%.2f')
            np.savetxt(os.path.join(a_path, 'atten_y.txt'), attention_y.cpu().detach().numpy(), fmt='%.2f')

    return attention_x, attention_y

def batch_block_pair_attention(data,
                               block_idx,
                               n_blocks,
                               similarity='dotproduct'):
    """Compute batched attention between pairs of blocks.

    This function partitions the batch data into blocks according to block_idx.
    For each pair of blocks, x = data[block_idx == 2i], and
    y = data[block_idx == 2i+1], we compute

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

    and

    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i.

    Args:
      data: NxD float tensor.
      block_idx: N-dim int tensor.
      n_blocks: integer.
      similarity: a string, the similarity metric.

    Returns:
      attention_output: NxD float tensor, each x_i replaced by attention_x_i.

    Raises:
      ValueError: if n_blocks is not an integer or not a multiple of 2.
    """
    if not isinstance(n_blocks, int):
        raise ValueError('n_blocks (%s) has to be an integer.' % str(n_blocks))

    if n_blocks % 2 != 0:
        raise ValueError('n_blocks (%d) must be a multiple of 2.' % n_blocks)
    # similarity = 'euclidean'
    sim = get_pairwise_similarity(similarity)

    results = []

    # This is probably better than doing boolean_mask for each i


    partitions = []
    for i in range(n_blocks):
        # print(data[block_idx == i, :])
        # print(block_idx == i)
        partitions.append(data[block_idx == i, :])

    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]

        attention_x, attention_y = compute_cross_attention(x, y, sim, i)
        results.append(attention_x)
        results.append(attention_y)
    results = torch.cat(results, dim=0)

    return results


class GraphPropMatchingLayer(GraphPropLayer):
    """A graph propagation layer that also does cross graph matching.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """

    def forward(self,
                node_states,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                similarity='dotproduct',
                edge_features=None,
                node_features=None):
        """Run one propagation step with cross-graph matching.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          graph_idx: [n_onodes] int tensor, graph id for each node.
          n_graphs: integer, number of graphs in the batch.
          similarity: type of similarity to use for the cross graph attention.
          edge_features: if not None, should be [n_edges, edge_feat_dim] tensor,
            extra edge features.
          node_features: if not None, should be [n_nodes, node_feat_dim] tensor,
            extra node features.

        Returns:
          node_states: [n_nodes, node_state_dim] float tensor, new node states.

        Raises:
          ValueError: if some options are not provided correctly.
        """

        # aggregated_messages = self._compute_aggregated_messages(
        #     node_states, from_idx, to_idx, edge_features=edge_features)

        cross_graph_attention = batch_block_pair_attention(
            node_states, graph_idx, n_graphs, similarity=similarity)

        attention_input = node_states - cross_graph_attention

        # return self._compute_node_update(node_states,
        #                                  [aggregated_messages, attention_input],
        #                                  node_features=node_features)

        #not aggregate edge information
        return self._compute_node_update(node_states,
                                         [attention_input],
                                         node_features=node_features)


class GraphMatchingNet(GraphEmbeddingNet):
    """Graph matching net.

    This class uses graph matching layers instead of the simple graph prop layers.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """

    def __init__(self,
                 encoder,
                 aggregator,
                 node_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 n_prop_layers,
                 share_prop_params=False,
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 layer_class=GraphPropLayer,
                 similarity='dotproduct',
                 prop_type='embedding'):

        super(GraphMatchingNet, self).__init__(
            encoder,
            aggregator,
            node_state_dim,
            edge_hidden_sizes,
            node_hidden_sizes,
            n_prop_layers,
            share_prop_params=share_prop_params,
            edge_net_init_scale=edge_net_init_scale,
            node_update_type=node_update_type,
            use_reverse_direction=use_reverse_direction,
            reverse_dir_param_different=reverse_dir_param_different,
            layer_norm=layer_norm,
            layer_class=GraphPropMatchingLayer,
            prop_type=prop_type,
        )
        self._similarity = similarity

    def _apply_layer(self,
                     layer,
                     node_states,
                     from_idx,
                     to_idx,
                     graph_idx,
                     n_graphs,
                     edge_features):

        """Apply one layer on the given inputs."""
        return layer(node_states, from_idx, to_idx, graph_idx, n_graphs,
                     similarity=self._similarity, edge_features=edge_features)
