"""Data processing utilities."""

import json
import logging
import math
import os
import os.path as osp
import pickle
from glob import glob

import networkx as nx
import random
import numpy as np
import torch
import torch.nn.functional as fn
import torch.nn.init as init
from texttable import Texttable
from torch_geometric.utils import erdos_renyi_graph, to_undirected, to_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_scatter import scatter_add


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))
    return data


def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transformed GED.
    :return score: Squared error.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction - target) ** 2
    return score


def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"] / (0.5 * (len(data["labels_1"]) + len(data["labels_2"])))
    return norm_ged


def calculate_histogram(abstract_features_1, abstract_features_2, num_bins):
    """
    Calculate histogram from similarity matrix.
    :param abstract_features_1: Feature matrix for graph 1.
    :param abstract_features_2: Feature matrix for graph 2.
    :param num_bins: number of bins
    :return hist: Histogram of similarity scores.
    """
    scores = torch.mm(abstract_features_1, abstract_features_2).detach()
    scores = scores.view(-1, 1)
    hist = torch.histc(scores, bins=num_bins)
    hist = hist / torch.sum(hist)
    hist = hist.view(1, -1)
    return hist


def calculate_normalized_dual_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"] / (0.5 * (len(data["p_labels_1"]) + len(data["p_labels_2"])))
    norm_ged += data["ged"] / (0.5 * (len(data["d_labels_1"]) + len(data["d_labels_2"])))
    return norm_ged


# Ahren Added

def calculate_normalized_primal_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"] / (0.5 * (len(data["p_labels_1"]) + len(data["p_labels_2"])))
    return norm_ged


def load_data(data_file_name):
    records = []
    file = get_file_path(data_file_name)
    if osp.isfile(file):
        with open(file, 'rb') as handle:
            records = pickle.load(handle)
    else:
        raise FileNotFoundError('Path is not correct: {}'.format(data_file_name))

    return list(records.items())


def get_file_path(data_file_name):
    full_name = data_file_name
    if type(data_file_name) is not str:
        raise RuntimeError('Wrong file name type')
    ext = '.pickle'

    if ext not in data_file_name:
        full_name += ext
    return full_name


def sorted_nicely(lst):
    def force_int(s):
        try:
            return int(s)
        except ValueError as e:
            # print("(Warning) [", str(s), "]: is not int, ", e)
            return s

    import re

    def alphanumeric_key(s):
        return [force_int(c) for c in re.split('([0-9]+)', s)]

    return sorted(lst, key=alphanumeric_key)


def fetch_raw_graphs(root_path, preprocess=None):
    graphs = []
    for file in sorted_nicely(glob(root_path + '/*.gexf')):
        gid = int(osp.basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        if preprocess:
            graphs.append(preprocess(g))
        else:
            graphs.append(g)
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))
    return graphs


def fetch_raw_graphs_dict(root_path, preprocess=None):
    graphs = {}
    for file in sorted_nicely(glob(root_path + '/*.gexf')):
        gid = int(osp.basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        if preprocess is not None:
            graphs[gid] = preprocess(g)
        else:
            graphs[gid] = g
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))
    return graphs


def process_labels(node_data_list):
    if len(node_data_list[0][1]) == 0:
        return [n[0] for n in node_data_list]
    elif len(node_data_list[0][1]) == 1:
        return [n[1]['label'] for n in node_data_list]
    else:
        return [n[1]['type'] for n in node_data_list]


def preprocess_graph(gr):
    """
    return dict of {labels: [], edges: []}
    Args:
        gr: networkx's graph formatted data
        one_hot: Boolean
    Returns:
        gd: dict
    """
    node_names = list(gr.nodes)
    labels = process_labels(list(gr.nodes.data()))
    # l2i = {node_names[i]: i for i in range(len(node_names))}  # label to index
    # edges = [[l2i[e[0]], l2i[e[1]]] for e in list(map(list, gr.edges))]  # process to int()
    edges = [[node_names.index(e[0]), node_names.index(e[1])] for e in list(map(list, gr.edges))]  # process to int()
    return {'labels': labels, 'edges': edges}


def process_dual_labels(node_data_list, dual_nodes, directed=False):
    if len(node_data_list[0][2]) == 0:
        return dual_nodes
    else:
        labels = [0] * len(dual_nodes)
        for i in range(len(node_data_list)):
            data = node_data_list[i]
            node = (data[0], data[1])
            if node not in dual_nodes:
                node = (data[1], data[0])
                if node not in dual_nodes:
                    raise ValueError(node, ' is not available.')
            if 'valence' in data[2].keys():
                labels[dual_nodes.index(node)] = data[2]['valence']
        return labels


def preprocess_dual_graph(gr, directed=False):
    """
    return dict of {labels: [], edges: []}
    Args:
        gr: networkx's graph formatted data
        directed: Boolean
    Returns:
        gd: dict
    """
    dgr = nx.line_graph(gr)
    d_nodes = list(dgr.nodes)
    labels = process_dual_labels(list(gr.edges.data()), d_nodes, directed=directed)
    edges = [[d_nodes.index(e[0]), d_nodes.index(e[1])] for e in list(map(list, dgr.edges))]  # process to int()
    # if len(edges) == 0:
    #     raise RuntimeError('edge number is 0: [id ', gr.graph['gid'], ']')
    return {'labels': labels, 'edges': edges}


def tuple_to_one_hot(t):  # TODO
    pass


def to_one_hot(lst):
    oh = []
    if isinstance(lst[0], str):
        oh = str2int(lst)
        max_val = max(oh) + 1
        for i in range(len(oh)):
            elm = [0] * max_val
            elm[oh[i]] = 1
            oh[i] = elm
    elif isinstance(lst[0], int):
        oh = lst  # TODO
    elif isinstance(lst[0], tuple):
        oh = [tuple_to_one_hot(t) for t in lst]
    else:
        raise TypeError('to_one_hot: not a recognized type')
    return oh


def str2int(lst):
    return list(map(int, lst))


def dual_graph_processing(gr):  # TODO
    """
    Args:
        gr: networkx's graph formatted data
        one_hot: Boolean
    Returns:
    """
    dg = nx.line_graph(gr)  # get the dual graph
    # dg = preprocess_graph(dg, one_hot=one_hot)
    nodes = list(dg.nodes)
    edges = [[nodes.index(e[0]), nodes.index(e[1])] for e in list(dg.edges)]
    labels = process_labels(list(dg.nodes.node()))
    return {'labels': labels, 'edges': edges}


def draw_graph(g, file=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure()
    # import networkx as nx
    nx.draw(g, ax=f.add_subplot(111))
    labels = nx.draw_networkx_labels(g, pos=nx.spring_layout(g))
    if file is not None:
        f.savefig(file)
        print('Saved graph to {}'.format(file))


def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weights.data, gain=1.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, mean=0.0, std=1.0)
        # m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def save_model(epoch, model, optimizer, name):
    """
    Args:
        epoch: number pf epochs
        model: model object
        optimizer
        name: name of the model to be saved. The model parameters will be
            saved as '<name>.pickle'
    Returns: None
    """
    PATH = "../checkpoints/{}_{:d}.pickle".format(name, epoch)
    print('Saving model: %s' % PATH)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)


def load_model(model, optimizer, name):
    """
    Args:
        epoch: number pf epochs
        model: model object
        optimizer
        name: name of the model to be loaded, named '<name>.pickle'
    Returns: Number of epochs to continue training
    """
    PATH = "../checkpoints/{0}.pickle".format(name)
    print('Loading model: %s' % PATH)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


def setup_logging(path, log_file_name):
    if not osp.exists(path):
        os.mkdir(path)
    l_f_name = osp.join(path, log_file_name)
    logging.basicConfig(filename=l_f_name, filemode='a', format='%(message)s', datefmt='%H:%M:%S',
                        level=logging.DEBUG)


def process_graph(graph, max_num_nodes, device, one_hot=False):
    if one_hot:
        x = fn.one_hot(graph.x, max_num_nodes)
    else:
        x = graph.x
    h = torch.zeros(max_num_nodes, x.shape[1]).to(device)
    h[0: x.shape[0], :] = x
    return h


def output_profiling_results(path, name, results):
    filename = osp.join(path, name)
    with open(filename, "a+") as f:
        for line in results:
            f.write(line)
        f.close()


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    temp = prediction.argsort()
    r_prediction = np.empty_like(temp)
    r_prediction[temp] = np.arange(len(prediction))

    temp = target.argsort()
    r_target = np.empty_like(temp)
    r_target[temp] = np.arange(len(target))

    return rank_corr_function(r_prediction, r_target).correlation


def calculate_prec_at_k(k, prediction, target):
    """
    Calculating precision at k.
    """
    best_k_pred = prediction.argsort()[:k]
    best_k_target = target.argsort()[:k]

    return len(set(best_k_pred).intersection(set(best_k_target))) / k


def denormalize_sim_score(g1, g2, sim_score):
    """
    Converts normalized similar into ged.
    """
    return denormalize_ged(g1, g2, -math.log(sim_score, math.e))


def denormalize_ged(g1, g2, nged):
    """
    Converts normalized ged into ged.
    """
    return round(nged * (g1.num_nodes + g2.num_nodes) / 2)


def gen_synth_data(count=200, nl=None, nu=50, p=0.5, kl=None, ku=2):
    """
    Generating synthetic data based on Erdos–Renyi model.
    :param count: Number of graph pairs to generate.
    :param nl: Minimum number of nodes in a source graph.
    :param nu: Maximum number of nodes in a source graph.
    :param p: Probability of an edge.
    :param kl: Minimum number of insert/remove edge operations on a graph.
    :param ku: Maximum number of insert/remove edge operations on a graph.
    """
    if nl is None:
        nl = nu
    if kl is None:
        kl = ku

    data = []
    data_new = []
    mat = torch.full((count, count), float('inf'))
    norm_mat = torch.full((count, count), float('inf'))

    for i in range(count):
        n = random.randint(nl, nu)
        edge_index = erdos_renyi_graph(n, p)
        x = torch.ones(n, 1)

        g1 = Data(x=x, edge_index=edge_index, i=torch.tensor([i]))
        g2, ged = gen_pair(g1, kl, ku)

        data.append(g1)
        data_new.append(g2)
        mat[i, i] = ged
        norm_mat[i, i] = ged / (0.5 * (g1.num_nodes + g2.num_nodes))

    return data, data_new, mat, norm_mat


def gen_pairs(graphs, kl=None, ku=2):
    gen_graphs_1 = []
    gen_graphs_2 = []

    count = len(graphs)
    mat = torch.full((count, count), float('inf'))
    norm_mat = torch.full((count, count), float('inf'))

    for i, g in enumerate(graphs):
        g = g.clone()
        g.i = torch.tensor([i])
        g2, ged = gen_pair(g, kl, ku)
        gen_graphs_1.append(g)
        gen_graphs_2.append(g2)
        mat[i, i] = ged
        norm_mat[i, i] = ged / (0.5 * (g.num_nodes + g2.num_nodes))

    return gen_graphs_1, gen_graphs_2, mat, norm_mat


def to_directed(edge_index):
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


def gen_pair(g, kl=None, ku=2):
    if kl is None:
        kl = ku

    directed_edge_index = to_directed(g.edge_index)

    n = g.num_nodes
    num_edges = directed_edge_index.size(1)
    to_remove = random.randint(kl, ku)

    edge_index_n = directed_edge_index[:, torch.randperm(num_edges)[to_remove:]]
    if edge_index_n.size(1) != 0:
        edge_index_n = to_undirected(edge_index_n)

    row, col = g.edge_index
    adj = torch.ones((n, n), dtype=torch.uint8)
    adj[row, col] = 0
    non_edge_index = adj.nonzero().t()

    directed_non_edge_index = to_directed(non_edge_index)
    num_edges = directed_non_edge_index.size(1)

    to_add = random.randint(kl, ku)

    edge_index_p = directed_non_edge_index[:, torch.randperm(num_edges)[:to_add]]
    if edge_index_p.size(1):
        edge_index_p = to_undirected(edge_index_p)
    edge_index_p = torch.cat((edge_index_n, edge_index_p), 1)

    if hasattr(g, 'i'):
        g2 = Data(x=g.x, edge_index=edge_index_p, i=g.i)
    else:
        g2 = Data(x=g.x, edge_index=edge_index_p)

    g2.num_nodes = g.num_nodes
    return g2, to_remove + to_add


def aids_labels(g):
    types = [
        'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'fn',
        'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
        'Sb', 'Se', 'Ni', 'Te'
    ]

    return [types[i] for i in g.x.argmax(dim=1).tolist()]


def draw_weighted_nodes(filename, g, model):
    """
    Draw graph with weighted nodes (for AIDS).
    """
    features = model.convolutional_pass(g.edge_index, g.x)
    coefs = model.attention.get_coefs(features)

    print(coefs)

    plt.clf()
    gr = to_networkx(g).to_undirected()

    label_list = aids_labels(g)
    labels = {}
    for i, node in enumerate(gr.nodes()):
        labels[node] = label_list[i]

    vmin = coefs.min().item() - 0.005
    vmax = coefs.max().item() + 0.005

    nx.draw(gr, node_color=coefs.tolist(), cmap=plt.cm.Reds, labels=labels, vmin=vmin, vmax=vmax)


# copied and modified from PyTorch Geometric
# link: https://tinyurl.com/yaarr697
def to_dense_batch(x, batch=None, fill_value=0, desired_max_num_nodes=0):
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
    :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).
    In addition, a second tensor holding
    :math:`[N_1, \ldots, N_B] \in \mathbb{N}^B` is returned.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        fill_value (float, optional): The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)

    :rtype: (:class:`Tensor`, :class:`BoolTensor`)
    """
    if batch is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    batch_size = batch[-1].item() + 1
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0,
                            dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    max_num_nodes = max(num_nodes.max().item(), desired_max_num_nodes)

    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                       device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask


def to_dense_adj(edge_index, batch=None, edge_attr=None, desired_max_num_nodes=0):
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)
    batch_size = batch[-1].item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_add(one, batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    max_num_nodes = max(num_nodes.max().item(), desired_max_num_nodes)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size = size if edge_attr is None else size + list(edge_attr.size())[1:]
    dtype = torch.float if edge_attr is None else edge_attr.dtype
    adj = torch.zeros(size, dtype=dtype, device=edge_index.device)

    edge_index_0 = batch[edge_index[0]].view(1, -1)
    edge_index_1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    edge_index_2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if edge_attr is None:
        adj[edge_index_0, edge_index_1, edge_index_2] = 1
    else:
        adj[edge_index_0, edge_index_1, edge_index_2] = edge_attr

    return adj


def parameters_count(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def make_checkpoint(path, epoch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def plot_grad_flow(named_parameters, path):
    ave_grads = []
    layers = []
    empty_grads = []
    # total_norm = 0
    for n, p in named_parameters:
        if p.requires_grad and not (("bias" in n) or ("norm" in n) or ("bn" in n) or ("gain" in n)):
            if p.grad is not None:
                # writer.add_scalar('gradients/' + n, p.grad.norm(2).item(), step)
                # writer.add_histogram('gradients/' + n, p.grad, step)
                # total_norm += p.grad.data.norm(2).item()
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
            else:
                empty_grads.append({n: p.mean().cpu().item()})
    # total_norm = total_norm ** (1. / 2)
    # print("Norm : ", total_norm)
    plt.tight_layout()
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1.5, color="k")
    plt.xticks(np.arange(0, len(ave_grads), 1), layers, rotation="vertical", fontsize=4)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(path, dpi=300)
    plt.clf()