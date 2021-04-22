import os
import os.path as osp
import glob
import pickle

import torch
import torch.nn.functional as fn
import networkx as nx
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip, extract_tar)
from torch_geometric.utils import to_undirected


class MCSDataset(InMemoryDataset):
    r"""The MCS datasets from the `"Graph Edit Distance Computation via Graph
    Neural Networks" <https://arxiv.org/abs/1808.05689>`_ paper.
    MCSs can be accessed via the global attributes :obj:`mcs` and
    :obj:`norm_mcs` and provide the exact MCSs for all train/train graph pairs
    and all train/test graph pairs.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (one of :obj:`"AIDS700nef"`,
            :obj:`"LINUX"`, :obj:`"PTC"`, :obj:`"IMDBMulti"`).
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://drive.google.com/uc?export=download&id={}'

    datasets = {
        'AIDS700nef': {
            'id': '1zgyxUcKtyZyXF7OzskhQ2SpF2inTdWQ7',
            'extract': extract_zip,
            'pickle': '1CkNioG2u6pkqRiKpOeh-Z8IAzXHN7nNk',
        },
        'LINUX': {
            'id': '1jQIzS0FRSB3xZHRSM8R5OoGSRGLo9uEW',
            'extract': extract_tar,
            'pickle': '1fNhB7uQGnzGW23veho9y_2GvqxiBAkn_',
        },
        'PTC': {
            'id': '14CZNSV3t8qY-beHntK1OV8hb-9afhXsB',
            'extract': extract_zip,
            'pickle': '1MQAmNV4ABoOASbdY34uMkJr8R3Osy2Wo',
        },
        'IMDBMulti': {
            'id': '1BRo5d3axRGRx5g73ecYevIWDbIb8xfX0',
            'extract': extract_zip,
            'pickle': '133lMTI_cmzFG_NNvXsdu_7lzEvwGWxwO',
        },
    }

    types = [
        'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
        'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
        'Sb', 'Se', 'Ni', 'Te'
    ]

    def __init__(self,
                 root,
                 name,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        assert self.name in self.datasets.keys()
        super(MCSDataset, self).__init__(root, transform, pre_transform,
                                         pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        path = osp.join(self.processed_dir, '{}_mcs.pt'.format(self.name))
        # if not osp.exists(path):
        self.process()
        self.mcs = torch.load(path)
        path = osp.join(self.processed_dir, '{}_norm_mcs.pt'.format(self.name))
        self.norm_mcs = torch.load(path)
        self.max_num_nodes = int(torch.max(self.data.x).item()) + 1

    @property
    def raw_file_names(self):
        return [osp.join(self.name, s) for s in ['train', 'test']]

    @property
    def processed_file_names(self):
        return ['{}_{}.pt'.format(self.name, s) for s in ['training', 'test']]

    def download(self):
        name = self.datasets[self.name]['id']
        path = download_url(self.url.format(name), self.raw_dir)
        self.datasets[self.name]['extract'](path, self.raw_dir)
        os.unlink(path)

        name = self.datasets[self.name]['pickle']
        path = download_url(self.url.format(name), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, self.name, 'mcs.pickle'))

    def process(self):
        ids, Ns = [], []
        for r_path, p_path in zip(self.raw_paths, self.processed_paths):
            names = glob.glob(osp.join(r_path, '*.gexf'))
            ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))

            data_list = []
            for i, idx in enumerate(ids[-1]):
                i = i if len(ids) == 1 else i + len(ids[0])
                gr = nx.read_gexf(osp.join(r_path, '{}.gexf'.format(idx)))
                mapping = {name: j for j, name in enumerate(gr.nodes())}
                gr = nx.relabel_nodes(gr, mapping)
                Ns.append(gr.number_of_nodes())
                edge_index = torch.tensor(list(gr.edges)).t().contiguous()
                if edge_index.numel() == 0:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_index = to_undirected(edge_index, num_nodes=Ns[-1])

                data = Data(edge_index=edge_index, i=i)
                data.num_nodes = Ns[-1]

                if self.name == 'AIDS700nef':
                    x = torch.zeros(data.num_nodes, dtype=torch.long)
                    for node, info in gr.nodes(data=True):
                        x[int(node)] = self.types.index(info['type'])
                    data.x = fn.one_hot(
                        x, num_classes=len(self.types)).to(torch.float)
                else:
                    x = torch.zeros(data.num_nodes, dtype=torch.long)
                    for n in range(len(gr.nodes)):
                        x[n] = n
                    data.x = x.unsqueeze(1).to(torch.float)   # fn.one_hot(x, num_classes=data.num_nodes)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            torch.save(self.collate(data_list), p_path)

        assoc = {idx: i for i, idx in enumerate(ids[0])}
        assoc.update({idx: i + len(ids[0]) for i, idx in enumerate(ids[1])})

        path = osp.join(self.raw_dir, self.name, 'mcs.pickle')
        mat = torch.full((len(assoc), len(assoc)), float('inf'))
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            xs, ys, gs = [], [], []
            for (x, y), g in obj.items():
                xs += [assoc[x]]
                ys += [assoc[y]]
                gs += [g]
            x, y, g = torch.tensor(xs), torch.tensor(ys), torch.tensor(gs, dtype=torch.float)
            mat[x, y], mat[y, x] = g, g

        path = osp.join(self.processed_dir, '{}_mcs.pt'.format(self.name))
        torch.save(mat, path)

        N = torch.tensor(Ns, dtype=torch.float)
        norm_mat = mat / (0.5 * (N.view(-1, 1) + N.view(1, -1)))

        path = osp.join(self.processed_dir, '{}_norm_mcs.pt'.format(self.name))
        torch.save(norm_mat, path)

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


if __name__ == "__main__":
    ds = 'IMDBMulti'
    dl = MCSDataset(root='./data/{}'.format(ds), name=ds)