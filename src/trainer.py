import os
import os.path as osp
import re

import torch
import torch.nn.functional as fn

import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import spearmanr, kendalltau

from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree

from data.ged import GEDDataset
from data.mcs import MCSDataset

from model import GraphMatchTR
from utils import parameters_count, load_checkpoint, make_checkpoint, \
    gen_pairs, calculate_ranking_correlation, calculate_prec_at_k
from tqdm import tqdm, trange


class GraphMatchTrainer(object):
    """
    DualGraphSim model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.training_graphs = self.testing_graphs = None
        self.norm_metric_matrix = None
        self.optimizer = None
        self.num_labels = 0
        self.max_num_nodes = 0
        self.synth_data_1 = self.synth_data_2 = []
        self.process_dataset()
        self.model = GraphMatchTR(self.args)
        self.num_params = parameters_count(self.model)
        self.rho = self.tau = self.prec_at_10 = self.prec_at_20 = self.model_error = 0.

    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")
        path = osp.join(self.args.dataset_root, self.args.metric, self.args.dataset_name)
        if bool(re.search('(ged|GED)', self.args.metric)):
            self.training_graphs = GEDDataset(path, self.args.dataset_name, train=True)
            self.testing_graphs = GEDDataset(path, self.args.dataset_name, train=False)
            self.norm_metric_matrix = self.training_graphs.norm_ged
        else:
            self.training_graphs = MCSDataset(path, self.args.dataset_name, train=True)
            self.testing_graphs = MCSDataset(path, self.args.dataset_name, train=False)
            self.norm_metric_matrix = self.training_graphs.norm_mcs

        real_data_size = self.norm_metric_matrix.size(0)
        self.max_num_nodes = max([max([n.num_nodes for n in self.training_graphs]),
                                  max([n.num_nodes for n in self.testing_graphs])])

        if self.args.synth:
            self.synth_data_1, self.synth_data_2, _, synth_nged_matrix = \
                gen_pairs(self.training_graphs.shuffle()[:500], 0, 3)

            real_data_size = self.norm_metric_matrix.size(0)
            synth_data_size = synth_nged_matrix.size(0)
            self.norm_metric_matrix = torch.cat(
                (self.norm_metric_matrix, torch.full((real_data_size, synth_data_size), float('inf'))), dim=1)
            synth_nged_matrix = torch.cat(
                (torch.full((synth_data_size, real_data_size), float('inf')), synth_nged_matrix), dim=1)
            self.norm_metric_matrix = torch.cat((self.norm_metric_matrix, synth_nged_matrix))

        if self.training_graphs[0].x is None:
            max_degree = 0
            for g in self.training_graphs + self.testing_graphs + (
                    self.synth_data_1 + self.synth_data_2 if self.args.synth else []):
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree

            # labeling of synth data according to real data format
            if self.args.synth:
                for g in self.synth_data_1 + self.synth_data_2:
                    g = one_hot_degree(g)
                    g.i = g.i + real_data_size
        elif self.args.synth:
            for g in self.synth_data_1 + self.synth_data_2:
                g.i = g.i + real_data_size

        self.num_labels = self.training_graphs.num_features
        self.args.num_labels = self.training_graphs.num_features

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        if self.args.synth:
            synth_data_ind = random.sample(range(len(self.synth_data_1)), 100)
        else:
            synth_data_ind = 0

        source_loader = DataLoader(self.training_graphs.shuffle() +
                                   ([self.synth_data_1[i] for i in synth_data_ind] if self.args.synth else []),
                                   batch_size=self.args.batch_size)
        target_loader = DataLoader(self.training_graphs.shuffle() +
                                   ([self.synth_data_2[i] for i in synth_data_ind] if self.args.synth else []),
                                   batch_size=self.args.batch_size)

        return list(zip(source_loader, target_loader))

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]

        normalized_ged = self.norm_metric_matrix[data[0]["i"].reshape(-1).tolist(), 
                                                 data[1]["i"].reshape(-1).tolist()].tolist()
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        return new_data

    def process_batch(self, data):
        """
        Forward pass with a data.
        :param data: Data that is essentially pair of batches, for source and target graphs.
        :return loss: Loss on the data.
        """
        self.optimizer.zero_grad()
        data = self.transform(data)
        target = data["target"]
        prediction = self.model(data['g1'], data['g2'])
        loss = fn.mse_loss(prediction, target, reduction='sum')
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # def initialize_model_weights(self, model):
    #     print("\nInitializing model.\n")
    #     for m in model.modules():
    #         if isinstance(m, (Conv2d, Linear)):
    #             init.xavier_uniform_(m.weight, gain=1)
    #         elif isinstance(m, ModuleList):
    #             for sub_m in m.modules():
    #                 if isinstance(sub_m, (Conv2d, Linear)):
    #                     init.xavier_uniform_(sub_m.weight, gain=1)

    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        print("Params Count: {}".format(self.num_params))
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.model.train()
        prev_epoch = 0
        loss_list = []
        checkpoint_root = osp.join(self.args.checkpoint_path, self.args.dataset_name)
        checkpoint_path = osp.join(checkpoint_root, 'ged.pt')
        if self.args.save_model and osp.exists(checkpoint_path):
            # checkpoint_path = osp.join(self.args.checkpoint_path, self.args.dataset, '_ged')
            prev_epoch, loss_list = load_checkpoint(checkpoint_path,
                                                    model=self.model,
                                                    optimizer=self.optimizer)

        epochs = trange((self.args.epochs - prev_epoch), total=self.args.epochs, initial=prev_epoch,
                        leave=True, desc="Epoch")
        loss_list_test = []
        for epoch in epochs:
            if self.args.plot:
                if epoch % 10 == 0:
                    self.model.train(False)
                    cnt_test = 20  # int(20 * self.args.valid_batch_factor)
                    cnt_train = 100  # int(100 * self.args.valid_batch_factor)
                    t = tqdm(total=cnt_test * cnt_train, position=2, leave=False, desc="Validation")
                    scores = torch.empty((cnt_test, cnt_train))

                    for i, g in enumerate(self.testing_graphs[:cnt_test].shuffle()):
                        source_batch = Batch.from_data_list([g] * cnt_train)
                        target_batch = Batch.from_data_list(self.training_graphs[:cnt_train].shuffle())
                        data = self.transform((source_batch, target_batch))
                        target = data["target"]
                        prediction = self.model(data['g1'], data['g2'])

                        scores[i] = fn.mse_loss(prediction, target, reduction='none').detach()
                        t.update(cnt_train)

                    t.close()
                    loss_list_test.append(scores.mean().item())
                    self.model.train(True)

            batches = self.create_batches()
            main_index = 0
            loss_sum = 0
            for _, batch_pair in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch_pair)
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss_score
            loss = loss_sum / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            loss_list.append(loss)

        if self.args.save_model:
            if not osp.exists(self.args.checkpoint_path):
                os.mkdir(self.args.checkpoint_path)
            if not osp.exists(checkpoint_root):
                os.mkdir(checkpoint_root)
            make_checkpoint(checkpoint_path,
                            self.args.epochs,
                            model=self.model,
                            optimizer=self.optimizer,
                            loss=loss_list)

        if self.args.plot:
            plt.plot(loss_list, label="Train")
            plt.plot([*range(0, self.args.epochs, 10)], loss_list_test, label="Validation")
            plt.ylim([0, 0.01])
            plt.legend()
            filename = self.args.dataset_name
            filename += '_' + self.args.gnn_operator
            filename = filename + str(self.args.epochs) + '.pdf'
            plt.savefig(filename)

    def measure_time(self):
        import time
        self.model.eval()
        count = len(self.testing_graphs) * len(self.training_graphs)

        t = np.empty(count)
        i = 0
        tq = tqdm(total=count, desc="Graph pairs")
        for g1 in self.testing_graphs:
            for g2 in self.training_graphs:
                source_batch = Batch.from_data_list([g1])
                target_batch = Batch.from_data_list([g2])
                data = self.transform((source_batch, target_batch))

                start = time.process_time()
                self.model(data["g1"], data["g2"])
                t[i] = (time.process_time() - start)
                i += 1
                tq.update()
        tq.close()

        print("Average time (ms): {}; Standard deviation: {}".format(
            round(t.mean() * 1000, 5), round(t.std() * 1000, 5)))

    def score(self):
        """
        Scoring.
        """
        print("\n\nModel evaluation.\n")
        print("Params Count: {}".format(self.num_params))
        self.model.eval()

        scores = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        prediction_mat = np.empty((len(self.testing_graphs), len(self.training_graphs)))

        rho_list = []
        tau_list = []
        prec_at_10_list = []
        prec_at_20_list = []

        t = tqdm(total=len(self.testing_graphs) * len(self.training_graphs))

        for i, g in enumerate(self.testing_graphs):
            source_batch = Batch.from_data_list([g] * len(self.training_graphs))
            target_batch = Batch.from_data_list(self.training_graphs)

            data = self.transform((source_batch, target_batch))
            target = data["target"]
            ground_truth[i] = target
            prediction = self.model(data['g1'], data['g2'])
            prediction_mat[i] = prediction.detach().numpy()

            scores[i] = fn.mse_loss(prediction, target, reduction='none').detach().numpy()

            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i]))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i]))

            t.update(len(self.training_graphs))

        self.rho = np.mean(rho_list)
        self.tau = np.mean(tau_list)
        self.prec_at_10 = np.mean(prec_at_10_list)
        self.prec_at_20 = np.mean(prec_at_20_list)
        self.model_error = np.mean(scores)
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): " + str(round(self.model_error * 1000, 5)) + ".")
        print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        print("p@20: " + str(round(self.prec_at_20, 5)) + ".")
