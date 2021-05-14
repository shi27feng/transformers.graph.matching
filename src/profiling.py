import os.path as osp

import torch
from deepspeed.profiling.flops_profiler import get_model_profile, FlopsProfiler
from torch.profiler import profiler
from torch_geometric.data import DataLoader
from torch_geometric.data.data import Data
from tqdm import tqdm

from args_parser import parameter_parser
#from data.dataset3 import SkeletonDataset, skeleton_parts
#from models.net2s import DualGraphEncoder

from model import GraphMatchTR

from data.ged import GEDDataset
from data.mcs import MCSDataset
from utils import tab_printer
import numpy as np


# from utility.helper import load_checkpoint
#adj_mat = skeleton_parts(cat=False).to(torch.device('cpu'))
device = torch.device('cuda:0') if True and torch.cuda.is_available() else torch.device('cpu')
args = parameter_parser()
ds = torch.load("/home/dusko/Documents/projects/APBGCN/processed/xsub_val_ntu_60.pt")
path = osp.join(args.dataset_root, args.metric, args.dataset_name)
graphs = GEDDataset(path, args.dataset_name, train=True)
norm_metric_matrix = graphs.norm_ged
args.num_labels = graphs.num_features
source_loader = DataLoader(graphs.shuffle(), batch_size=args.batch_size)
target_loader = DataLoader(graphs.shuffle(), batch_size=args.batch_size)
ds = list(zip(source_loader, target_loader))

num_batches = 100
ds = ds[:args.batch_size * num_batches]
total_batch_ = int(len(ds) / args.batch_size) + 1

def transform(data, norm_metric_matrix):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]

        normalized_ged = norm_metric_matrix[data[0]["i"].reshape(-1).tolist(), 
                                                 data[1]["i"].reshape(-1).tolist()].tolist()
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        return new_data

def time_trace_handler(p):
    print(p.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=10))


def memory_trace_handler(p):
    print(p.key_averages().table(
        sort_by="cuda_memory_usage" if torch.cuda.is_available() else "cpu_memory_usage",
        row_limit=10))


def run(model, dataloader, total_batch, prof, device, norm_metric_matrix):
    model.eval()
    for i, batch in tqdm(enumerate(dataloader),
                         total=total_batch,
                         desc="Profiling"):
        batch = transform(batch, norm_metric_matrix)
        target = batch["target"].to(device)
        # with profiler.profile(record_shapes=True,
        #                   activities=[
        #                       torch.profiler.ProfilerActivity.CPU,
        #                       torch.profiler.ProfilerActivity.CUDA],
        #                   schedule=torch.profiler.schedule(
        #                       wait=1,
        #                       warmup=5,
        #                       active=2),
        #                   with_stack=True,
        #                   profile_memory=True,
        #                   on_trace_ready=memory_trace_handler) as prof:
        prediction = model(batch['g1'].to(device), batch['g2'].to(device))
        prof.step()
        #prof.export_chrome_trace(osp.join(args.save_root, "time_trace_batch_" + str(i) + ".json"))
    return prediction

test_input = transform(ds[0], norm_metric_matrix)

def input_constructor():
    return test_input['g1'].to(device), test_input['g2'].to(device)


def input_const_2(tensor_dim):
    return {'s': test_input['g1'].to(device), 't': test_input['g2'].to(device)}


def _mac_ops(model, in_const):
    """

    :param model:
    :param in_const: data = (x, adj, bi)
    :return:
    """
    s, t = input_constructor()
    macs, params = get_model_profile(model=model,  # model
                                     input_res=None, #tuple(s.shape),
                                     # input shape or input to the in_const
                                     input_constructor=in_const,
                                     # if specified, a constructor taking input_res is used as input to the model
                                     print_profile=True,
                                     # prints the model graph with the measured profile attached to each module
                                     # print_aggregated_profile=True,  # print the aggregated profile for the top modules
                                     module_depth=-1,
                                     # depth into the nested modules with -1 being the inner most modules
                                     top_modules=3,  # the number of top modules to print aggregated profile
                                     warm_up=10,  # the number of warm-ups before measuring the time of each module
                                     as_string=True,
                                     # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                     ignore_modules=None)  # the list of modules to ignore in the profiling
    print("{:<30}  {:<8}".format("Batch size: ", torch.max(bi).item() + 1))
    print('{:<30}  {:<8}'.format('Number of MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def mac_ops(model):

    warm_up=10
    prof = FlopsProfiler(model)
    model.eval()

    for _ in range(warm_up):
        _ = model(test_input['g1'].to(device), test_input['g2'].to(device))

    prof.start_profile(ignore_list=None)

    _ = model(test_input['g1'].to(device), test_input['g2'].to(device))


    flops = prof.get_total_flops()
    params = prof.get_total_params()
    prof.print_model_profile(profile_step=warm_up,
                                module_depth=1,
                                top_modules=3,
                                detailed=True)

    prof.end_profile()


    #print("{:<30}  {:<8}".format("Batch size: ", torch.max(bi).item() + 1))
    print('{:<30}  {:<8}'.format('Number of MACs: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def profile(device, args):
    #ds = SkeletonDataset(_args.dataset_root, name='ntu_60',
    #                     num_channels=args.in_channels, sample='val')
    # Load model
    model = GraphMatchTR(args).to(device)

    print('Profiling of performance ....')

    with profiler.profile(record_shapes=True,
                          activities=[
                              torch.profiler.ProfilerActivity.CPU,
                              torch.profiler.ProfilerActivity.CUDA] if device == 'cuda' else
                          [torch.profiler.ProfilerActivity.CPU],
                          schedule=torch.profiler.schedule(
                              wait=1,
                              warmup=5,
                              active=2),
                          on_trace_ready=time_trace_handler
                          ) as prof:
        _ = run(model, ds, total_batch_, prof, device, norm_metric_matrix)

    prof.export_chrome_trace(osp.join(args.save_root, "time_trace.json"))

    print('Profiling of memory usage ....')
    with profiler.profile(record_shapes=True,
                          activities=[
                              torch.profiler.ProfilerActivity.CPU,
                              torch.profiler.ProfilerActivity.CUDA],
                          schedule=torch.profiler.schedule(
                              wait=1,
                              warmup=5,
                              active=2),
                          with_stack=True,
                          profile_memory=True,
                          on_trace_ready=memory_trace_handler) as prof:
        _ = run(model, ds, total_batch_, prof, device, norm_metric_matrix)

    prof.export_chrome_trace(osp.join(args.save_root, "memory_trace.json"))

    mac_ops(model)


if __name__ == '__main__':
    tab_printer(args)
    #dev = torch.device('cpu')
    profile(device=device, args=args)
