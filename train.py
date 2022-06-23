from torch_geometric.data import Batch
import numpy as np
import torch
import os
import argparse
import importlib
import random
from torch_geometric.data import DataLoader
from torch_geometric.utils import contains_isolated_nodes, to_networkx
from augmentation import RWR, DropEdgeAndFeature
from model import RoSA
from adversarial import ad_training
from eval_utils import eval
import global_var
import networkx as nx
import yaml
from yaml import SafeLoader
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

def train(args):
    trans = RWR(walk_step=args.walk_step, graph_num=args.graph_num, restart_ratio=args.restart_ratio, aligned=args.aligned, inductive=args.inductive)
    load_dataset = getattr(importlib.import_module(f"dataset_apis.{args.dataset.lower()}"), 'load_trainset')
    dataset = load_dataset(trans)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Model
    model = RoSA(
        # model
        encoder=args.encoder,
        # shape
        input_dim=args.input_dim,
        # model configuration
        layer_num=args.layer_num,
        hidden=args.hidden,
        proj_shape=(args.proj_middim, args.proj_outdim),
        # loss
        is_rectified=args.rectified,
        T = args.tau,
        topo_t=args.topo_t
    ).to(device)

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # save checkpoints
    path = f'./checkpoints/{args.dataset}/'
    if not os.path.exists(path):
        os.makedirs(path)

    patience = args.patience # early stopping
    stop_cnt = 0
    best = 9999
    best_t = 0
    
    loop = tqdm(range(args.epochs))
    for epoch in loop:
        model.train()
        for idx, graphs in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # transform the [batch, batch, batch] into one batch
            view1_list = []
            view2_list = []
            assert len(graphs[0]) == len(graphs[1])
            all_graphs_num = len(graphs[0])
            for i in range(all_graphs_num):
                view1_list.extend(graphs[0][i].to_data_list())
                view2_list.extend(graphs[1][i].to_data_list())

            # shuffle views list
            shuf_idx = np.random.permutation(all_graphs_num)
            view1_list_shuff = [view1_list[i] for i in shuf_idx]
            view2_list_shuff = [view2_list[i] for i in shuf_idx]

            views1 = Batch().from_data_list(view1_list_shuff).to(device)
            views2 = Batch().from_data_list(view2_list_shuff).to(device)
            # additional augmentation
            drop = DropEdgeAndFeature(d_fea=args.drop_feature_rate_1, d_edge=args.drop_edge_rate_1)
            views1 = drop.drop_fea_edge(views1)
            drop.set_drop_ratio(d_fea=args.drop_feature_rate_2, d_edge=args.drop_edge_rate_2)
            views2 = drop.drop_fea_edge(views2)

            if args.ad:
                def node_attack(perturb):
                    views1.x += perturb
                    return model(views1, views2, views1.batch, views2.batch)

                loss = ad_training(model, node_attack, views1.x.shape, args, device)
            else:
                loss = model(views1, views2, views1.batch, views2.batch)
            loop.set_postfix(loss = loss.item())
            loss.backward()
            optimizer.step()

        if loss < best:
            stop_cnt = 0
            best = loss
            best_t = epoch + 1
            torch.save(model.state_dict(), os.path.join(path, 'best.pt'))
        else:
            stop_cnt += 1
        
        if stop_cnt >= patience:
            print("early stopping")
            break
    
    if patience < args.epochs:
        print('Loading {}th epoch'.format(best_t))
        model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
            
    return model

def register_topology(args):
    MAX_HOP = 100
    load_dataset = getattr(importlib.import_module(f"dataset_apis.{args.dataset.lower()}"), 'load_eval_trainset')
    data =  load_dataset()[0]
    topo_file = f"./dataset_apis/topology_dist/{args.dataset.lower()}_padding.pt"
    exist = os.path.isfile(topo_file)
    if not exist:
        node_num = data.x.shape[0]
        G = to_networkx(data)
        generator = dict(nx.shortest_path_length(G))
        topology_dist = torch.zeros((node_num+1, node_num+1)) # we shift the node index with 1, in order to store 0-index for padding nodes
        mask = torch.zeros((node_num+1, node_num+1)).bool()

        topology_dist[0, :] = 1000 # used for padding nodes 
        topology_dist[:, 0] = 1000

        for i in tqdm(range(1, node_num+1)):
            # print(f"processing {i}-th node")
            for j in range(1, node_num+1):
                if j-1 in generator[i-1].keys():
                    topology_dist[i][j] = generator[i-1][j-1]
                else:
                    topology_dist[i][j] = MAX_HOP
                    mask[i][j] = True # record nodes that do not have connections
        torch.save(topology_dist, topo_file)
    else:
        topology_dist = torch.load(topo_file)
    global_var._init()
    global_var.set_value("topology_dist", topology_dist)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','y','true','t','1'):
        return True
    if v.lower() in ('no','n','false','f','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    parser.add_argument('--config', type=str, default='config.yaml')
    # dataset
    parser.add_argument('--dataset',help='Cora, Citeseer , Pubmed, etc. Default=Cora', default='Cora')
    # augmentation
    parser.add_argument('--aligned', type=str2bool, help='aligned views or not', default=False)
    # adversarial training
    parser.add_argument('--ad', type=str2bool, help='combine with adversarial training', default=False)
    parser.add_argument('--step-size', type=float, default=1e-3)
    parser.add_argument('--m', type=int, default=3)
    # loss
    parser.add_argument('--rectified', type=str2bool, help='use rectified cost matrix', default=False)
    parser.add_argument('--topo_t', type=int, help='temperature for sigmoid', default=2)

    return parser

if __name__ == "__main__":
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        exit()
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    # combine args and config
    for k, v in config.items():
        args.__setattr__(k, v)

    # repeated experiment
    torch.manual_seed(args.seed)
    random.seed(12345)

    if args.rectified:
        register_topology(args)

    model = train(args)
    test_acc = eval(args, model, device)






