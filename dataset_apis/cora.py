from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

def load_trainset(trans):
    dataset = Planetoid(root='~/datasets', name='Cora', transform=T.Compose([trans]))
    return dataset

def load_eval_trainset():
    return Planetoid(root='~/datasets', name='Cora')

def load_testset():
    return Planetoid(root='~/datasets', name='Cora')