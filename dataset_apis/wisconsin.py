from torch_geometric.datasets import WebKB
import torch_geometric.transforms as T

def load_trainset(trans):
    return WebKB(root='~/datasets', name='Wisconsin', transform=trans)

def load_eval_trainset():
    return WebKB(root='~/datasets', name='Wisconsin')

def load_testset():
    return WebKB(root='~/datasets', name='Wisconsin')