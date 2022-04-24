from torch_geometric.datasets import CitationFull
import torch_geometric.transforms as T

def load_trainset(trans):
    dataset = CitationFull(root='~/datasets', name='dblp', transform=T.Compose([trans]))
    return dataset

def load_eval_trainset():
    return CitationFull(root='~/datasets', name='dblp')

def load_testset():
    return CitationFull(root='~/datasets', name='dblp')