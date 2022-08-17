from cgi import test
from collections import OrderedDict
from eval_utils import eval
import argparse
import random
from model import RoSA
from eval_utils import eval
import yaml
from yaml import SafeLoader
import torch
import numpy as np
from tqdm import tqdm


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

    return parser

device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = RoSA(
        # model
        encoder=args.encoder,
        # shape
        input_dim=args.input_dim,
        # model configuration
        layer_num=args.layer_num,
        hidden=args.hidden,
        proj_shape=(args.proj_middim, args.proj_outdim),
    ).to(device)
    
    test_acc_list = []
    for i in tqdm(range(100)):
        pretrained_dicts = torch.load(f"G:/20runs_cora_ckpts/cora_run{i%20}.pt")
        model.load_state_dict(pretrained_dicts)
        test_acc = eval(args, model, device)
        test_acc_list.append(test_acc)
    print(f"test acc: {round(np.mean(test_acc_list)*100, 2)} ± {round(np.std(test_acc_list)*100, 2)}")