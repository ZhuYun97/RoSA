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
import os, wget


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

eval_seeds = {
    'Cora': 8085,
    'Citeseer': 2230
}

download_url = 'https://raw.githubusercontent.com/ZhuYun97/RoSA_ckpts/main/'

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
    seed = eval_seeds[args.dataset]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True

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

    dir = f'./runs_ckpts_{args.dataset}'
    exist_ckpts = os.path.exists(dir)
    if not exist_ckpts:
        os.makedirs(dir)
    
    test_acc_list = []
    progress = tqdm(range(20))
    for i in progress:
        file = os.path.join(dir, f'{args.dataset}_run{i}.pt')
        if not os.path.exists(file):
            # download non-existing ckpt
            file_url = os.path.join(download_url, f'runs_ckpts_{args.dataset}/{args.dataset}_run{i}.pt')
            wget.download(file_url, file)
        pretrained_dicts = torch.load(file)
        model.load_state_dict(pretrained_dicts)
        test_acc = eval(args, model, device)
        test_acc_list.append(test_acc)
        progress.set_postfix({'RUN': i, 'ACC': test_acc*100})
    print(f"After 20 runs, test acc: {round(np.mean(test_acc_list)*100, 2)} ± {round(np.std(test_acc_list)*100, 2)}")