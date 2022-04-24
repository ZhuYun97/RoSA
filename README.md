# RoSA: A Robust Self-Aligned Framework for Node-Node Graph Contrastive Learning
This repo is Pytorch implemention of <br>[RoSA: A Robust Self-Aligned Framework for Node-Node Graph Contrastive Learning]()[\[poster\]]()[\[appendix\]]() <br><br>
Yun Zhu\*, Jianhao Guo\*, Fei Wu, Siliang Tang† <br><br>
In IJCAI 2022 <br>

## Overview
This is the first work dedicated to solving non-aligned node-node graph contrastive learning problems. To tackle the non-aligned problem, we introduce a novel graph-based optimal transport algorithm, g-EMD, which does not require explicit node-node correspondence and can fully utilize graph topological and attributive information for non-aligned node-node contrasting. Moreover, to compensate for the possible information loss caused by non-aligned sub-sampling, we propose a nontrivial unsupervised graph adversarial training to improve the diversity of sub-sampling and strengthen the robustness of the model. The overview of our method is depicted in Figure 1.
![FRAMEWORK](./assets/framework.PNG)

## Files
```
   .
    ├── dataset_apis                  # Code process datasets.
    │   ├── topology_dist             # Storing the distance of the shortest path (SPD) between vi and vj.
    │   ├── citeseer.py               # processing for citeseer dataset.
    │   ├── cora.py                   # processing for cora dataset. 
    │   ├── dblp.py                   # processing for dblp dataset.
    │   ├── pubmed.py                 # processing for pubmed dataset. 
    │   └── ...                       # More datasets will be added.
    │
    ├── adversarial.py                # Code for unsupervised adversarial training.
    ├── augmentation.py               # Code for augmentation.
    ├── config.yaml                   # Configurations for our method.
    ├── eval.py                       # Code for evaluation.
    ├── global_var.py                 # Code for storing global variable.
    ├── model.py                      # Code for building up model.
    ├── train.py                      # Training process.
    └── ...
```

## Setup
Recommand you to set up a Python virtual environment with the required dependencies as follows:
```
conda create -n rosa python==3.9
conda activate rosa 
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
```
## Usage
**Command for  training model on Cora dataset**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True
```
Now supported datasets include Cora, Citeseer, Pubmed, DBLP. More datasets are coming soon!

### Illustration of arguements

```
--dataset: default Cora, [Cora, Citeseer, Pubmed, DBLP] can also be choosen
--rectified: defalut False, use rectified cost matrix instead of vanilla cost matrix (if True)
--ad: default False, use unsupervised adversarial training (if True)
--aligned: default False,  use aligned views (if True)
```

## Citation
If you use this code for you research, please cite our paper. TBD