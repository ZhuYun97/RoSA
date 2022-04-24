# RoSA: A Robust Self-Aligned Framework for Node-Node Graph Contrastive Learning
This repo is pyg implemention of [RoSA: A Robust Self-Aligned Framework for Node-Node Graph Contrastive Learning]()[\[poster\]]()[\[appendix\]]() <br>
Yun Zhu\*, Jianhao Guo\*, Fei Wu, Siliang Tang† <br>
In IJCAI 2022 <br>

## overview
![FRAMEWORK](./assets/framework.PNG)

## Usage
### Command for  training model on Cora dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True
```

### Command for  training model on Citeseer dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=Cora --config=config.yaml --ad=True --rectified=True
```

### Command for  training model on Pubmed dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=Pubmed --config=config.yaml --ad=True --rectified=True
```

### Command for  training model on DBLP dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=DBLP --config=config.yaml --ad=True --rectified=True
```

### Other arguements

```
--rectified: defalut False, use rectified cost matrix instead of vanilla cost matrix (if True)
--ad: default False, use unsupervised adversarial training (if True)
--aligned: default False,  use aligned views (if True)
```
