## Command for  training model on Cora dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=Cora --config=config.yaml
```

## Command for  training model on Citeseer dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=Cora --config=config.yaml
```

## Command for  training model on Pubmed dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=Pubmed --config=config.yaml
```

## Command for  training model on DBLP dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=DBLP --config=config.yaml
```

## Other arguements

```
--rectified: defalut False, use rectified cost matrix instead of vanilla cost matrix (if True)
--ad: default False, use unsupervised adversarial training (if True)
--aligned: default False,  use aligned views (if True)
```

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=Citeseer --config=config.yaml --ad=True --rectified=True
```