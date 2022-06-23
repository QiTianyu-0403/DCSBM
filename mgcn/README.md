# M-GCN 

This example aims to explore the relation of topology and feature's contribution to the final node classification 
task.

We contains three graph embeddings method: Node2Vec, NetMF and ProNE.


## Run
```
cd examples/mgcn
python main.py -d acm -l 20 -m netmf
python main.py -d uai -l 40 -m node2vec
```