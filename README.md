# HTLA

## Requirements
- Python >= 3.6
- torch >= 1.6.0
- transformers >= 4.2.1
- fairseq == 0.10.0
- torch-geometric == 1.7.2
- torch-scatter == 2.0.8
- torch-sparse == 0.6.12

## Data
- The repository contains tokenized versions of the WOS dataset in `data/wos` folder. This is obtained following the same way as in [contrastive-htc](https://github.com/wzh9969/contrastive-htc#preprocess).
- Specific details on how to obtain the original datasets (WOS and RCV1-V2) and the corresponding scripts  to preprocess them are mentioned in [contrastive-htc](https://github.com/wzh9969/contrastive-htc#preprocess) and will be added here as well later on.

## Train
The `train.py` can be used to train all the models by setting different arguments.  

### For BERT (does flat multi-label classification) 
`python train.py --name='ckp_bert' --batch 10 --data='wos' --graph 0` </br> </br>
Some Important arguments: </br>
- `--name` name of directory in which your model will be saved. For e.g. the above model will be saved in `./HTLA/data/wos/ckp_bert`
- `--data` name of directory which contains your data and related files
- `--graph` whether to use graph encoder
###  FOR HTLA (does Hierarchical Text Classification)
`python3 train.py --name='ckp_htla' --batch 10 --data='wos' --graph 1 --graph_type='GCN' --trpmg 1 --mg_list 0.1 0.2` </br>
</br>
Some Important arguments: </br>
- `--graph_type` type of graph encoder. Possible choices are 'GCN,'GAT' and 'graphormer'. HTLA uses GCN 
- `--trpmg` whether Hiearchical Triplet Loss required or not
- `--mg_list` margin distance for each level (WOS has two levels we use 0.1 and 0.2 as margin distance)

### Some irrelevant arguments in train.py:
Last four argumnets of train.py `--mine`, `--mine_pen`, `--netw` and `--min_proj` are part of another component which are not relevant to this work. They can be ignored. 



## Test
To run the trained model on test set run the script `test_lha.py` </br> 
`python test.py --name ckpt1 --data wos --extra _macro` </br> </br>
Some Important arguments
- `--name` The name of the directory which contains the saved checkpoint. The checkpoint is saved in `../HTLA/data/wos/`
- `--data` The name of directory which contains your data and related files
- `--extra` Two checkpoints are kept based on macro-F1 and micro-F1 respectively. The possible choices are  `_macro` and `_micro` to choose from the two checkpoints

