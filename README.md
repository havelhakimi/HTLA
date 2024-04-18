# HTLA: Hierarchical Text-Label Association 

## Requirements
- Python >= 3.6
- torch >= 1.6.0
- transformers >= 4.2.1
- fairseq == 0.10.0
- torch-geometric == 2.4.0
- torch-sparse == 0.6.17

## Data
- The repository contains tokenized versions of the WOS dataset processed using the BERT tokenizer, along with its related files in the `data/wos` folder. This is obtained following the same way as in [contrastive-htc](https://github.com/wzh9969/contrastive-htc#preprocess).
- All datasets are publicly available and accessible via the following links: [WOS](https://github.com/kk7nc/HDLTex), [RCV1-V2](https://trec.nist.gov/data/reuters/reuters.html) and [NYT](https://catalog.ldc.upenn.edu/LDC2008T19). 
- We have followed the specific details outlined in the [contrastive-htc](https://github.com/wzh9969/contrastive-htc#preprocess) repository (which contains HGCLR model ) to obtain and preprocess the original datasets (WOS, RCV1-V2, and NYT). The corresponding scripts for preprocessing will be added here later on.

## Train
The `train.py` can be used to train all the models by setting different arguments.  

### For BERT (does flat multi-label classification) 
`python train.py --name='ckp_bert' --batch 10 --data='wos' --graph 0` </br> </br>
Some Important arguments: </br>
- `--name` name of directory in which your model will be saved. For e.g. the above model will be saved in `./HTLA/data/wos/ckp_bert`
- `--data` name of dataset directory which contains your data and related files
- `--graph` whether to use graph encoder
###  For HTLA (does Hierarchical Text Classification with Margin Separation Loss (MSL))
`python train.py --name='ckp_htla' --batch 10 --data='wos' --graph 1 --graph_type='graphormer'  --msl 1 --msl_pen 1 --mg_list 0.1 0.1` </br>
</br>
Some Important arguments: </br>
- `--graph_type` type of graph encoder. Possible choices are 'graphormer', 'GCN', and 'GAT'. HTLA uses graphormer as the graph encoder. The code for graph encoder is in the script graph.py 
- `--msl` whether Margin Separation Loss required or not
- `--msl_pen` weight for the MSL component (we set it to 1 for all datasets)
- `--mg_list` margin distance for each level (We use 0.1 as margin distance for each level in all datasets)
- The node feature is fixed as 768 to match the text feature size and is not included as run time argument

###  For BERT-Graphormer (does Hierarchical Text Classification without MSL)
`python train.py --name='ckp_bgrapho' --batch 10 --data='wos' --graph 1 --graph_type='graphormer' --msl 0`  </br>
### For multiple  random runs
In `train.py` set the `--seed=None` for multiple random runs



## Test
To run the trained model on test set run the script `test.py` </br> 
`python test.py --name ckp_htla --data wos --extra _macro` </br> </br>
Some Important arguments
- `--name` name of the directory which contains the saved checkpoint. The checkpoint is saved in `../HTLA/data/wos/` when working with WOS dataset
- `--data` name of dataset directory which contains your data and related files
- `--extra` two checkpoints are kept based on best macro-F1 and micro-F1 respectively. The possible choices are  `_macro` and `_micro` to choose from the two checkpoints

