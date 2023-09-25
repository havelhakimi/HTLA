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

### BERT for falt multi-label classification 
`python train.py --name='ckp_bert' --batch 10 --data='wos' --graph 0` </br> </br>
Some Important arguments: </br>
- `--name` name of directory in which your model will be saved. For e.g. the above model will be saved in `./LHA-HTC/data/wos/ckp_bert`
- `--data` name of directory which contains your data and related files
- `--graph` whether to use graph encoder
###  FOR HTLA 
`python3 train.py --name='ckp_htla' --batch 10 --data='wos' --graph 1 --graph_type='GCN' --trpmg 1 --mg_list 0.1 0.2` </br>
</br>
Some Important arguments: </br>
- `--graph_type` type of graph encoder. Possible choices are 'GCN,'GAT' and 'graphormer'
- `--trpmg` whether Hiearchical Triplet Loss required or not
- `--mg_list` margin distance for each level (WOS has two levels we use 0.1 and 0.2 as margin distance)


### FOR LHA-CON
`python3 train_lha.py --name='ckpt_con' --batch 10 --data='wos' --graph 1  --lamb 0.05 --thre 0.02 --hsampling 1 --hcont_wt 0.4` </br>
</br>
Some Important arguments: </br>
- `--hsampling` whether to use LHA-CON module
-  `--hcont_wt` weight term of the LHA-CON module.
### FOR LHA-ADV
`python3 train_lha.py --name='ckpt_adv' --batch 10 --data='wos' --graph 1 --lamb 0.05 --thre 0.02 --label_reg 1 --prior_wt 0.5 --hlayer 900` </br> </br>
Some Important arguments: </br>
- `--label_reg` whether to use LHA-ADV
-  `--prior_wt` weight term of the LHA-ADV module
-  `--hlayer` The size of the first  hidden layer of the neural network

## Test
To run the trained model on test set run the script `test_lha.py` </br> 
`python test.py --name ckpt1 --data wos --extra _macro` </br> </br>
Some Important arguments
- `--name` The name of the directory which contains the saved checkpoint. The checkpoint is saved in `../LHA-HTC/data/wos/`
- `--data` The name of directory which contains your data and related files
- `--extra` Two checkpoints are kept based on macro-F1 and micro-F1 respectively. The possible choices are  `_macro` and `_micro` to choose from the two checkpoints

