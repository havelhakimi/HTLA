# HTLA<sup>n</sup>: Hierarchical Text-Label Association 
Implementation for the 2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA) accepted paper "**Local Hierarchy-Aware Text-Label Association for Hierarchical Text Classification**" [paper-link](https://ieeexplore.ieee.org/abstract/document/10722840)
## Requirements
- Python >= 3.6
- torch >= 1.6.0
- transformers >= 4.30.2
- Below libraries only if you want to run on GAT/GCN as the graph encoder
  - torch-geometric == 2.4.0
  - torch-sparse == 0.6.17
  - torch-scatter == 2.1.1
## Data
- All datasets are publically available and can be accessed at [WOS](https://github.com/kk7nc/HDLTex), [RCV1-V2](https://trec.nist.gov/data/reuters/reuters.html) and [NYT](https://catalog.ldc.upenn.edu/LDC2008T19). 
- We followed the specific details mentioned in the  [contrastive-htc](https://github.com/wzh9969/contrastive-htc#preprocess) repository to obtain and preprocess the original datasets (WOS, RCV1-V2, and NYT).
- After accessing the dataset, run the scripts in the folder `preprocess` for each dataset separately to obtain tokenized version of dataset and the related files. These will be added in the `data/x` folder where x is the name of dataset with possible choices as: wos, rcv and nyt.
- Detailed steps regarding how to obtain and preprocess each dataset are mentioned in the readme file of `preprocess` folder 
- For reference we have added tokenized versions of the WOS and NYT dataset along with its related files in the `data` folder. The RCV1-V2 dataset exceeds 400 MB in size, which prevents us from uploading it to GitHub due to size constraints.

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


## Citation
If you find our work helpful, please cite it using the following BibTeX entry:
```bibtex
@INPROCEEDINGS{10722840,
  author={Kumar, Ashish and Toshniwal, Durga},
  booktitle={2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA)}, 
  title={Local Hierarchy-Aware Text-Label Association for Hierarchical Text Classification}, 
  year={2024},
  volume={},
  number={},
  pages={1-10},
  doi={10.1109/DSAA61799.2024.10722840}}

