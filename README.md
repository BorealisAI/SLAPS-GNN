# SLAPS-GNN

This repo contains the implementation of the model proposed
in `SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks`.

## Datasets

`ogbn-arxiv` dataset will be loaded automatically, while `Cora`, `Citeseer`, and `Pubmed` are included in the GCN
package, available [here](https://github.com/tkipf/gcn/tree/master/gcn/data). Place the relevant files in the folder
data_tf.

## Dependencies

* `Python` version 3.7.2
* [`Numpy`](https://numpy.org/) version 1.18.5
* [`PyTorch`](https://pytorch.org/) version 1.5.1
* [`DGL`](https://www.dgl.ai/) version 0.5.2
* [`sklearn`](https://scikit-learn.org/stable/) version 0.21.3
* [`scipy`](https://www.scipy.org/) version 1.2.1
* [`torch-geometric`](https://github.com/rusty1s/pytorch_geometric) 1.6.1
* [`ogb`](https://ogb.stanford.edu/) version 1.2.3

To train the models, you need a machine with a GPU.

To install the dependencies, it is recommended to use a virtual environment. You can create a virtual environment and
install all the dependencies with the following command:

```bash
conda env create -f environment.yml
```

The file `requirements.txt` was written for CUDA 9.2 and Linux so you may need to adapt it to your infrastructure.

## Usage

To run the model you should define the following parameters:

- `dataset`: The dataset you want to run the model on
- `ntrials`: number of runs
- `epochs_adj`: number of epochs
- `epochs`: number of epochs for GNN_C (used for knn_gcn and 2step learning of the model)
- `lr_adj`: learning rate of GNN_DAE
- `lr`: learning rate of GNN_C
- `w_decay_adj`: l2 regularization parameter for GNN_DAE
- `w_decay`: l2 regularization parameter for GNN_C
- `nlayers_adj`: number of layers for GNN_DAE
- `nlayers`: number of layers for GNN_C
- `hidden_adj`: hidden size of GNN_DAE
- `hidden`: hidden size of GNN_C
- `dropout1`: dropout rate for GNN_DAE
- `dropout2`: dropout rate for GNN_C
- `dropout_adj1`: dropout rate on adjacency matrix for GNN_DAE
- `dropout_adj2`: dropout rate on adjacency matrix for GNN_C
- `dropout2`: dropout rate for GNN_C
- `k`: k for knn initialization with knn
- `lambda_`: weight of loss of GNN_DAE
- `nr`: ratio of zeros to ones to mask out for binary features
- `ratio`: ratio of ones to mask out for binary features and ratio of features to mask out for real values features
- `model`: model to run (choices are end2end, knn_gcn, or 2step)
- `sparse`: whether to make the adjacency sparse and run operations on sparse mode
- `gen_mode`: identifies the graph generator
- `non_linearity`: non-linearity to apply on the adjacency matrix
- `mlp_act`: activation function to use for the mlp graph generator
- `mlp_h`: hidden size of the mlp graph generator
- `noise`: type of noise to add to features (mask or normal)
- `loss`: type of GNN_DAE loss (mse or bce)
- `epoch_d`: epochs_adj / epoch2 of the epochs will be used for training GNN_DAE
- `half_val_as_train`: use half of validation for train to get Cora390 and Citeseer370

## Reproducing the Results in the Paper

In order to reproduce the results presented in the paper, you should run the following commands:

### Cora

#### FP

Run the following command:

```bash
python main.py -dataset cora -ntrials 10 -epochs_adj 2000 -lr 0.001 -lr_adj 0.01 -w_decay 0.0005 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 512 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.5 -dropout_adj2 0.25 -k 30 -lambda_ 10.0 -nr 5 -ratio 10 -model end2end -sparse 0 -gen_mode 0 -non_linearity elu -epoch_d 5
```

#### MLP

Run the following command:

```bash
python main.py -dataset cora -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.001 -w_decay 0.0005 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 512 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.25 -dropout_adj2 0.5 -k 20 -lambda_ 10.0 -nr 5 -ratio 10 -model end2end -sparse 0 -gen_mode 1 -non_linearity relu -mlp_h 1433 -mlp_act relu -epoch_d 5
```

### MLP-D

Run the following command:

```bash
python main.py -dataset cora -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.001 -w_decay 0.05 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 512 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.25 -dropout_adj2 0.5 -k 15 -lambda_ 10.0 -nr 5 -ratio 10 -model end2end -sparse 0 -gen_mode 2 -non_linearity relu -mlp_act relu -epoch_d 5
```

### Citeseer

#### FP

Run the following command:

```bash
python main.py -dataset citeseer -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.01 -w_decay 0.05 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 1024 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.4 -dropout_adj2 0.4 -k 30 -lambda_ 1.0 -nr 1 -ratio 10 -model end2end -sparse 0 -gen_mode 0 -non_linearity elu -epoch_d 5
```

#### MLP

Run the following command:

```bash
python main.py -dataset citeseer -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.001 -w_decay 0.0005 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 1024 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.25 -dropout_adj2 0.5 -k 30 -lambda_ 10.0 -nr 5 -ratio 10 -model end2end -sparse 0 -gen_mode 1 -non_linearity relu -mlp_act relu -mlp_h 3703 -epoch_d 5
```

#### MLP-D

Run the following command:

```bash
python main.py -dataset citeseer -ntrials 10 -epochs_adj 2000 -lr 0.001 -lr_adj 0.01 -w_decay 0.05 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 1024 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.5 -dropout_adj2 0.5 -k 20 -lambda_ 10.0 -nr 5 -ratio 10 -model end2end -sparse 0 -gen_mode 2 -non_linearity relu -mlp_act tanh -epoch_d 5
```

### Cora390

#### FP

Run the following command:

```bash
python main.py -dataset cora -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.01 -w_decay 0.0005 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 512 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.25 -dropout_adj2 0.5 -k 20 -lambda_ 100.0 -nr 5 -ratio 10 -model end2end -sparse 0 -gen_mode 0 -non_linearity elu -epoch_d 5 -half_val_as_train 1
```

#### MLP

Run the following command:

```bash
python main.py -dataset cora -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.001 -w_decay 0.0005 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 512 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.25 -dropout_adj2 0.5 -k 20 -lambda_ 10.0 -nr 5 -ratio 10 -model end2end -sparse 0 -gen_mode 1 -non_linearity relu -mlp_h 1433 -mlp_act relu -epoch_d 5 -half_val_as_train 1
```

#### MLP-D

Run the following command:

```bash
python main.py -dataset cora -ntrials 10 -epochs_adj 2000 -lr 0.001 -lr_adj 0.001 -w_decay 0.0005 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 512 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.25 -dropout_adj2 0.5 -k 20 -lambda_ 10.0 -nr 5 -ratio 10 -model end2end -sparse 0 -gen_mode 2 -non_linearity relu -mlp_act relu -epoch_d 5 -half_val_as_train 1
```

### Citeseer370

#### FP

Run the following command:

```bash
python main.py -dataset citeseer -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.01 -w_decay 0.05 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 1024 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.5 -dropout_adj2 0.5 -k 30 -lambda_ 1.0 -nr 1 -ratio 10 -model end2end -sparse 0 -gen_mode 0 -non_linearity elu -epoch_d 5 -half_val_as_train 1
```

#### MLP

Run the following command:

```bash
python main.py -dataset citeseer -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.001 -w_decay 0.0005 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 1024 -dropout1 0.25 -dropout2 0.5 -dropout_adj1 0.25 -dropout_adj2 0.5 -k 30 -lambda_ 10.0 -nr 5 -ratio 10 -model end2end -sparse 0 -gen_mode 1 -non_linearity relu -mlp_act tanh -mlp_h 3703 -epoch_d 5 -half_val_as_train 1
```

#### MLP-D

Run the following command:

```bash
python main.py -dataset citeseer -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.01 -w_decay 0.05 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 1024 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.25 -dropout_adj2 0.5 -k 20 -lambda_ 10.0 -nr 5 -ratio 10 -model end2end -sparse 0 -gen_mode 2 -non_linearity relu -mlp_act tanh -epoch_d 5 -half_val_as_train 1
```

### Pubmed

#### MLP

Run the following command:

```bash
python main.py -dataset pubmed -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.01 -w_decay 0.0005 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 128 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.5 -dropout_adj2 0.5 -k 15 -lambda_ 10.0 -nr 5 -ratio 20 -model end2end -gen_mode 1 -non_linearity relu -mlp_h 500 -mlp_act relu -epoch_d 5 -sparse 1
```

#### MLP-D

Run the following command:

```bash
python main.py -dataset pubmed -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.01 -w_decay 0.0005 -nlayers 2 -nlayers_adj 2 -hidden 32 -hidden_adj 128 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.25 -dropout_adj2 0.25 -k 15 -lambda_ 100.0 -nr 5 -ratio 20 -model end2end -sparse 0 -gen_mode 2 -non_linearity relu -mlp_act tanh -epoch_d 5 -sparse 1
```

### ogbn-arxiv

#### MLP

Run the following command:

```bash
python main.py -dataset ogbn-arxiv -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.001 -w_decay 0.0 -nlayers 2 -nlayers_adj 2 -hidden 256 -hidden_adj 256 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.25 -dropout_adj2 0.5 -k 15 -lambda_ 10.0 -nr 5 -ratio 100 -model end2end -sparse 0 -gen_mode 1 -non_linearity relu -mlp_h 128 -mlp_act relu -epoch_d 2001 -sparse 1 -loss mse -noise mask
```

#### MLP-D

Run the following command:

```bash
python main.py -dataset ogbn-arxiv -ntrials 10 -epochs_adj 2000 -lr 0.01 -lr_adj 0.001 -w_decay 0.0 -nlayers 2 -nlayers_adj 2 -hidden 256 -hidden_adj 256 -dropout1 0.5 -dropout2 0.5 -dropout_adj1 0.5 -dropout_adj2 0.25 -k 15 -lambda_ 10.0 -nr 5 -ratio 100 -model end2end -sparse 0 -gen_mode 2 -non_linearity relu -mlp_act relu -epoch_d 2001 -sparse 1 -loss mse -noise normal
```

# Cite SLAPS
If you use this package for published work, please cite the following:

    @inproceedigs{fatemi2021slaps,
      title={SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks},
      author={Fatemi, Bahare and Asri, Layla El and Kazemi, Seyed Mehran},
      booktitle={Advances in Neural Information Processing Systems},
      year={2021}
    }

