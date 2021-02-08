# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import warnings

import torch

from citation_networks import load_citation_network, sample_mask

warnings.simplefilter("ignore")


def load_ogb_data(dataset_str):
    from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(dataset_str)

    data = dataset[0]
    features = data.x
    nfeats = data.num_features
    nclasses = dataset.num_classes
    labels = data.y

    split_idx = dataset.get_idx_split()

    train_mask = sample_mask(split_idx['train'], data.x.shape[0])
    val_mask = sample_mask(split_idx['valid'], data.x.shape[0])
    test_mask = sample_mask(split_idx['test'], data.x.shape[0])

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels).view(-1)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask


def load_data(args):
    dataset_str = args.dataset

    if dataset_str.startswith('ogb'):
        return load_ogb_data(dataset_str)

    return load_citation_network(dataset_str)
