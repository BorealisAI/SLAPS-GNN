# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import dgl
import torch.nn as nn

from layers import Diag
from utils import *


class FullParam(nn.Module):
    def __init__(self, features, non_linearity, k, knn_metric, i, sparse):
        super(FullParam, self).__init__()

        self.non_linearity = non_linearity
        self.k = k
        self.knn_metric = knn_metric
        self.i = i
        self.sparse = sparse

        if self.non_linearity == "exp":
            self.Adj = nn.Parameter(
                torch.from_numpy(nearest_neighbors_pre_exp(features, self.k, self.knn_metric, self.i)))
        elif self.non_linearity == "elu":
            self.Adj = nn.Parameter(
                torch.from_numpy(nearest_neighbors_pre_elu(features, self.k, self.knn_metric, self.i)))
        elif self.non_linearity == 'none':
            self.Adj = nn.Parameter(torch.from_numpy(nearest_neighbors(features, self.k, self.knn_metric)))
        else:
            raise NameError('No non-linearity has been specified')

    def forward(self, h):
        if not self.sparse:
            if self.non_linearity == "exp":
                Adj = torch.exp(self.Adj)
            elif self.non_linearity == "elu":
                Adj = F.elu(self.Adj) + 1
            elif self.non_linearity == "none":
                Adj = self.Adj
        else:
            if self.non_linearity == 'exp':
                Adj = self.Adj.coalesce()
                Adj.values = torch.exp(Adj.values())
            elif self.non_linearity == 'elu':
                Adj = self.Adj.coalesce()
                Adj.values = F.elu(Adj.values()) + 1
            elif self.non_linearity == "none":
                Adj = self.Adj
            else:
                raise NameError('Non-linearity is not supported in the sparse setup')
        return Adj


class MLP_Diag(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, non_linearity, i, sparse, mlp_act):
        super(MLP_Diag, self).__init__()

        self.i = i
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Diag(isize))
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = non_linearity
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def forward(self, features):
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities


class MLP(nn.Module):
    def __init__(self, nlayers, isize, hsize, osize, mlp_epochs, k, knn_metric, non_linearity, i, sparse, mlp_act):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(isize, hsize))
        else:
            self.layers.append(nn.Linear(isize, hsize))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(hsize, hsize))
            self.layers.append(nn.Linear(hsize, osize))

        self.input_dim = isize
        self.output_dim = osize
        self.mlp_epochs = mlp_epochs
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = non_linearity
        self.mlp_knn_init()
        self.i = i
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def mlp_knn_init(self):
        if self.input_dim == self.output_dim:
            print("MLP full")
            for layer in self.layers:
                layer.weight = nn.Parameter(torch.eye(self.input_dim))
        else:
            optimizer = torch.optim.Adam(self.parameters(), 0.01)
            labels = torch.from_numpy(nearest_neighbors(self.features.cpu(), self.k, self.knn_metric)).cuda()

            for epoch in range(1, self.mlp_epochs):
                self.train()
                logits = self.forward()
                loss = F.mse_loss(logits, labels, reduction='sum')
                if epoch % 10 == 0:
                    print("MLP loss", loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def forward(self, features):
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities
