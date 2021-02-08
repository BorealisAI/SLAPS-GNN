# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import load_data
from model import GCN, GCN_C, GCN_DAE
from utils import accuracy, get_random_mask, get_random_mask_ogb, nearest_neighbors, normalize

EOS = 1e-10


class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()

    def get_loss_learnable_adj(self, model, mask, features, labels, Adj):
        logits = model(features, Adj)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def get_loss_fixed_adj(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def get_loss_adj(self, model, features, feat_ind):
        labels = features[:, feat_ind].float()
        new_features = copy.deepcopy(features)
        new_features[:, feat_ind] = torch.zeros(new_features[:, feat_ind].shape)
        logits = model(new_features)
        loss = F.binary_cross_entropy_with_logits(logits[:, feat_ind], labels, weight=labels + 1)
        return loss

    def get_loss_masked_features(self, model, features, mask, ogb, noise, loss_t):
        if ogb:
            if noise == 'mask':
                masked_features = features * (1 - mask)
            elif noise == "normal":
                noise = torch.normal(0.0, 1.0, size=features.shape).cuda()
                masked_features = features + (noise * mask)

            logits, Adj = model(features, masked_features)
            indices = mask > 0

            if loss_t == 'bce':
                features_sign = torch.sign(features).cuda() * 0.5 + 0.5
                loss = F.binary_cross_entropy_with_logits(logits[indices], features_sign[indices], reduction='mean')
            elif loss_t == 'mse':
                loss = F.mse_loss(logits[indices], features[indices], reduction='mean')
        else:
            masked_features = features * (1 - mask)
            logits, Adj = model(features, masked_features)
            indices = mask > 0
            loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')
        return loss, Adj

    def half_val_as_train(self, val_mask, train_mask):
        val_size = np.count_nonzero(val_mask)
        counter = 0
        for i in range(len(val_mask)):
            if val_mask[i] and counter < val_size / 2:
                counter += 1
                val_mask[i] = False
                train_mask[i] = True
        return val_mask, train_mask

    def train_classification_gcn(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):

        model = GCN(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses, num_layers=args.nlayers,
                    dropout=args.dropout2, dropout_adj=args.dropout_adj2, Adj=Adj, sparse=args.sparse)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

        bad_counter = 0
        best_val = 0
        best_model = None
        best_loss = 0
        best_train_loss = 0

        if torch.cuda.is_available():
            model = model.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            test_mask = test_mask.cuda()
            features = features.cuda()
            labels = labels.cuda()

        for epoch in range(1, args.epochs + 1):
            model.train()
            loss, accu = self.get_loss_fixed_adj(model, train_mask, features, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                val_loss, accu = self.get_loss_fixed_adj(model, val_mask, features, labels)
                if accu > best_val:
                    bad_counter = 0
                    best_val = accu
                    best_model = copy.deepcopy(model)
                    best_loss = val_loss
                    best_train_loss = loss
                else:
                    bad_counter += 1

                if bad_counter >= args.patience:
                    break

        print("Val Loss {:.4f}, Val Accuracy {:.4f}".format(best_loss, best_val))
        best_model.eval()
        test_loss, test_accu = self.get_loss_fixed_adj(best_model, test_mask, features, labels)
        print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accu))
        return best_val, test_accu, best_model

    def train_knn_gcn(self, args):
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_data(args)
        val_accuracies = []
        test_accuracies = []

        Adj = torch.from_numpy(nearest_neighbors(features, args.k, args.knn_metric)).cuda()
        Adj = normalize(Adj, args.normalization, args.sparse)

        if torch.cuda.is_available():
            features = features.cuda()

        if args.half_val_as_train:
            val_mask, train_mask = self.half_val_as_train(val_mask, train_mask)

        for trial in range(args.ntrials):
            val_accu, test_accu, best_model = self.train_classification_gcn(Adj, features, nfeats, labels, nclasses,
                                                                            train_mask, val_mask, test_mask, args)
            val_accuracies.append(val_accu.item())
            test_accuracies.append(test_accu.item())

        self.print_results(val_accuracies, test_accuracies)

    def train_two_steps(self, args):
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_data(args)

        if args.half_val_as_train:
            val_mask, train_mask = self.half_val_as_train(val_mask, train_mask)

        test_accuracies = []
        validation_accuracies = []

        for trial in range(args.ntrials):

            model = GCN_DAE(nlayers=args.nlayers_adj, in_dim=nfeats, hidden_dim=args.hidden_adj, nclasses=nfeats,
                            dropout=args.dropout1, dropout_adj=args.dropout_adj1,
                            features=features.cpu(), k=args.k, knn_metric=args.knn_metric, i_=args.i,
                            non_linearity=args.non_linearity, normalization=args.normalization, mlp_h=args.mlp_h,
                            mlp_epochs=args.mlp_epochs, gen_mode=args.gen_mode, sparse=args.sparse,
                            mlp_act=args.mlp_act)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_adj, weight_decay=args.w_decay_adj)

            if torch.cuda.is_available():
                model = model.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()

            best_val = 0
            best_val_test = 0
            for epoch in range(1, args.epochs_adj + 1):

                model.train()
                if args.dataset.startswith('ogb') or args.dataset in ["wine", "digits", "breast_cancer"]:
                    mask = get_random_mask_ogb(features, args.ratio).cuda()
                    ogb = True
                elif args.dataset == "20news10":
                    mask = get_random_mask(features, args.ratio, args.nr).cuda()
                    ogb = True
                else:
                    mask = get_random_mask(features, args.ratio, args.nr).cuda()
                    ogb = False

                loss, Adj = self.get_loss_masked_features(model, features, mask, ogb, args.noise, args.loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("Epoch {:05d} | Train Loss {:.4f}".format(epoch, loss.item()))

                if epoch % 1 == 0:
                    model.eval()
                    accu, test_accu, classification_model = self.train_classification_gcn(Adj.detach(), features,
                                                                                          nfeats, labels, nclasses,
                                                                                          train_mask, val_mask,
                                                                                          test_mask, args)
                    if accu > best_val:
                        best_val = accu
                        best_val_test = test_accu

            validation_accuracies.append(best_val.item())
            test_accuracies.append(best_val_test.item())

        self.print_results(validation_accuracies, test_accuracies)

    def train_end_to_end(self, args):
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_data(args)

        if args.half_val_as_train:
            val_mask, train_mask = self.half_val_as_train(val_mask, train_mask)

        test_accu = []
        validation_accu = []
        added_edges_list = []
        removed_edges_list = []

        for trial in range(args.ntrials):
            model1 = GCN_DAE(nlayers=args.nlayers_adj, in_dim=nfeats, hidden_dim=args.hidden_adj, nclasses=nfeats,
                             dropout=args.dropout1, dropout_adj=args.dropout_adj1,
                             features=features.cpu(), k=args.k, knn_metric=args.knn_metric, i_=args.i,
                             non_linearity=args.non_linearity, normalization=args.normalization, mlp_h=args.mlp_h,
                             mlp_epochs=args.mlp_epochs, gen_mode=args.gen_mode, sparse=args.sparse,
                             mlp_act=args.mlp_act)
            model2 = GCN_C(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                           num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                           sparse=args.sparse)

            optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr_adj, weight_decay=args.w_decay_adj)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.w_decay)

            if torch.cuda.is_available():
                model1 = model1.cuda()
                model2 = model2.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()

            best_val_accu = 0.0
            best_model2 = None
            best_Adj = None

            for epoch in range(1, args.epochs_adj + 1):
                model1.train()
                model2.train()

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                if args.dataset.startswith('ogb') or args.dataset in ["wine", "digits", "breast_cancer"]:
                    mask = get_random_mask_ogb(features, args.ratio).cuda()
                    ogb = True
                elif args.dataset == "20news10":
                    mask = get_random_mask(features, args.ratio, args.nr).cuda()
                    ogb = True
                else:
                    mask = get_random_mask(features, args.ratio, args.nr).cuda()
                    ogb = False

                if epoch < args.epochs_adj // args.epoch_d:
                    model2.eval()
                    loss1, Adj = self.get_loss_masked_features(model1, features, mask, ogb, args.noise, args.loss)
                    loss2 = torch.tensor(0).cuda()
                else:
                    loss1, Adj = self.get_loss_masked_features(model1, features, mask, ogb, args.noise, args.loss)
                    loss2, accu = self.get_loss_learnable_adj(model2, train_mask, features, labels, Adj)

                loss = loss1 * args.lambda_ + loss2
                loss.backward()
                optimizer1.step()
                optimizer2.step()

                if epoch % 100 == 0:
                    print("Epoch {:05d} | Train Loss {:.4f}, {:.4f}".format(epoch, loss1.item() * args.lambda_,
                                                                            loss2.item()))

                if epoch >= args.epochs_adj // args.epoch_d and epoch % 1 == 0:
                    with torch.no_grad():
                        model1.eval()
                        model2.eval()

                        val_loss, val_accu = self.get_loss_learnable_adj(model2, val_mask, features, labels, Adj)
                        if val_accu > best_val_accu:
                            best_val_accu = val_accu
                            print("Val Loss {:.4f}, Val Accuracy {:.4f}".format(val_loss, val_accu))
                            test_loss_, test_accu_ = self.get_loss_learnable_adj(model2, test_mask, features, labels,
                                                                                 Adj)
                            print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss_, test_accu_))

            validation_accu.append(best_val_accu.item())
            model1.eval()
            model2.eval()

            with torch.no_grad():
                print("Test Loss {:.4f}, test Accuracy {:.4f}".format(test_loss_, test_accu_))
                test_accu.append(test_accu_.item())

        self.print_results(validation_accu, test_accu)

    def print_results(self, validation_accu, test_accu):
        print(test_accu)
        print("std of test accuracy", np.std(test_accu))
        print("average of test accuracy", np.mean(test_accu))
        print(validation_accu)
        print("std of val accuracy", np.std(validation_accu))
        print("average of val accuracy", np.mean(validation_accu))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('-epochs_adj', type=int, default=2000, help='Number of epochs to learn the adjacency.')
    parser.add_argument('-lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('-lr_adj', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('-w_decay', type=float, default=0.0005, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-w_decay_adj', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-hidden', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('-hidden_adj', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('-dropout1', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dropout2', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dropout_adj1', type=float, default=0.25, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dropout_adj2', type=float, default=0.25, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dataset', type=str, default='cora', help='See choices',
                        choices=['cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'ogbn-proteins'])
    parser.add_argument('-nlayers', type=int, default=2, help='#layers')
    parser.add_argument('-nlayers_adj', type=int, default=2, help='#layers')
    parser.add_argument('-patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('-ntrials', type=int, default=1, help='Number of trials')
    parser.add_argument('-k', type=int, default=20, help='k for initializing with knn')
    parser.add_argument('-half_val_as_train', type=int, default=0, help='use first half of validation for training')
    parser.add_argument('-ratio', type=int, default=20, help='ratio of ones to select for each mask')
    parser.add_argument('-epoch_d', type=float, default=5,
                        help='epochs_adj / epoch_d of the epochs will be used for training only with DAE.')
    parser.add_argument('-lambda_', type=float, default=0.1, help='ratio of ones to take')
    parser.add_argument('-nr', type=int, default=5, help='ratio of zeros to ones')
    parser.add_argument('-knn_metric', type=str, default='cosine', help='See choices', choices=['cosine', 'minkowski'])
    parser.add_argument('-model', type=str, default="end2end", help='See choices',
                        choices=['end2end', 'knn_gcn', '2step'])
    parser.add_argument('-i', type=int, default=6)
    parser.add_argument('-non_linearity', type=str, default='elu')
    parser.add_argument('-mlp_act', type=str, default='relu', choices=["relu", "tanh"])
    parser.add_argument('-normalization', type=str, default='sym')
    parser.add_argument('-mlp_h', type=int, default=50)
    parser.add_argument('-mlp_epochs', type=int, default=100)
    parser.add_argument('-gen_mode', type=int, default=0)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-noise', type=str, default="mask", choices=['mask', 'normal'])
    parser.add_argument('-loss', type=str, default="mse", choices=['mse', 'bce'])
    args = parser.parse_args()

    experiment = Experiment()

    if args.model == "end2end":
        experiment.train_end_to_end(args)
    elif args.model == "2step":
        experiment.train_two_steps(args)
    elif args.model == "knn_gcn":
        experiment.train_knn_gcn(args)
