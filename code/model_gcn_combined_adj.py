import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sqrtm import sqrtm
import dataloader

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


import pdb


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm, self).__init__()

        self.batchnorm_layer = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        # pdb.set_trace()
        x = x.permute(0, 2, 1)
        x = self.batchnorm_layer(x)
        x = x.permute(0, 2, 1)
        return x

class BatchNormAdj(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNormAdj, self).__init__()
        self.batchnorm_layer = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        # pdb.set_trace()
        batch_size = x.size(0)
        num_region = x.size(1)
        # pdb.set_trace()
        x = x.contiguous().view(batch_size, -1)
        x = self.batchnorm_layer(x)
        x = x.contiguous().view(batch_size, num_region, -1)
        return x


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # pdb.set_trace()
        support = torch.matmul(input, self.weight)
        # print adj.shape
        # print support.shape
        output = torch.einsum('bij,bjd->bid', [adj, support])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class CombinedADJ(Module):

    def __init__(self, M_row, M_col, var, bias=False):
        super(CombinedADJ, self).__init__()
        self.M_row = M_row
        self.M_col = M_col
        self.var = var
        # self.weight1 = Parameter(torch.FloatTensor(M_row, M_col))
        self.weight1 = Parameter(torch.eye(M_row))
        self.weight2 = Parameter(torch.FloatTensor(2,))
        if bias:
            self.bias = Parameter(torch.FloatTensor(M_col))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight1.size(1))
    #     self.weight1.data.uniform_(-stdv, stdv)
    #     stdv = 1. / math.sqrt(self.weight2.size(0))
    #     self.weight2.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    # def reset_parameters(self):
    #     nn.init.xavier_normal_(self.weight1)
    #     nn.init.constant_(self.weight2, 1.0)
    #     if self.bias is not None:
    #         nn.init.constant_(self.bias, 0.0)

    # def reset_parameters(self):
    #     nn.init.xavier_normal_(self.weight1)
    #     stdv = 1. / math.sqrt(self.weight2.size(0))
    #     self.weight2.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters(self):
        nn.init.eye_(self.weight1)
        nn.init.constant_(self.weight2[0], 0.0)
        nn.init.constant_(self.weight2[1], 0.0)
        # print self.weight2
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def batch_eye(self, size):
        batch_size = size[0]
        n = size[1]
        I = torch.eye(n).unsqueeze(0)
        I = I.repeat(batch_size, 1, 1)
        return I

    def laplace(self, A):
        # pdb.set_trace()
        eps = 1e-16
        # A = torch.div(A, A.max())  #cause nan
        D = self.batch_eye(A.size()).to(A.device)
        A = self.batch_eye(A.size()).to(A.device) - ((self.batch_eye(A.size()).to(A.device)-1)*A)
        # pdb.set_trace()
        D = D * torch.sum(A, dim=2, keepdim=True).repeat(1,1,A.size(2))
        # pdb.set_trace()

        L1 = D - A
        L2 = torch.bmm(torch.bmm(torch.bmm(torch.inverse(D), sqrtm(D)), L1), torch.bmm(torch.inverse(D), sqrtm(D)))
        # L3 = torch.bmm(torch.inverse(D), A)
        # pdb.set_trace()

        return L2

    def adjacency_graph(self, pairwise_distance):
        # pdb.set_trace()
        # print self.weight1

        wighted_distance = torch.einsum('ij,bmnj->bmni', [self.weight1, pairwise_distance])
        # euclidean_distance_square1 = torch.einsum('bmni,bmin->bmnn', [wighted_distance, wighted_distance.permute(0,1,3,2)])
        euclidean_distance_square = torch.norm(wighted_distance, p=2, dim=3).pow(2)
        # print euclidean_distance_square1
        # print euclidean_distance_square
        # exit()
        As = torch.exp(torch.div(-1*euclidean_distance_square, 2*self.var * self.var))
        # pdb.set_trace()
        return As

    # # this part needs to be correct by following the draft
    # # pairwise distnace cannot be computed outside since the weight1 is updated internally during training
    # # correct this function using pytorch language
    # def adjacency_graph_correct(self, feature_matrix):
    #     project_feature_matrix = torch.einsum("ij,bmnj->bmni", [self.weight1, feature_matrix])
    #     euclidean_distance_square = dataloader.pairwise_distance(project_feature_matrix)
    #     As = torch.exp(torch.div(-1 * euclidean_distance_square, 2 * self.var * self.var))
    #     # pdb.set_trace()
    #     return As

    def parameter_creator(self):
        # print "beta:", self.weight2
        ratio = F.softmax(self.weight2, dim=0)
        # pdb.set_trace()
        return ratio

    def forward(self, input, adj, pair):
        ratio = self.parameter_creator()
        # print ratio
        # pdb.set_trace()

        # As = self.adjacency_graph(pair)
        As = self.adjacency_graph(pair)

        # adj = torch.div(adj, adj.max())
        I = torch.eye(adj.size(1)).to(input.device)
        I = I.reshape((1, adj.size(1), adj.size(1)))
        Is = I.repeat(adj.size(0), 1, 1)
        adj_matrix = torch.add(torch.add(ratio[0]*As, ratio[1]*adj), Is)

        # out = self.laplace(adj_matrix)*0.4 + adj_matrix
        # out = self.laplace(adj_matrix)
        out = adj_matrix

        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, in_feature, out1_feature, out2_feature, nclass, var, M_row, M_col,  dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_feature, out1_feature)
        self.gc2 = GraphConvolution(out1_feature, out2_feature)
        self.combinedADJ = CombinedADJ(M_row, M_col, var)
        self.classifer = nn.Linear(148*out2_feature, nclass)
        self.dropout = dropout

    def forward(self, feature_matrix, adj_matrix, pairwise_distance, fmri_Pearson, isTest = False):
        batchSize = feature_matrix.shape[0]
        regionSize = adj_matrix.shape[1]
        adj_matrix = self.combinedADJ(feature_matrix, adj_matrix, pairwise_distance)
        # pdb.set_trace()
        # x = feature_matrix.permute(0,2,1,3).squeeze()
        # x = feature_matrix.permute(0, 2, 1, 3).contiguous().view(batchSize,regionSize, -1) # if use orig fmri dignal
        x = fmri_Pearson
        x = F.relu(self.gc1(x, adj_matrix), inplace=True)
        x = F.relu(self.gc2(x, adj_matrix), inplace=True)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.contiguous().view(batchSize, -1)
        outputs = self.classifer(x)
        if isTest is True:
            return outputs.squeeze(), adj_matrix
        else:
            return outputs.squeeze()


class GCN_BN(nn.Module):
    def __init__(self, in_feature, out1_feature, out2_feature, nclass, var, M_row, M_col,  dropout):
        super(GCN_BN, self).__init__()

        self.gc1 = GraphConvolution(in_feature, out1_feature)
        self.batchnorm1 = BatchNorm(out1_feature)
        self.gc2 = GraphConvolution(out1_feature, out2_feature)
        self.batchnorm2 = BatchNorm(out2_feature)
        self.combinedADJ = CombinedADJ(M_row, M_col, var)
        self.classifer = nn.Linear(148*out2_feature, nclass)
        self.dropout = dropout

    def forward(self, feature_matrix, adj_matrix, pairwise_distance, fmri_Pearson, isTest = False):
        batchSize = feature_matrix.shape[0]
        adj_matrix = self.combinedADJ(feature_matrix, adj_matrix, pairwise_distance)
        # pdb.set_trace()
        # x = feature_matrix.permute(0,2,1,3).squeeze()
        # x = feature_matrix.permute(0, 2, 1, 3).contiguous().view(batchSize,regionSize, -1) # if use orig fmri dignal
        x = fmri_Pearson
        x = self.batchnorm1(self.gc1(x, adj_matrix))
        x = F.relu(x, inplace=True)
        x = self.batchnorm2(self.gc2(x, adj_matrix))
        x = F.relu(x, inplace=True)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.contiguous().view(batchSize, -1)
        outputs = self.classifer(x)
        if isTest is True:
            return outputs.squeeze(), adj_matrix
        else:
            return outputs.squeeze()


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    feats = torch.ones(16, 45, 148, 1).to(device)
    adjs = torch.ones(16, 148, 148).to(device)
    pairwise_distance = torch.ones(16, 148, 148, 45).to(device)
    fmri_Pearson = torch.ones(16, 148, 148).to(device)
    labels = torch.randint(0,2,(16,), dtype=torch.long).to(device)

    var = 2
    M_row = 45
    M_col = 45
    time = 45
    region = 148

    model = GCN_BN(region, 45, 45*2, 2, var, M_row, M_col, 0.6).to(device)
    # model = RNN_GCN(1, 32, 32*2, 32, 2, var, 0.6)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5)

    for i in range(100):
        scores, final_adj = model(feats, adjs, pairwise_distance, fmri_Pearson, isTest = True)
        # pdb.set_trace()
        # print final_adj.shape
        loss = criterion(scores, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())