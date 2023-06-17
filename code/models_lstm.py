import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

import pdb

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm, self).__init__()

        self.batchnorm_layer = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x = x.permute(0,1,3,2)
        batch_size = x.size(0)
        time_size = x.size(1)
        feat_size = x.size(2)

        x = x.contiguous().view(batch_size*time_size, feat_size, -1)
        x = self.batchnorm_layer(x)
        x = x.contiguous().view(batch_size, time_size, feat_size, -1)
        x = x.permute(0,1,3,2)
        return x

class GCN_RNN(nn.Module):
    def __init__(self, in_feature, out1_feature, out2_feature, nhid, nclass, dropout):
        super(GCN_RNN, self).__init__()

        self.gc1 = GraphConvolution(in_feature, out1_feature)
        self.gc2 = GraphConvolution(out1_feature, out2_feature)
        self.rnn = nn.LSTM(input_size=148*out2_feature, hidden_size=nhid, batch_first=True)
        self.classifer = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, feature_matrix, adj_matrix):
        batchSize = feature_matrix.shape[0]
        timeSize = feature_matrix.shape[1]
        gcn_outputs = []
        # pdb.set_trace()
        x = feature_matrix
        x = F.relu(self.gc1(x, adj_matrix), inplace=True)
        x = F.relu(self.gc2(x, adj_matrix), inplace=True)
        x = F.dropout(x, self.dropout, training=self.training)
        # pdb.set_trace()
        gcn_outputs = x.contiguous().view(batchSize, timeSize, -1)
        hiddens, (hn, cn) = self.rnn(gcn_outputs)
        hn = F.dropout(F.relu(hn), self.dropout, training=self.training)
        outputs = self.classifer(hn)

        return outputs.squeeze()

class GCN_RNN_BN(nn.Module):
    def __init__(self, in_feature, out1_feature, out2_feature, nhid, nclass, dropout):
        super(GCN_RNN_BN, self).__init__()

        self.gc1 = GraphConvolution(in_feature, out1_feature)
        self.batchnorm1 = BatchNorm(out1_feature)
        self.gc2 = GraphConvolution(out1_feature, out2_feature)
        self.batchnorm2 = BatchNorm(out2_feature)
        self.rnn = nn.LSTM(input_size=148*out2_feature, hidden_size=nhid, batch_first=True)
        self.classifer = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, feature_matrix, adj_matrix):
        batchSize = feature_matrix.shape[0]
        timeSize = feature_matrix.shape[1]
        gcn_outputs = []
        # pdb.set_trace()
        x = feature_matrix
        x = self.batchnorm1(self.gc1(x, adj_matrix))
        x = F.relu(x, inplace=True)
        x = self.batchnorm2(self.gc2(x, adj_matrix))
        x = F.relu(x, inplace=True)
        x = F.dropout(x, self.dropout, training=self.training)
        # pdb.set_trace()
        gcn_outputs = x.contiguous().view(batchSize, timeSize, -1)
        hiddens, (hn, cn) = self.rnn(gcn_outputs)
        hn = F.dropout(F.relu(hn), self.dropout, training=self.training)
        outputs = self.classifer(hn)

        return outputs.squeeze()

if __name__ == '__main__':
    feats = torch.randn(16, 45, 148, 1).to('cuda:0')
    adjs = torch.randn(16, 148, 148).to('cuda:0')
    labels = torch.randint(0,2,(16,), dtype=torch.long).to('cuda:0')

    model = GCN_RNN(1, 32, 32*2, 512, 2, 0.6).to('cuda:0')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)

    for i in range(1000):
        scores = model(feats, adjs)
        # pdb.set_trace()
        loss = criterion(scores, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        print loss.item()