import torch
import torch.nn as nn
from torch_sparse import spmm
from torch_geometric.nn.inits import glorot, zeros


class GraphConv(nn.Module):
    """
        Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, input, adj):
        return adj @ input @ self.weight

    def __repr__(self):
        return self.__class__.__name__ + "({}->{})".format(
            self.in_features, self.out_features)


class GraphAttConv(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout):
        super(GraphAttConv, self).__init__()
        assert out_features % heads == 0
        out_perhead = out_features // heads

        self.graph_atts = nn.ModuleList([GraphAttConvOneHead(
            in_features, out_perhead, dropout=dropout) for _ in range(heads)])

        self.in_features = in_features
        self.out_perhead = out_perhead
        self.heads = heads

    def forward(self, input, adj):
        output = torch.cat([att(input, adj) for att in self.graph_atts], dim=1)
        # notice that original GAT use elu as activation func.
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->[{}x{}])".format(
            self.in_features, self.heads, self.out_perhead)


class GraphAttConvOneHead(nn.Module):
    """
    Sparse version GAT layer, single head
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttConvOneHead, self).__init__()
        self.weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        # init
        nn.init.xavier_normal_(self.weight.data, gain=nn.init.calculate_gain('relu'))  # look at here
        nn.init.xavier_normal_(self.a.data, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, input, adj):
        edge = adj._indices()
        h = torch.mm(input, self.weight)
        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # edge_h: 2*D x E
        # do softmax for each row, this need index of each row, and for each row do softmax over it
        alpha = self.leakyrelu(self.a.mm(edge_h).squeeze())  # E
        n = len(input)
        alpha = softmax(alpha, edge[0], n)
        output = spmm(edge, self.dropout(alpha), n, n, h)  # h_prime: N x out
        # output = spmm(edge, self.dropout(alpha), n, n, self.dropout(h)) # h_prime: N x out
        return output


class GraphNorm(nn.Module):
    def __init__(self, nhid, mode='None', scale=1):
        """
            mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
              'BN' : Batch Normalization
              'LN' : Layer Normalization

        """
        assert mode in ['None', 'BN', 'LN', 'PN', 'PN-SI', 'PN-SCS', 'mean-sub']
        super(GraphNorm, self).__init__()
        self.nhid = nhid
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == 'None':
            return x

        if self.mode == 'BN':
            return nn.BatchNorm1d(self.nhid)(x)

        if self.mode == 'LN':
            return nn.BatchNorm1d(self.nhid)(x)

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        if self.mode == 'mean-sub':
            x = x - x.mean(1, keepdim=True)

        return x


"""
    helpers
"""
from torch_scatter import scatter_max, scatter_add


def softmax(src, index, num_nodes=None):
    """
        sparse softmax
    """
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out
