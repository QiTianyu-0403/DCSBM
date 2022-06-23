import torch.nn as nn
import torch.nn.functional as F
from graph.layers import GraphConv
import torch


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()


class GRAPH(nn.Module):
    def __init__(self, args):
        super(GRAPH, self).__init__()
        z = torch.load('ckpt/' + '{}'.format(args.model) + '/{}-z.pt'.format(args.dataset))
        z = torch.tensor(z, dtype=torch.float32)
        self.z = torch.nn.Parameter(z)

    def forward(self):
        return self.z

    def reset_parameters(self):
        pass


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class MGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, dropout, args):
        super(MGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        # self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.feature = nn.Sequential(
            nn.Linear(nfeat, nhid1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(nhid1, nhid2),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        self.tanh = nn.Tanh()

        self.MLP = nn.Linear(nhid2 * 3, nclass)
        self.z = GRAPH(args)

    def forward(self, x, sadj, fadj):
        emb1 = self.z()

        # emb2 = self.feature(x)
        emb2 = self.SGCN1(x, fadj)

        Xcom = self.SGCN2(x, sadj)
        # Xcom = emb1 * emb2

        emb = torch.cat((emb1, emb2, Xcom), dim=1)
        # emb = emb1 + emb2 + Xcom

        output = self.MLP(emb)

        # return output, att, emb1, com1, com2, emb2, emb
        return output, None, None, None, None, None, None

    def reset_parameters(self):
        self.SGCN1.reset_parameters()
        self.SGCN2.reset_parameters()
        self.MLP.reset_parameters()
        self.feature.apply(weight_reset)
        print("initial")
