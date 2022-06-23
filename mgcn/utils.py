import scipy.sparse as sp
import torch
import numpy as np
import os.path as osp
import torch_geometric.datasets as geo_data
import random
import networkx as nx
import heapq
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2) ** 2)
    return cost


def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim) - (1 / dim) * torch.ones(dim, dim)
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx,rowsum-1

def load_data(config, to_tensor=True):
    f = np.loadtxt(config.feature_path, dtype=float)
    label = np.loadtxt(config.label_path, dtype=int)
    test = np.loadtxt(config.test_path, dtype=int)
    train = np.loadtxt(config.train_path, dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)

    idx_test = test.tolist()
    idx_train = train.tolist()
    if to_tensor:
        features = torch.FloatTensor(np.array(features.todense()))
        label = torch.LongTensor(np.array(label))
        idx_train = torch.LongTensor(idx_train)
        idx_test = torch.LongTensor(idx_test)

    return features, label, idx_train, idx_test

def divide_data(n,p):
    train = []
    test = []
    train_num = int(n*p)
    while len(train)<train_num:
        x = random.randint(0,n-1)
        if x not in train:
            train.append(x)
    for i in range(0,n):
        if i not in train:
            test.append(i)
    return train,test


def load_graph2(args,p):
    DATA_ROOT = '../datasets'
    path = osp.join(DATA_ROOT, args.data)
    data = geo_data.Planetoid(path, args.data)[0]
    n = len(data.x)
    data.train,data.test = divide_data(n,p)
    adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index), shape=(n, n))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    data.adj,degree = normalize_adj_row(adj)  # symmetric normalization works bad, but why? Test more.
    return data,degree

def choose(list1,a):
    result = []
    loss = []
    mark = []
    list1.sort()
    for i in range(0,len(list1)):
        loss.append(abs(list1[i]-a))
    smallest = heapq.nsmallest(3,loss)
    for i in range(0,len(smallest)):
        for j in range(0,len(loss)):
            if smallest[i]==loss[j]:
                mark.append(j)
    for i in range(0,len(mark)):
        result.append(list1[mark[i]])
    return result




def SBM_to_data(G,p):
    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    adj,degree = normalize_adj_row(adj)  # symmetric normalization works bad, but why? Test more.
    n = len(degree)

    y = []
    for i in range(len(G.nodes)):
        y.append(G.nodes[i]['block'])
    y = np.array(y)

    train, test = divide_data(n,p)

    return adj,y,train,test


def load_graph(dataset, config, to_tensor=True):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'

    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))
    if to_tensor:
        nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
        nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nsadj, nfadj


def draw(cora_data,citeseer_data,dcsbm_cora,dcsbm_citeseer):
    x = [1, 2, 3, 4, 5, 6, 7]
    xticklabes = ['20%', '30%', '40%', '50%', '60%', '70%', '80%']
    plt.plot(x, cora_data, c='b')
    plt.plot(x, citeseer_data, c='g')
    dt1 = pd.DataFrame({
        'a': dcsbm_cora[0],
        'b': dcsbm_cora[1],
        'c': dcsbm_cora[2],
        'd': dcsbm_cora[3],
        'e': dcsbm_cora[4],
        'f': dcsbm_cora[5],
        'g': dcsbm_cora[6]
    })
    dt2 = pd.DataFrame({
        'a': dcsbm_citeseer[0],
        'b': dcsbm_citeseer[1],
        'c': dcsbm_citeseer[2],
        'd': dcsbm_citeseer[3],
        'e': dcsbm_citeseer[4],
        'f': dcsbm_citeseer[5],
        'g': dcsbm_citeseer[6]
    })
    plt.boxplot(dt1, widths=0.1, patch_artist=True, boxprops={'color': 'b', 'facecolor': 'b'},
                medianprops={'linestyle': '--', 'color': 'r'})
    plt.boxplot(dt2, widths=0.1, patch_artist=True, boxprops={'color': 'g', 'facecolor': 'g'},
                medianprops={'linestyle': '--', 'color': 'r'})
    plt.xticks(x, xticklabes)
    plt.show()