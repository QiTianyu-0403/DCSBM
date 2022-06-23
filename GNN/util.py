import numpy as np
import os.path as osp
import torch
import torch_geometric.datasets as geo_data
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import random
from sklearn.metrics import roc_auc_score


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_data(args):
    DATA_ROOT = '../../datasets'
    path = osp.join(DATA_ROOT, args.data)
    data = geo_data.Planetoid(path, args.data)[0]
    print(dir(data))

    adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index), shape=(data.num_nodes, data.num_nodes))
    adj = adj + adj.T - adj.multiply(adj.T)  # + sp.eye(adj.shape[0])
    adj_dense = np.array(adj.todense())
    adj_self_loop = adj + sp.eye(adj.shape[0])
    adj_sym = normalize_adj(adj_self_loop)
    adj_rw, _ = normalize_adj_row(adj_self_loop)
    laplacian = np.diag(adj_dense.sum(1)) - adj_dense
    return data, adj, adj_dense, adj_sym, adj_rw, laplacian


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # add self-loop and normalization also affects performance a lot
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx, r_mat_inv


def to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def col_x_normalize(X):
    # norm = 1e-6 + X.sum(dim=0, keepdim=True)
    if type(X) is np.ndarray:
        norm = np.linalg.norm(X, ord=2, axis=0)
    if type(X) is torch.Tensor:
        norm = 1e-6 + torch.norm(X.float(), p=2, dim=0)
    return X / norm


def draw_graph(data, adj_graph):
    graph = nx.from_numpy_matrix(adj_graph)
    # print(graph.nodes(data=True))
    label = np.array((data.y))
    print(label)
    for i in range(len(graph.nodes)):
        graph.nodes[i].update({"label": label[i]})
    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
    ]
    color_list = [colors[i] for i in label]
    nx.draw(graph, node_color=color_list, node_size=20, width=0.5)
    plt.axis('off')
    plt.show()


def draw_heatmap(mat, name):
    plt.imshow(mat, vmin=-0.05, vmax=0.05, aspect='auto', origin='lower', cmap=cm.RdBu_r)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(name)
    plt.colorbar()
    plt.savefig('fig/{}.pdf'.format(name))
    plt.close()


def train(args, net, optimizer, criterion, data):
    net.train()
    optimizer.zero_grad()
    output = net(data)
    if args.data == 'proteins':
        loss = criterion(output[data.train_mask], data.y[data.train_mask].to(torch.float))
        acc = roc_auc(output[data.train_mask], data.y[data.train_mask].to(torch.float))
    else:
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=2)
    optimizer.step()
    return loss, acc


def roc_auc(output, labels):
    # TODO: convert roc_auc to support tensor
    return roc_auc_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy())


def val(args, net, criterion, data):
    net.eval()
    output = net(data)
    loss_val = criterion(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    return loss_val, acc_val


@torch.no_grad()
def test(args, net, criterion, data):
    net.eval()
    output = net(data)
    loss_test = criterion(output[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    return loss_test, acc_test, output


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
