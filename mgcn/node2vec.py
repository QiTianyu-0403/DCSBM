from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec
import argparse
from config import Config
from utils import *
import os

DATA_ROOT = 'ckpt/node2vec'
if not os.path.isdir(DATA_ROOT):
    os.mkdir(DATA_ROOT)


def get_parse():
    parser = argparse.ArgumentParser(description='(Node2Vec)')
    parser.add_argument('--data', type=str, default='uai')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--walk_length', type=int, default=20)
    parser.add_argument('--context_size', type=int, default=10)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--draw', action='store_true')
    parser.add_argument("-d", "--dataset", help="dataset", type=str, default='uai')
    parser.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, default=20)
    args = parser.parse_args()
    return args


def train(model, edge_index, loader, optimizer, device):
    model.train()
    total_loss = 0
    for subset in loader:
        optimizer.zero_grad()
        loss = model.loss(edge_index, subset.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test(model, num_nodes, labels, train_index, test_index, device, name):
    model.eval()
    model.load_state_dict(torch.load('ckpt/node2vec/{}-params.pth'.format(name)))
    z = model(torch.arange(num_nodes, device=device))
    # if not osp.exists('node2vec.pt'):
    torch.save(z, 'ckpt/node2vec/{}-z.pt'.format(name))
    acc = model.test(z[train_index], labels[train_index],
                     z[test_index], labels[test_index], max_iter=150)
    return acc


@torch.no_grad()
def plot_points(model, data, colors, device):
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(data.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()


def save_embedding(model, name):
    torch.save(model.state_dict(), 'ckpt/node2vec/{}-params.pth'.format(name))


def main():
    args = get_parse()
    print(args)

    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sadj, fadj = load_graph(args.labelrate, config)
    features, labels, idx_train, idx_test = load_data(config)
    num_nodes = features.shape[0]
    sadj = sadj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_test = idx_test.to(device)
    edge_index = sadj._indices()

    loader = DataLoader(torch.arange(num_nodes), batch_size=args.batch_size, shuffle=True)

    model = Node2Vec(num_nodes, embedding_dim=config.nhid2, walk_length=args.walk_length,
                     context_size=args.context_size, walks_per_node=args.walks_per_node).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train(model, edge_index, loader, optimizer, device)
        print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss))
    save_embedding(model, args.dataset)

    acc = test(model, num_nodes, labels, idx_train, idx_test, device, args.dataset)
    print('Accuracy: {:.4f}'.format(acc))


def main2():
    args = get_parse()
    print(args)

    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cora_data = []
    citeseer_data = []
    data_name = ['cora','citeseer']
    dcsbm_cora = []
    dcsbm_citeseer = []
    theta = [48,70]
    size = [1354,1663,1354,1664]


if __name__ == '__main__':
    main()
