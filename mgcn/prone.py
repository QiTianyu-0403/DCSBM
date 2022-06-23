import __init__
import argparse
from config import Config
from utils import *
from graphs.model.prone import ProNE
import os
from sklearn.linear_model import LogisticRegression

from models import SBM, DCSBM
from get_func import p_random, p_random_simple, zipf, get_weight

DATA_ROOT = 'ckpt/prone'
if not os.path.isdir(DATA_ROOT):
    os.mkdir(DATA_ROOT)


def get_parse():
    parser = argparse.ArgumentParser(description='(NetMF)')
    parser.add_argument('--data', type=str, default='blogcatalog', help="ppi, blogcatalog, wikipedia")
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument("--step", type=int, default=5,
                        help="Number of items in the chebyshev expansion")
    parser.add_argument("--mu", type=float, default=0.2)
    parser.add_argument("--theta", type=float, default=0.5)

    parser.add_argument("-d", "--dataset", help="dataset", type=str, default='acm')
    parser.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, default=60)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-data', type=str, default='cora',
                        help='{cora, pubmed, citeseer, ogbn-arxiv ,ogbn-proteins}.')
    args = parser.parse_args()
    return args


def test(X, y, train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = LogisticRegression(solver="liblinear", multi_class="ovr")
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f'Acc: {acc * 100:.2f}')
    return acc*100


def main():
    args = get_parse()
    print(args)

    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    sadj, fadj = load_graph(args.labelrate, config, to_tensor=False)
    features, labels, idx_train, idx_test = load_data(config, to_tensor=False)
    print(type(sadj))
    print('load model')
    model = ProNE(args.hidden_size, args.step, args.mu, args.theta)
    embs = model.train(sadj)
    torch.save(embs, 'ckpt/prone/{}-z.pt'.format(args.dataset))
    test(embs, labels, idx_train, idx_test)


def main2():
    args = get_parse()
    print(args)

    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    cora_data = [70.18874019381634, 72.36286919831224, 75.13846153846154, 77.25258493353027, 78.78228782287823, 79.05781057810579, 81.12177121771218]
    citeseer_data = [51.77986476333584, 53.52984113353371, 54.58187280921381, 56.030769230769226, 55.09992486851991, 56.15615615615616, 58.5045045045045]
    pubmed_data = [78.17294281729428, 78.40168091580931, 79.5875243005663, 79.39953342123948, 79.54697603651579, 79.41176470588235, 80.50202839756592]
    data_name = ['cora','citeseer','pubmed']
    dcsbm_cora = []
    dcsbm_citeseer = []
    dcsbm_pubmed = []
    theta = [48,70,120]
    size = [1354,1663,9858,1354,1664,9859]

    # for i in range(0,3):
    #     args.data = data_name[i]
    #     pro = 0.2
    #     for j in range(0,7):
    #         data, degree = load_graph2(args,pro)
    #         sadj = data.adj
    #         idx_train = data.train
    #         idx_test = data.test
    #         labels = data.y.numpy()
    #
    #         print('load model')
    #         model = ProNE(args.hidden_size, args.step, args.mu, args.theta)
    #         embs = model.train(sadj)
    #         # torch.save(embs, 'ckpt/netmf/{}-z.pt'.format(args.dataset))
    #         result = test(embs, labels, idx_train, idx_test)
    #         pro += 0.1
    #
    #         if i==0:
    #             cora_data.append(result)
    #         elif i == 1:
    #             citeseer_data.append(result)
    #         elif i == 2:
    #             pubmed_data.append(result)
    # print(cora_data)
    # print(citeseer_data)
    # print(pubmed_data)

    for i in range(0,2):
        args.data = data_name[i]
        pro = 0.2
        data, degree = load_graph2(args, pro)
        G = DCSBM(sizes=[size[i], size[i + 3]], p=p_random_simple(2), theta=get_weight(degree, theta[i]), sparse=True)
        for j in range(0,7):
            dcsbm_data = []
            for z in range(0,8):
                print(j,z)
                sadj, labels, idx_train, idx_test = SBM_to_data(G,pro)
                print('load model')
                model = ProNE(args.hidden_size, args.step, args.mu, args.theta)
                embs = model.train(sadj)
                # torch.save(embs, 'ckpt/netmf/{}-z.pt'.format(args.dataset))
                result = test(embs, labels, idx_train, idx_test)
                dcsbm_data.append(result)
            if i==0:
                dcsbm_cora.append(dcsbm_data)
            elif i==1:
                dcsbm_citeseer.append(dcsbm_data)
            pro += 0.1
    for i in range(0,7):
        dcsbm_cora[i] = choose(dcsbm_cora[i],cora_data[i])
        dcsbm_citeseer[i] = choose(dcsbm_citeseer[i],citeseer_data[i])
    print(cora_data)
    print(citeseer_data)
    print(dcsbm_cora)
    print(dcsbm_citeseer)
    return cora_data,citeseer_data,dcsbm_cora,dcsbm_citeseer


if __name__ == '__main__':
    main2()
