import __init__
import argparse
from config import Config
from utils import *
from graphs.model.netmf import NetMF
import os
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

from models import SBM, DCSBM
from get_func import p_random, p_random_simple, zipf, get_weight

DATA_ROOT = 'ckpt/netmf'
if not os.path.isdir(DATA_ROOT):
    os.mkdir(DATA_ROOT)


def get_parse():
    parser = argparse.ArgumentParser(description='(NetMF)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--negative", type=int, default=1)
    parser.add_argument('--is-large', action='store_true',default=False)

    parser.add_argument("-d", "--dataset", help="dataset", type=str, default='acm')
    parser.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data', type=str, default='cora',
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

    #sadj, fadj = load_graph(args.labelrate, config, to_tensor=False)
    data,degree = load_graph2(args)
    #sadj = data.adj
    #idx_train = data.train
    #idx_test = data.test
    #labels = data.y.numpy()
    #features, labels, idx_train, idx_test = load_data(config, to_tensor=False)
    G = DCSBM(sizes=[1354, 1354], p=p_random_simple(2), theta=get_weight(degree, 50), sparse=True)
    sadj, labels, idx_train, idx_test = SBM_to_data(G)

    print('load model')
    model = NetMF(config.nhid2, args.window_size, args.rank, args.negative, args.is_large)
    embs = model.train(sadj)
    #torch.save(embs, 'ckpt/netmf/{}-z.pt'.format(args.dataset))
    test(embs, labels, idx_train, idx_test)

def main2():
    args = get_parse()
    print(args)

    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    cora_data = [78.63405629903092, 80.90717299578058, 82.49630723781388, 82.65682656826569, 83.63076923076923, 86.10086100861008, 86.53136531365314]
    citeseer_data = [58.414725770097675, 60.62687848862173, 61.84276414621933, 62.980769230769226, 62.05860255447032,  63.813813813813816, 65.26526526526526,]
    data_name = ['cora','citeseer']
    dcsbm_cora = []
    dcsbm_citeseer = []
    theta = [48,70]
    size = [1354,1663,1354,1664]

    # for i in range(0,2):
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
    #         model = NetMF(config.nhid2, args.window_size, args.rank, args.negative, args.is_large)
    #         embs = model.train(sadj)
    #         # torch.save(embs, 'ckpt/netmf/{}-z.pt'.format(args.dataset))
    #         result = test(embs, labels, idx_train, idx_test)
    #         pro += 0.1
    #
    #         if i==0:
    #             cora_data.append(result)
    #         elif i == 1:
    #             citeseer_data.append(result)

    for i in range(0,2):
        args.data = data_name[i]
        pro = 0.2
        data, degree = load_graph2(args, pro)
        G = DCSBM(sizes=[size[i], size[i + 2]], p=p_random_simple(2), theta=get_weight(degree, theta[i]), sparse=True)
        for j in range(0,7):
            dcsbm_data = []
            for z in range(0,8):
                print(j,z)
                sadj, labels, idx_train, idx_test = SBM_to_data(G,pro)
                print('load model')
                model = NetMF(config.nhid2, args.window_size, args.rank, args.negative, args.is_large)
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



def draw1():
    x = [1,2,3,4,5,6,7]
    xticklabes = ['20%','30%','40%','50%','60%','70%','80%']
    y1 = [10,20,30,40,50,60,70]
    y2 = [5, 20, 3, 40, 56, 60, 70]
    plt.plot(x,y1,c='b')
    plt.plot(x, y2, c='g')
    dt1 = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [5, 6, 7],
        'c': [9, 10, 11],
        'd': [13, 14, 15],
    })
    dt2 = pd.DataFrame({
        'a': [3, 4, 5],
        'b': [6, 7, 8],
        'c': [11, 12, 13],
        'd': [15, 17, 18]
    })
    plt.boxplot(dt1,widths=0.1,patch_artist=True,boxprops={'color':'b','facecolor':'b'},medianprops = {'linestyle':'--','color':'r'})
    plt.boxplot(dt2,widths=0.1,patch_artist=True,boxprops={'color':'g','facecolor':'g'},medianprops = {'linestyle':'--','color':'r'})
    plt.xticks(x, xticklabes)
    plt.show()


if __name__ == '__main__':

    cora_data,citeseer_data,dcsbm_cora,dcsbm_citeseer = main2()
    #cora_data = [79.18781725888326, 81.27637130801688, 84.0, 83.75184638109306, 84.50184501845018, 86.71586715867159, 84.31734317343174]
    #citeseer_data = [57.81367392937641, 60.58394160583942, 62.64396594892339, 63.52163461538461, 64.16228399699474, 64.26426426426426, 62.46246246246246]
    #dcsbm_cora = [[79.32625749884633, 78.63405629903092, 76.69589293954776], [82.01476793248945, 82.27848101265823, 82.96413502109705], [83.26153846153846, 84.73846153846154, 83.26153846153846], [83.45642540620383, 82.12703101920236, 77.03101920236337], [83.30258302583026, 85.88560885608855, 82.93357933579337], [85.48585485854858, 84.00984009840099, 80.93480934809348], [80.99630996309963, 88.00738007380073, 78.78228782287823]]
    #dcsbm_citeseer = [[58.22689706987227, 58.5274229902329, 58.865514650638616], [62.08673250322027, 58.60884499785316, 62.55903821382568], [62.24336504757135, 60.741111667501244, 60.54081121682524], [61.89903846153846, 61.598557692307686, 65.44471153846155], [64.16228399699474, 63.26070623591284, 63.11044327573253], [64.16416416416416, 64.66466466466466, 63.76376376376376], [60.36036036036037, 64.56456456456456, 60.210210210210214]]
    draw(cora_data,citeseer_data,dcsbm_cora,dcsbm_citeseer)
