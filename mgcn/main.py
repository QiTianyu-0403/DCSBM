import __init__
import os
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from graph.model.mgcn import MGCN
import argparse
from config import Config
import logging

DATA_ROOT = 'logs'
if not os.path.isdir(DATA_ROOT):
    os.mkdir(DATA_ROOT)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--runs", help="runs", type=int, default=5)
    parse.add_argument("-d", "--dataset", help="dataset", type=str, default='acm')
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, default=20)
    parse.add_argument("-m", "--model", help="netmf or node2vec", type=str, default='netmf')

    args = parse.parse_args()
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    logging.basicConfig(
        filename='logs/' + '{}'.format(args.dataset) + '-{}'.format(args.model) + '-{}.log'.format(args.labelrate),
        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        level=logging.DEBUG, filemode='w')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_seed = not config.no_seed
    logging.info("set_seed: {}".format(use_seed))
    if use_seed:
        from graph.util import setup_seed

        setup_seed(config.seed)

    sadj, fadj = load_graph(args.labelrate, config)
    features, labels, idx_train, idx_test = load_data(config)

    model = MGCN(nfeat=config.fdim,
                 nhid1=config.nhid1,
                 nhid2=config.nhid2,
                 nclass=config.class_num,
                 dropout=config.dropout, args=args).to(device)

    features = features.to(device)
    sadj = sadj.to(device)
    fadj = fadj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_test = idx_test.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    ignored_params = list(map(id, model.z.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer = optim.Adam([
        {'params': base_params, 'lr': config.lr},
        {'params': model.z.parameters(), 'lr': 0.001 * config.lr}
    ], weight_decay=config.weight_decay)


    def train(model, epochs):
        model.train()
        optimizer.zero_grad()
        output, att, emb1, com1, com2, emb2, emb = model(features, sadj, fadj)
        loss = F.cross_entropy(output[idx_train], labels[idx_train])
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        acc_test, emb_test = main_test(model)
        logging.info(
            'epoch:{}, loss: {:.4f} , train_acc: {:.2f}, test_acc: {:.2f}'.
                format(epochs, loss.item(), acc.item() * 100, acc_test.item() * 100)
        )
        return loss.item(), acc_test.item(), emb_test


    def main_test(model):
        model.eval()
        output, att, emb1, com1, com2, emb2, emb = model(features, sadj, fadj)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test, emb


    acc = []
    for run in range(args.runs):
        logging.info("Round: {}".format(run))
        model.reset_parameters()
        acc_max = 0
        epoch_max = 0
        for epoch in range(config.epochs):
            loss, acc_test, emb = train(model, epoch)
            if acc_test >= acc_max:
                acc_max = acc_test
                epoch_max = epoch
        logging.info('epoch:{}, acc_max: {:.2f}'.format(epoch_max, acc_max * 100))
        acc.append(acc_max)
    logging.info('{} Average acc:{:.2f}'.format(args.runs, np.mean(acc) * 100))
