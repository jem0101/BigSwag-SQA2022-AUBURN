from itertools import product

import argparse
from kernel.datasets import get_dataset
from kernel.train_eval import cross_validation_with_val_set

from kernel.gcn import GCN
from kernel.graph_sage import GraphSAGE, GraphSAGEWithoutJK
from kernel.gin import GIN0, GIN
from kernel.graclus import Graclus
from kernel.top_k import TopK
from kernel.diff_pool import DiffPool
from kernel.global_attention import GlobalAttentionNet
from kernel.set2set import Set2SetNet
from kernel.sort_pool import SortPool

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
args = parser.parse_args()

layers = [1, 2, 3, 4, 5]
hiddens = [16, 32, 64, 128]
datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']  # , 'COLLAB']
nets = [
    GCN,
    GraphSAGE,
    GIN0,
    GIN,
    Graclus,
    TopK,
    DiffPool,
    GraphSAGEWithoutJK,
    GlobalAttentionNet,
    Set2SetNet,
    SortPool,
]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))


results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))
    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        model = Net(dataset, num_layers, hidden)
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=None)
        if loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = '{:.3f} ?? {:.3f}'.format(best_result[1], best_result[2])
    print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
print('-----\n{}'.format('\n'.join(results)))
