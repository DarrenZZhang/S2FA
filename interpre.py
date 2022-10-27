from utils import interpretation
import argparse
import sys
from model.schizophrenia import schiNet, schiClassifier
from model.ABIDE import ABIDENet, ABIDEClassifier
from utils.federated_utils import *
from train.train import train, test
from datasets.Schizophrenia import get_schizophrenia_dloader
from datasets.ABIDE import get_ABIDE_dloader
import os
from os import path
import shutil
import yaml
import numpy as np

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Default settings
parser = argparse.ArgumentParser(description='K3DA Official Implement')
# Dataset Parameters
parser.add_argument("--config", default="schizophrenia.yaml")
parser.add_argument('-bp', '--base-path', default="../data/")
parser.add_argument('--target-domain', default="Xiangya_143_L1", type=str, help="The target domain we want to perform domain adaptation")   #domains = ['Leuven', 'NYU', 'UCLA', 'UM', 'USM']
parser.add_argument('--source-domains', type=str, nargs="+", help="The source domains we want to use")      # domains = ['data_huang45', 'COBRE_120_L1', 'Nottingham_68_L1', 'Taiwan_131_L1', 'Xiangya_143_L1']
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--beta', default=0.01, type=float)
parser.add_argument('--delta', default=0.9, type=float)
parser.add_argument('--tsne', default=False, type=bool)

# Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=str,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# Optimizer Parameters
parser.add_argument('--optimizer', default="SGD", type=str, metavar="Optimizer Name")
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='Momentum in SGD')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
parser.add_argument('-bm', '--bn-momentum', type=float, default=0.1, help="the batchnorm momentum parameter")
parser.add_argument("--gpu", default="0", type=str, metavar='GPU plans to use', help='The GPU id plans to use')
args = parser.parse_args()
# import config files
with open(r"./config/{}".format(args.config)) as file:
    configs = yaml.full_load(file)

def ffff(grad):
    g = []
    for i in range(len(grad)):
        grad[i] = grad[i][0][0].sum(axis=0)
        t_min, t_max = grad[i].min(0), grad[i].max(0)
        if t_max != 0:
            g.append(grad[i] / t_max)
    g = np.array(g)
    g = g.sum(axis=0) / g.shape[0]
    index = np.argsort(g)[-10:]
    # for i in index[::-1]:
    #     print(g[i])

    # print(np.sort(index) + 1)
    print(index[::-1]+1)


if __name__ == '__main__':
    if args.config == "ABIDE.yaml":
        domains = ['Leuven', 'NYU', 'UCLA', 'UM', 'USM']

        target_train_dloader = get_ABIDE_dloader(args.base_path,
                                                 domains.index(args.target_domain),
                                                 configs["TrainingConfig"]["batch_size"],
                                                 args.workers)
        f = ABIDENet()
        c = ABIDEClassifier()
        load_model = torch.load(
            '../data/ABIDE/parameter/train_time_1/{}.pth.tar'.format(args.target_domain))

    if args.config == "schizophrenia.yaml":
        domains = ['data_huang45', 'COBRE_120_L1', 'Nottingham_68_L1', 'Taiwan_131_L1', 'Xiangya_143_L1']
        target_train_dloader = get_schizophrenia_dloader(args.base_path,
                                                         args.target_domain,
                                                         configs["TrainingConfig"]["batch_size"],
                                                         args.workers)
        f = schiNet()
        c = schiClassifier()
        load_model = torch.load(
            '../data/schizophrenia/parameter/train_time_1/{}.pth.tar'.format(args.target_domain))



    f.load_state_dict(load_model['backbone'])
    c.load_state_dict(load_model['classifier'])

    model = interpretation.model(f, c)


    model.cuda()
    grad_normal = list()
    grad_pa = list()
    gdbp = interpretation.GuidedBackPropogation(model)

    model.eval()
    for data, label in target_train_dloader:
        data = data.cuda()
        data = data.requires_grad_()
        out_b = gdbp(data)
        out_b[:, label.item()].backward()
        grad_b = gdbp.get(data)
        if label == 1:
            grad_pa.append(grad_b)
        else:
            grad_normal.append(grad_b)
    # print(grad)
    print("patient:")
    ffff(grad_pa)
    print("normal")
    ffff(grad_normal)
    # print(g)



    # indices = np.argpartition(g, -10, axis=1)[:,-10:]
    # print(indices)
    # a = [0] * 117
    # for x in indices:
    #     for y in x:
    #         a[y+1] = a[y+1] + 1
    # print(a)