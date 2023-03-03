import math
import random

import numpy as np
from opacus import PrivacyEngine

seed = 0
random.seed(seed)
np.random.seed(seed)

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

# Default settings
parser = argparse.ArgumentParser(description='K3DA Official Implement')
# Dataset Parameters
parser.add_argument("--config", default="ABIDE.yaml")
parser.add_argument('-bp', '--base-path', default="../data/")
parser.add_argument('--target-domain', default="NYU", type=str, help="The target domain we want to perform domain adaptation")   #domains = ['Leuven', 'NYU', 'UCLA', 'UM', 'USM']
parser.add_argument('--source-domains', type=str, nargs="+", help="The source domains we want to use")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--beta', default=0.01, type=float)
parser.add_argument('--delta', default=0.85, type=float)
parser.add_argument('--com', default=1, type=int)
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
# set the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn


# sys.stdout = open("D:/YQM/code/MSDA/console/{}.txt".format(args.target_domain), "w")


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def main(args=args, configs=configs):
    # set the dataloader list, model list, optimizer list, optimizer schedule list
    train_dloaders = []
    test_dloaders = []
    models = []
    classifiers = []
    optimizers = []
    classifier_optimizers = []
    optimizer_schedulers = []
    classifier_optimizer_schedulers = []
    # build dataset
    if configs["DataConfig"]["dataset"] == "schizophrenia":
        domains = ['data_huang45', 'COBRE_120_L1', 'Nottingham_68_L1', 'Taiwan_131_L1', 'Xiangya_143_L1']
        target_train_dloader = get_schizophrenia_dloader(args.base_path,
                                                         args.target_domain,
                                                         configs["TrainingConfig"]["batch_size"],
                                                         args.workers)
        train_dloaders.append(target_train_dloader)
        # test_dloaders.append(target_test_dloader)
        models.append(schiNet().cuda())
        classifiers.append(schiClassifier().cuda())
        domains.remove(args.target_domain)
        args.source_domains = domains
        for domain in domains:
            source_train_dloader = get_schizophrenia_dloader(args.base_path, domain,
                                                                              configs["TrainingConfig"]["batch_size"],
                                                                              args.workers)
            train_dloaders.append(source_train_dloader)
            # test_dloaders.append(source_test_dloader)
            models.append(schiNet().cuda())
            classifiers.append(schiClassifier().cuda())
        num_classes = 2
    elif configs["DataConfig"]["dataset"] == "ABIDE":
        domains = ['Leuven', 'NYU', 'UCLA', 'UM', 'USM']
        target_train_dloader = get_ABIDE_dloader(args.base_path,
                                                         domains.index(args.target_domain),
                                                         configs["TrainingConfig"]["batch_size"],
                                                         args.workers)
        train_dloaders.append(target_train_dloader)
        # test_dloaders.append(target_test_dloader)
        models.append(ABIDENet().cuda())
        classifiers.append(ABIDEClassifier().cuda())
        # load model dic
        if args.train_time > 1:
            load_model = torch.load('../data/ABIDE/parameter/train_time_{}/{}.pth.tar'.format(args.train_time-1, args.target_domain))
            models[0].load_state_dict(load_model['backbone'])
            classifiers[0].load_state_dict(load_model['classifier'])
        for domain in domains:
            if domain == args.target_domain:
                continue
            source_train_dloader = get_ABIDE_dloader(args.base_path,
                                                            domains.index(domain),
                                                            configs["TrainingConfig"]["batch_size"],
                                                            args.workers)
            train_dloaders.append(source_train_dloader)
            models.append(ABIDENet().cuda())
            classifiers.append(ABIDEClassifier().cuda())
        num_classes = 2
        domains.remove(args.target_domain)
        args.source_domains = domains
    else:
        raise NotImplementedError("Dataset {} not implemented".format(configs["DataConfig"]["dataset"]))
    # federated learning step 1: initialize model with the same parameter (use target as standard)
    for model in models[1:]:
        for source_weight, target_weight in zip(model.named_parameters(), models[0].named_parameters()):
            # consistent parameters
            source_weight[1].data = target_weight[1].data.clone()
    # create the optimizer for each model
    for model in models:
        optimizers.append(
            torch.optim.SGD(model.parameters(), momentum=args.momentum,
                            lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
    for classifier in classifiers:
        classifier_optimizers.append(
            torch.optim.SGD(classifier.parameters(), momentum=args.momentum,
                            lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
    # create the optimizer scheduler with cosine annealing schedule
    for optimizer in optimizers:
        optimizer_schedulers.append(
            CosineAnnealingLR(optimizer, configs["TrainingConfig"]["total_epochs"],
                              eta_min=configs["TrainingConfig"]["learning_rate_end"]))
    for classifier_optimizer in classifier_optimizers:
        classifier_optimizer_schedulers.append(
            CosineAnnealingLR(classifier_optimizer, configs["TrainingConfig"]["total_epochs"],
                              eta_min=configs["TrainingConfig"]["learning_rate_end"]))

    # 是否使用DP加密
    if configs["UMDAConfig"]["is_dp"] == True:
        for i in range(5):
            privacy_engine = PrivacyEngine()
            models[i], optimizers[i], train_dloaders[i] = privacy_engine.make_private(
                module=models[i],
                optimizer=optimizers[i],
                data_loader=train_dloaders[i],
                noise_multiplier=0.3,
                max_grad_norm=1.0,
                poisson_sampling=False,
            )

            classifiers[i], classifier_optimizers[i], _ = privacy_engine.make_private(
                module=classifiers[i],
                optimizer=classifier_optimizers[i],
                data_loader=train_dloaders[i],
                noise_multiplier=0.3,
                max_grad_norm=1.0,
            )


    # create the event to save log info
    if configs["UMDAConfig"]["sparsity_mmd"] == True:
        writer_log_dir = path.join("logs", configs["DataConfig"]["dataset"], "runs_contrastive_com_{}".format(args.com),
                                   "train_seed_{}_{}_{}".format(seed, args.delta, args.beta) + "_" +
                                   args.target_domain)
    else:
        writer_log_dir = path.join("logs", configs["DataConfig"]["dataset"], "runs_contrastive",
                                   "train_seed_{}_no_mmd".format(seed) + "_" +
                                   args.target_domain)



    print("create writer in {}".format(writer_log_dir))
    if os.path.exists(writer_log_dir):
        # flag = input("{} train_time:{} will be removed, input yes to continue:".format(
        #     configs["DataConfig"]["dataset"], args.train_time))
        flag = "yes"
        if flag == "yes":
            shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)
    # begin train
    print("Begin the {} time's training, Dataset:{}, Source Domains {}, Target Domain {}".format(args.train_time,
                                                                                                 configs[
                                                                                                     "DataConfig"][
                                                                                                     "dataset"],
                                                                                                 args.source_domains,
                                                                                                 args.target_domain))

    # create the initialized domain weight
    domain_weight = create_domain_weight(len(args.source_domains))
    # adjust training strategy with communication round
    # train model
    feature_rep = [torch.tensor([0]*128, dtype=float)] * 4
    positive, negative = 0, 0
    accs = []
    for epoch in range(args.start_epoch, configs["TrainingConfig"]["total_epochs"]):
        domain_weight, feature_rep, positive, negative = train(train_dloaders, models, classifiers, optimizers,
                              classifier_optimizers, epoch, writer, num_classes=num_classes,
                              domain_weight=domain_weight, source_domains=args.source_domains,
                              sparsity_mmd=configs["UMDAConfig"]["sparsity_mmd"],
                              communication_rounds=args.com,
                              confidence_gate_begin=args.delta,
                              confidence_gate_end=configs["UMDAConfig"]["confidence_gate_end"],
                              feature_rep = feature_rep,
                              positive_b=positive,
                              negative_b=negative,
                              beta=args.beta,
                              t_sne=args.tsne)
        acc = test(args.target_domain, args.source_domains, train_dloaders, models, classifiers, epoch,
             writer, num_classes=num_classes, top_5_accuracy=(num_classes > 10))
        accs.append(acc)
        # if epoch > 40:
        #     if accs[-20:].count(acc) == 20:
        #         print('epoch end:{}    acc:{}'.format(epoch-18, acc))
        #         break
        for scheduler in optimizer_schedulers:
            scheduler.step()
        for scheduler in classifier_optimizer_schedulers:
            scheduler.step()
        # save models every 10 epochs
        if (epoch + 1) % 100 == 0:
            print('*****'*30)
            print("epoch:{}".format(epoch))
        if (epoch + 1) % 10 == 0:
            # save target model with epoch, domain, model, optimizer
            save_checkpoint(
                {"epoch": epoch + 1,
                 "domain": args.target_domain,
                 "backbone": models[0].state_dict(),
                 "classifier": classifiers[0].state_dict(),
                 "optimizer": optimizers[0].state_dict(),
                 "classifier_optimizer": classifier_optimizers[0].state_dict()
                 },
                filename="{}.pth.tar".format(args.target_domain))


def save_checkpoint(state, filename):
    filefolder = "{}/{}/parameter/train_time_{}_{}_{}".format(args.base_path, configs["DataConfig"]["dataset"],
                                                        args.train_time,
                                                              args.target_domain,
                                                              args.delta)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == "__main__":
    main()
