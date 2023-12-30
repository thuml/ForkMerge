import abc
import time
import random
import warnings
import argparse
from copy import deepcopy
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.logger import CompleteLogger
from utils.data import ForeverDataIterator
from utils.meter import AverageMeter, ProgressMeter
from utils.metric import accuracy
from utils.classifier import MultiOutputClassifier
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sum_all_weights(theta_dict, alpha_dict, no_sum=[]):
    theta_0 = theta_dict[list(theta_dict.keys())[0]]
    alpha_dict = {k: v / sum(alpha_dict.values()) for k, v in alpha_dict.items()}

    theta = {}
    for key in theta_0.keys():
        if not any(ns in key for ns in no_sum):
            theta[key] = 0
            for task_name in theta_dict.keys():
                theta[key] += alpha_dict[task_name] * theta_dict[task_name][key]
        else:
            theta[key] = theta_0[key]
    return theta


class BaseMultiTaskSampler(metaclass=abc.ABCMeta):
    def __init__(self, task_dict: dict, rng: Union[int, np.random.RandomState, None]):
        self.task_dict = task_dict
        if isinstance(rng, int) or rng is None:
            rng = np.random.RandomState(rng)
        self.rng = rng
        self.task_names = list(task_dict.keys())

    def pop(self):
        raise NotImplementedError()

    def iter(self):
        yield self.pop()


class SpecifiedProbMultiTaskSampler(BaseMultiTaskSampler):
    def __init__(
            self,
            task_dict: dict,
            rng: Union[int, np.random.RandomState],
            task_to_unweighted_probs: dict,
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_unweighted_probs.keys()
        self.task_to_unweighted_probs = task_to_unweighted_probs
        self.task_names = list(task_to_unweighted_probs.keys())
        self.unweighted_probs_arr = np.array([task_to_unweighted_probs[k] for k in self.task_names])
        self.task_p = self.unweighted_probs_arr / self.unweighted_probs_arr.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]


class Trainer:
    def __init__(self, name, model, save_path, optimizer_name, lr, momentum, weight_decay, epochs,
                 task_to_unweighted_probs):
        self.name = name
        self.model = model
        self.save_path = save_path
        self.epochs = epochs
        self.optimizer = utils.get_optimizer(model, optimizer_name, lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.best_acc1 = {}
        self.task_to_unweighted_probs = task_to_unweighted_probs

    def train(self, train_loaders, epoch_start, epoch_end, iters_per_epoch, val_loaders):
        dataset_sampler = SpecifiedProbMultiTaskSampler(train_loaders, 0, self.task_to_unweighted_probs)
        for epoch in range(epoch_start, epoch_end):
            print(self.scheduler.get_lr())
            # train for one epoch
            self.train_one_epoch(dataset_sampler, self.model, self.optimizer, epoch, iters_per_epoch, args, device)
            self.scheduler.step()

            # evaluate on validation set
            acc1 = utils.validate_all(val_loaders, self.model, args, device)

            # remember best acc@1 and save checkpoint
            if sum(acc1.values()) > sum(self.best_acc1.values()):
                self.save()
                self.best_acc1 = acc1
            print(self.name, "Epoch:", epoch, "lr:", self.scheduler.get_lr()[0], "val_criteria:",
                  round(sum(acc1.values()) / len(acc1), 3),
                  "best_val_criteria:", round(sum(self.best_acc1.values()) / len(self.best_acc1), 3))

    def train_one_epoch(self, dataset_sampler, model, optimizer, epoch, iters_per_epoch, args, device):
        batch_time = AverageMeter('Time', ':5.2f')
        data_time = AverageMeter('Data', ':5.2f')
        losses = AverageMeter('Loss', ':6.2f')
        accs = AverageMeter('Acc', ':3.1f')
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, losses, accs],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()
        for i in range(iters_per_epoch):
            dataset_name, dataloader = dataset_sampler.pop()
            x, labels = next(dataloader)[:2]

            x = x.to(device)
            labels = labels.to(device)

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            y = model(x, dataset_name)

            loss = F.cross_entropy(y, labels)

            acc = accuracy(y, labels)[0]

            losses.update(loss.item(), x.size(0))
            accs.update(acc.item(), x.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    def test(self, val_loaders):
        acc1 = utils.validate_all(val_loaders, self.model, args, device)
        return sum(acc1.values()) / len(acc1), acc1

    def load_best(self):
        self.model.load_state_dict(torch.load(self.save_path, map_location='cpu'))

    def save(self):
        torch.save(self.model.state_dict(), self.save_path)


def main(args):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_datasets, train_target_datasets, val_datasets, test_datasets, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    val_loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers) for
                   name, dataset in val_datasets.items()}
    test_loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers) for
                    name, dataset in test_datasets.items()}

    train_loaders = {name: ForeverDataIterator(DataLoader(dataset, batch_size=args.batch_size,
                                                          shuffle=True, num_workers=args.workers, drop_last=True)) for
                     name, dataset in train_source_datasets.items()}

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    heads = nn.ModuleDict({
        dataset_name: nn.Linear(backbone.out_features, num_classes) for dataset_name in args.source
    })
    model = MultiOutputClassifier(backbone, heads, pool_layer=pool_layer, finetune=not args.scratch).to(device)
    if args.pretrained is not None:
        print("Loading from ", args.pretrained)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
    print(heads)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        model.load_state_dict(checkpoint)

    if args.phase == 'test':
        acc1 = utils.validate_all(test_loaders, model, args, device)
        print(acc1)
        return

    init_model_parameters = deepcopy(model.state_dict())

    performance_dict = {}
    task_weights = {name: 0 for name in args.source}
    for name in args.target:
        task_weights[name] = 1 / len(args.target)

    for aux_task_name in args.source:
        if aux_task_name not in args.target:
            model.load_state_dict(init_model_parameters)
            new_task_weights = deepcopy(task_weights)
            new_task_weights[aux_task_name] = 1
            print(new_task_weights)
            trainer = Trainer("ranking_{}".format(aux_task_name), model, logger.get_checkpoint_path(aux_task_name),
                              args.optimizer,
                              args.lr, args.momentum, args.wd, args.epochs, new_task_weights)
            trainer.train(train_loaders, 0, args.pruning_epochs, args.iters_per_epoch, val_loaders)
            performance_dict[aux_task_name] = sum(trainer.best_acc1.values()) / len(trainer.best_acc1)

    ranked_performances = sorted(performance_dict.items(), key=lambda kv: kv[1], reverse=True)
    ranked_task_names = [task_name for task_name, _ in ranked_performances]
    print(ranked_performances)
    print(ranked_task_names)

    model.load_state_dict(init_model_parameters)
    trainers = {}
    for topk in args.topk:
        task_names = ranked_task_names[:topk]
        new_task_weights = deepcopy(task_weights)
        for task_name in task_names:
            new_task_weights[task_name] = 1. / topk
        print("top {}: {}".format(topk, new_task_weights))

        trainers[topk] = Trainer("top_{}".format(topk), deepcopy(model),
                                 logger.get_checkpoint_path("top_{}".format(topk)), args.optimizer,
                                 args.lr,
                                 args.momentum, args.wd, args.epochs, new_task_weights)
    target_trainer = list(trainers.values())[0]

    epoch_start = 0
    epoch_end = args.epoch_step
    performance_dict = {}
    theta_dict = {}

    while epoch_start < args.epochs:
        print("Epoch: {}=>{}".format(epoch_start, epoch_end))
        for name, trainer in trainers.items():
            print("forking top", name)
            trainer.train(train_loaders, epoch_start, epoch_end, args.iters_per_epoch, val_loaders)
            trainer.load_best()
            theta_dict[name] = deepcopy(trainer.model.state_dict())
            performance_dict[name] = sum(trainer.best_acc1.values()) / len(trainer.best_acc1)

        print("merging")
        ranked_performances = sorted(performance_dict.items(), key=lambda kv: kv[1], reverse=True)
        print(ranked_performances)
        lambda_dict = {name: 0. for name, _ in ranked_performances}
        lambda_dict[ranked_performances[0][0]] = 1

        criteria_dict = {}

        best_criteria = 0

        for i, (name, _) in enumerate(ranked_performances[1:]):
            best_lambda = 0
            upper_bound = sum(lambda_dict.values()) / (i + 1)

            for alpha in args.alphas:
                new_lambda = alpha * upper_bound
                lambda_dict[name] = new_lambda
                theta = sum_all_weights(theta_dict, lambda_dict)
                target_trainer.model.load_state_dict(theta)

                # evaluate on validation set
                val_criteria, _ = target_trainer.test(val_loaders)

                # remember best acc@1 and save checkpoint
                if val_criteria > best_criteria:
                    best_criteria = val_criteria
                    best_lambda = new_lambda
                criteria_dict["{}_{}".format(name, new_lambda)] = val_criteria
                print("lambda: {}, val_criteria: {}".format(new_lambda, val_criteria))
            lambda_dict[name] = best_lambda
        print("Epoch: {} Result: {} Best lambda: {} Best criteria: {}".format(epoch_end, criteria_dict, lambda_dict,
                                                                              best_criteria))

        theta = sum_all_weights(theta_dict, lambda_dict)

        for name, trainer in trainers.items():
            trainer.model.load_state_dict(theta)

        epoch_start = epoch_end
        epoch_end = min(epoch_end + args.epoch_step, args.epochs)

    test_criteria, test_acc1_dict = target_trainer.test(test_loaders)
    print("test_performance: {} {}".format(test_criteria, test_acc1_dict))

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='forkmerge algorithm')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='DomainNet', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: DomainNet)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N',
                        help='mini-batch size (default: 48)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=2500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--alphas', type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1.0], nargs='+')
    parser.add_argument('--epoch_step', type=int, default=5)
    parser.add_argument('--topk', type=int, default=[0, 3, 5], nargs='+')
    parser.add_argument('--pruning_epochs', type=int, default=1)
    parser.add_argument('--pretrained', default=None)
    args = parser.parse_args()
    main(args)
