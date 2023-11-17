import json
import math
from copy import deepcopy

import torch.nn.functional as F

from LibMTL.aspp import DeepLabHead
from LibMTL.create_dataset import NYUv2, Cityscapes
from LibMTL import Trainer
from LibMTL.model import resnet_dilated
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from LibMTL._record import _PerformanceMeter
from LibMTL.utils import *

from utils.logger import CompleteLogger


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


def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_mode', default='train', type=str, help='train')
    parser.add_argument('--train_bs', default=8, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=8, type=int, help='batch size for test')
    parser.add_argument('--dataset', default="NYUv2", choices=['NYUv2', 'Cityscapes'])
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--source_tasks', default=['segmentation', 'depth', 'normal'], nargs='+')
    parser.add_argument('--target_tasks', default=None, nargs='+')
    parser.add_argument('--alphas', type=float, default=[0, 0.2, 0.4, 0.6, 0.8, 1.0], nargs='+')
    parser.add_argument('--epoch_step', type=int, default=1)
    parser.add_argument('--pruning_epochs', type=int, default=10)
    parser.add_argument('--topk', type=int, default=[0, 1, 2], nargs='+')
    parser.add_argument('--task_weights', default=[1, 1, 1], type=float, nargs='+', help='weight specific for EW')
    parser.add_argument('--pretrained', default=None)

    return parser.parse_args()


def main(params):
    logger = CompleteLogger(params.log, params.phase)
    print(params)
    params.tasks = params.source_tasks
    kwargs, optim_param, scheduler_param = prepare_args(params)

    # prepare dataloaders
    if params.dataset == "NYUv2":
        dataset = NYUv2
        base_result = {'segmentation': [0.5251, 0.7478], 'depth': [0.4047, 0.1719],
                       'normal': [22.6744, 15.9096, 0.3717, 0.6353, 0.7418]}
    else:
        dataset = Cityscapes
        base_result = {'segmentation': [0.7401, 0.9316], 'depth': [0.0125, 27.77]}
    nyuv2_train_set = dataset(root=params.dataset_path, mode="train", augmentation=params.aug)
    nyuv2_val_set = dataset(root=params.dataset_path, mode="val", augmentation=False)
    nyuv2_test_set = dataset(root=params.dataset_path, mode='test', augmentation=False)
    print("train: {} val: {} test: {}".format(len(nyuv2_train_set), len(nyuv2_val_set), len(nyuv2_test_set)))

    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set,
        batch_size=params.train_bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True)

    nyuv2_val_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_val_set,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    # define tasks
    task_dict = {'segmentation': {'metrics': ['mIoU', 'pixAcc'],
                                  'metrics_fn': SegMetric(num_classes=nyuv2_train_set.num_classes),
                                  'loss_fn': SegLoss(),
                                  'weight': [1, 1]},
                 'depth': {'metrics': ['abs_err', 'rel_err'],
                           'metrics_fn': DepthMetric(),
                           'loss_fn': DepthLoss(),
                           'weight': [0, 0]},
                 'normal': {'metrics': ['mean', 'median', '<11.25', '<22.5', '<30'],
                            'metrics_fn': NormalMetric(),
                            'loss_fn': NormalLoss(),
                            'weight': [0, 0, 1, 1, 1]},
                 'noise': {'metrics': ['dummy metric'],
                           'metrics_fn': NoiseMetric(),
                           'loss_fn': NoiseLoss(),
                           'weight': [1]}}
    source_task_dict = {task_name: task_dict[task_name] for task_name in params.source_tasks}
    target_task_dict = {task_name: task_dict[task_name] for task_name in params.target_tasks}
    base_result = {task_name: base_result[task_name] for task_name in params.target_tasks}

    # define encoder and decoders
    def encoder_class():
        return resnet_dilated('resnet50')

    num_out_channels = nyuv2_train_set.num_out_channels
    decoders = nn.ModuleDict(
        {task: DeepLabHead(2048, num_out_channels[task]) for task in list(source_task_dict.keys())})
    print(decoders)

    class NYUtrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class,
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, save_path, base_result, img_size,
                     **kwargs):
            super(NYUtrainer, self).__init__(task_dict=task_dict,
                                             weighting=weighting_method.__dict__[weighting],
                                             architecture=architecture_method.__dict__[architecture],
                                             encoder_class=encoder_class,
                                             decoders=decoders,
                                             rep_grad=rep_grad,
                                             multi_input=multi_input,
                                             optim_param=optim_param,
                                             scheduler_param=scheduler_param,
                                             **kwargs)
            self.img_size = img_size
            self.base_result = base_result
            self.best_result = None
            self.best_epoch = None
            self.weight = {'segmentation': [1., 1.], 'depth': [0., 0.], 'normal': [0., 0., 1., 1., 1.]}
            self.best_improvement = -math.inf
            self.meter = _PerformanceMeter(self.task_dict, self.multi_input, base_result=None)

            self.save_path = save_path

        def process_preds(self, preds):
            for task in self.task_name:
                preds[task] = F.interpolate(preds[task], self.img_size, mode='bilinear', align_corners=True)
            return preds

        def _prepare_optimizer(self, optim_param, scheduler_param):
            optim_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'adagrad': torch.optim.Adagrad,
                'rmsprop': torch.optim.RMSprop,
                'adamw': torch.optim.AdamW
            }
            scheduler_dict = {
                'exp': torch.optim.lr_scheduler.ExponentialLR,
                'step': torch.optim.lr_scheduler.StepLR,
                'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
            }
            optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
            self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
            if scheduler_param is not None:
                scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
                self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
            else:
                self.scheduler = None

        def train(self, train_dataloaders, val_dataloaders, epoch_start, epoch_end, return_weight=False):
            r'''The training process of multi-task learning.

            Args:
                train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                                If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                                Otherwise, it is a single dataloader which returns data and a dictionary \
                                of name-label pairs in each iteration.

                test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                                The same structure with ``train_dataloaders``.
                epochs (int): The total training epochs.
                return_weight (bool): if ``True``, the loss weights will be returned.
            '''
            train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
            train_batch = max(train_batch) if self.multi_input else train_batch

            for epoch in range(epoch_start, epoch_end):
                self.model.epoch = epoch
                self.model.train()
                self.meter.record_time('begin')
                for batch_index in range(train_batch):
                    if not self.multi_input:
                        train_inputs, train_gts = self._process_data(train_loader)
                        train_preds = self.model(train_inputs)
                        train_preds = self.process_preds(train_preds)
                        train_losses = self._compute_loss(train_preds, train_gts)
                        self.meter.update(train_preds, train_gts)
                    else:
                        train_losses = torch.zeros(self.task_num).to(self.device)
                        for tn, task in enumerate(self.task_name):
                            train_input, train_gt = self._process_data(train_loader[task])
                            train_pred = self.model(train_input, task)
                            train_pred = train_pred[task]
                            train_pred = self.process_preds(train_pred, task)
                            train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                            self.meter.update(train_pred, train_gt, task)

                    self.optimizer.zero_grad()
                    w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                    self.optimizer.step()

                self.meter.record_time('end')
                self.meter.get_score()
                self.meter.display(epoch=epoch, mode='train')
                self.meter.reinit()

                if val_dataloaders is not None:
                    self.val(val_dataloaders, epoch)
                print()
                if self.scheduler is not None:
                    self.scheduler.step()
            self.display_best_result()

        def display_best_result(self):
            print('=' * 40)
            print('Best Result: Epoch {}, result {} improvement: {}'.format(self.best_epoch, self.best_result,
                                                                            self.best_improvement))
            print('=' * 40)

        def val(self, val_dataloaders, epoch=None):
            self.meter.has_val = True
            new_result = self.test(val_dataloaders, epoch, mode='val')

            from LibMTL.utils import count_improvement
            improvement = count_improvement(self.base_result, new_result, self.weight)
            print("improvement", improvement)
            if improvement > self.best_improvement:
                self.save()
                self.best_result = new_result
                self.best_epoch = epoch
            self.best_improvement = max(improvement, self.best_improvement)

        def test(self, test_dataloaders, epoch=None, mode='test'):
            r'''The test process of multi-task learning.

            Args:
                test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                                it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                                dataloader which returns data and a dictionary of name-label pairs in each iteration.
                epoch (int, default=None): The current epoch.
            '''
            test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)

            self.model.eval()
            self.meter.record_time('begin')
            with torch.no_grad():
                if not self.multi_input:
                    for batch_index in range(test_batch):
                        test_inputs, test_gts = self._process_data(test_loader)
                        test_preds = self.model(test_inputs)
                        test_preds = self.process_preds(test_preds)
                        test_losses = self._compute_loss(test_preds, test_gts)
                        self.meter.update(test_preds, test_gts)
                else:
                    for tn, task in enumerate(self.task_name):
                        for batch_index in range(test_batch[tn]):
                            test_input, test_gt = self._process_data(test_loader[task])
                            test_pred = self.model(test_input, task)
                            test_pred = test_pred[task]
                            test_pred = self.process_preds(test_pred)
                            test_loss = self._compute_loss(test_pred, test_gt, task)
                            self.meter.update(test_pred, test_gt, task)
            self.meter.record_time('end')
            self.meter.get_score()
            results = deepcopy(self.meter.results)
            self.meter.display(epoch=epoch, mode=mode)
            self.meter.reinit()
            results = {task_name: results[task_name] for task_name in params.target_tasks}
            return results

        def load_best(self):
            self.model.load_state_dict(torch.load(self.save_path, map_location='cpu'))

        def save(self):
            torch.save(self.model.state_dict(), self.save_path)

    lambda_history = []
    performance_dict = {}

    for aux_task_name in params.source_tasks:
        if aux_task_name not in params.target_tasks:
            new_task_dict = deepcopy(target_task_dict)
            new_task_dict[aux_task_name] = task_dict[aux_task_name]
            print(new_task_dict)
            kwargs['weight_args']['weights'] = len(new_task_dict) * [1]
            trainer = NYUtrainer(task_dict=new_task_dict,
                                 weighting=params.weighting,
                                 architecture=params.arch,
                                 encoder_class=encoder_class,
                                 decoders=deepcopy(decoders),
                                 rep_grad=params.rep_grad,
                                 multi_input=params.multi_input,
                                 optim_param=optim_param,
                                 scheduler_param=scheduler_param,
                                 base_result=base_result,
                                 save_path=logger.get_checkpoint_path(aux_task_name),
                                 img_size=nyuv2_train_set.image_size,
                                 **deepcopy(kwargs))

            trainer.train(nyuv2_train_loader, nyuv2_val_loader, 0, params.pruning_epochs)
            performance_dict[aux_task_name] = trainer.best_improvement

    ranked_performances = sorted(performance_dict.items(), key=lambda kv: kv[1], reverse=True)
    ranked_task_names = [task_name for task_name, _ in ranked_performances]
    print(ranked_performances)
    print(ranked_task_names)

    trainers = {}
    performance_dict = {}
    for topk in params.topk:
        task_names = ranked_task_names[:topk]
        new_task_dict = deepcopy(target_task_dict)
        for aux_task_name in task_names:
            new_task_dict[aux_task_name] = task_dict[aux_task_name]
        print(new_task_dict)
        kwargs['weight_args']['weights'] = len(new_task_dict) * [1]
        print(kwargs['weight_args']['weights'])
        trainers[topk] = NYUtrainer(task_dict=new_task_dict,
                                    weighting=params.weighting,
                                    architecture=params.arch,
                                    encoder_class=encoder_class,
                                    decoders=deepcopy(decoders),
                                    rep_grad=params.rep_grad,
                                    multi_input=params.multi_input,
                                    optim_param=optim_param,
                                    scheduler_param=scheduler_param,
                                    base_result=base_result,
                                    save_path=logger.get_checkpoint_path("top_{}".format(topk)),
                                    img_size=nyuv2_train_set.image_size,
                                    **deepcopy(kwargs))

        if params.pretrained is not None:
            print("Loading from ", params.pretrained)
            trainers[topk].model.load_state_dict(torch.load(params.pretrained, map_location='cpu'), strict=False)

    target_trainer = list(trainers.values())[0]
    theta_dict = {}

    epoch_start = 0
    epoch_end = params.epoch_step
    overall_best_improvement = -math.inf
    while epoch_start < params.n_epochs:
        print("Epoch: {}=>{}".format(epoch_start, epoch_end))
        for name, trainer in trainers.items():
            print("forking top", name)
            trainer.train(nyuv2_train_loader, nyuv2_val_loader, epoch_start, epoch_end)
            trainer.load_best()
            theta_dict[name] = deepcopy(trainer.model.state_dict())
            performance_dict[name] = trainer.best_improvement

        print("merging")
        ranked_performances = sorted(performance_dict.items(), key=lambda kv: kv[1], reverse=True)
        print(ranked_performances)
        lambda_dict = {name: 0 for name, _ in ranked_performances}
        lambda_dict[ranked_performances[0][0]] = 1

        criteria_dict = {}

        best_improvement = -math.inf

        for i, (name, _) in enumerate(ranked_performances[1:]):
            best_lambda = 0
            upper_bound = sum(lambda_dict.values()) / (i + 1)
            # print("upper_bound", upper_bound)
            for alpha in params.alphas:
                new_lambda = alpha * upper_bound
                lambda_dict[name] = new_lambda
                theta = sum_all_weights(theta_dict, lambda_dict)
                # strict = False
                target_trainer.model.load_state_dict(theta, strict=False)

                # evaluate on validation set
                new_result = target_trainer.test(nyuv2_val_loader, epoch_end, mode='val')

                improvement = count_improvement(target_trainer.base_result, new_result, target_trainer.weight)

                # remember best acc@1 and save checkpoint
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_lambda = new_lambda
                criteria_dict["{}_{}".format(name, new_lambda)] = improvement
                print("lambda: {}, val_criteria: {}".format(new_lambda, improvement))
            lambda_dict[name] = best_lambda

        print("Epoch: {} Result: {} Best lambda: {} Best criteria: {}".format(epoch_end, criteria_dict, lambda_dict,
                                                                              best_improvement))

        print(lambda_dict)
        lambda_history.append(lambda_dict)
        theta = sum_all_weights(theta_dict, lambda_dict)

        for name, trainer in trainers.items():
            trainer.model.load_state_dict(theta, strict=False)

        if best_improvement > overall_best_improvement:
            overall_best_improvement = best_improvement
            torch.save(target_trainer.model.state_dict(), logger.get_checkpoint_path('best'))
        epoch_start = epoch_end
        epoch_end = min(epoch_end + params.epoch_step, params.n_epochs)

    with open(os.path.join(params.log, 'lambda_history.json'), 'w') as f:
        json.dump(lambda_history, f)
    print("overall_best_improvement", overall_best_improvement)
    target_trainer.model.load_state_dict(torch.load(logger.get_checkpoint_path('best')), strict=False)
    final_results = target_trainer.test(nyuv2_test_loader, 200, mode='test')
    print("test", final_results)
    logger.close()


if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    set_device(params.gpu_id)
    set_random_seed(params.seed)
    main(params)
