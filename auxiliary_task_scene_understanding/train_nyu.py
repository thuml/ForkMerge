import math
from copy import deepcopy
import shutil

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


def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_mode', default='train', type=str, help='train')
    parser.add_argument('--train_bs', default=8, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=8, type=int, help='batch size for test')
    parser.add_argument('--dataset', default="NYUv2", choices=['NYUv2', 'Cityscapes'])
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--tasks', default=['segmentation', 'depth', 'normal'], nargs='+')
    parser.add_argument('--target_tasks', default=None, nargs='+')
    parser.add_argument('--task_weights', default=[1, 1, 1], type=float, nargs='+', help='weight specific for EW')
    parser.add_argument('--pretrained', default=None)
    return parser.parse_args()


def main(params):
    logger = CompleteLogger(params.log, params.phase)
    print(params)
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
    task_dict = {task_name: task_dict[task_name] for task_name in params.tasks}
    if params.target_tasks is None:
        params.target_tasks = params.tasks
    base_result = {task_name: base_result[task_name] for task_name in params.target_tasks}

    # define encoder and decoders
    def encoder_class():
        return resnet_dilated('resnet50')

    num_out_channels = nyuv2_train_set.num_out_channels
    decoders = nn.ModuleDict({task: DeepLabHead(2048,
                                                num_out_channels[task]) for task in list(task_dict.keys())})

    class NYUtrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class,
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, logger, base_result, img_size,
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

            self.logger = logger

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

        def train(self, train_dataloaders, val_dataloaders, epochs, return_weight=False):
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

            self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
            self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
            for epoch in range(epochs):
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
                    if w is not None:
                        self.batch_weight[:, epoch, batch_index] = w
                    self.optimizer.step()

                self.meter.record_time('end')
                self.meter.get_score()
                self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
                self.meter.display(epoch=epoch, mode='train')
                self.meter.reinit()
                torch.save(self.model.state_dict(), logger.get_checkpoint_path('latest'))

                if val_dataloaders is not None:
                    self.meter.has_val = True
                    new_result = self.test(val_dataloaders, epoch, mode='val')

                    from LibMTL.utils import count_improvement
                    improvement = count_improvement(self.base_result, new_result, self.weight)
                    print("improvement", improvement)
                    if improvement > self.best_improvement:
                        shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
                        self.best_result = new_result
                        self.best_epoch = epoch
                    self.best_improvement = max(improvement, self.best_improvement)
                print()
                if self.scheduler is not None:
                    self.scheduler.step()
            self.display_best_result()
            if return_weight:
                return self.batch_weight

        def display_best_result(self):
            print('=' * 40)
            print('Best Result: Epoch {}, result {} improvement: {}'.format(self.best_epoch, self.best_result,
                                                                            self.best_improvement))
            print('=' * 40)

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

    NYUmodel = NYUtrainer(task_dict=task_dict,
                          weighting=params.weighting,
                          architecture=params.arch,
                          encoder_class=encoder_class,
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          logger=logger,
                          base_result=base_result,
                          img_size=nyuv2_train_set.image_size,
                          **kwargs)
    if params.pretrained is not None:
        print("Loading from ", params.pretrained)
        NYUmodel.model.load_state_dict(torch.load(params.pretrained, map_location='cpu'), strict=False)

    NYUmodel.train(nyuv2_train_loader, nyuv2_val_loader, params.epochs)
    NYUmodel.model.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    final_results = NYUmodel.test(nyuv2_test_loader, 200, mode='test')
    print("test", final_results)
    logger.close()


if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    set_device(params.gpu_id)
    set_random_seed(params.seed)
    main(params)
