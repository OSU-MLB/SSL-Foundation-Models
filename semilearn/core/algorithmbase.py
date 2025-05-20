# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import os
import contextlib
import numpy as np
from inspect import signature
from collections import OrderedDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score

import pprint
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from semilearn.core.hooks import Hook, get_priority, CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, ParamUpdateHook, EvaluationHook, EMAHook, WANDBHook, AimHook
from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, Bn_Controller, EMA
from semilearn.core.criterions import CELoss, ConsistencyLoss
from semilearn.core.utils.metrics import unsupervised_scores

class AlgorithmBase:
    """
        Base class for algorithms
        init algorithm specific parameters and common parameters
        
        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
    """
    def __init__(
        self,
        args,
        net_builder,
        tb_log=None,
        logger=None,
        **kwargs):
        
        # common arguments
        self.args = args
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.actual_epochs = args.actual_epochs
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.lambda_u = args.ulb_loss_ratio 
        self.use_cat = args.use_cat
        self.use_amp = args.amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm
        self.img_size = args.img_size

        # commaon utils arguments
        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size

        # early stopping related arguments
        self.es_patience = args.es_patience
        self.es_patience_iters = int(self.es_patience * self.num_train_iter)
        print(f"es_patience_iters: {self.es_patience_iters}")
        # self.es_criteria = args.es_criteria
        if self.algorithm in ['supervised', 'fullysupervised']:
            print(f"Algorithm: {self.algorithm}. Setting es_criteria to train/sup_loss_avg")
            self.es_criteria = 'train/sup_loss_avg'
        else:
            print(f"Algorithm: {self.algorithm}. Setting es_criteria to eval/top-1-acc")
            self.es_criteria = 'train/total_loss_avg'
        self.early_stop = False
        self.resume_es = args.resume_es

        # common model related parameters
        self.it = 0
        self.start_epoch = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder
        self.ema = None

        # early stopping related parameters
        self.best_it_es = 0
        self.best_metric_es = None

        # best test accuracy
        self.best_it_test = 0
        self.best_metric_test = None

        # build dataset
        self.dataset_dict = self.set_dataset()

        # build data loader
        self.loader_dict = self.set_data_loader()

        # cv, nlp, speech builder different arguments
        self.model = self.set_model()
        self.ema_model = self.set_ema_model()

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()
        num_params_requires_grad = sum(p.numel() for group in self.optimizer.param_groups for p in group['params'] if p.requires_grad)
        self.print_fn(f"Number of parameters requires grad: {num_params_requires_grad}")

        # build supervised loss and unsupervised loss
        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()

        # other arguments specific to the algorithm
        # self.init(**kwargs)

        # set common hooks during training
        self._hooks = []  # record underlying hooks 
        self.hooks_dict = OrderedDict() # actual object to be used to call hooks
        self.set_hooks()

    def init(self, **kwargs):
        """
        algorithm specific init function, to add parameters into class
        """
        raise NotImplementedError
    

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, self.args.data_dir, self.args.include_lb_to_ulb)
        if dataset_dict is None:
            return dataset_dict

        ############ TODO: refactor this part
        train_lb_dset = dataset_dict['train_lb']
        train_ulb_dset = dataset_dict['train_ulb']
        eval_dset = dataset_dict['eval']
        eval_transform = eval_dset.transform
        train_lb_dset_oracle = copy.deepcopy(train_lb_dset)
        train_lb_dset_oracle.transform = eval_transform
        train_ulb_dset_oracle = copy.deepcopy(train_ulb_dset)
        train_ulb_dset_oracle.transform = eval_transform
        train_ulb_dset_oracle.is_ulb = False
        train_ulb_dset_oracle_aug = copy.deepcopy(train_ulb_dset)
        train_ulb_dset_oracle_aug.is_ulb = False
        dataset_dict['train_lb_oracle'] = train_lb_dset_oracle
        dataset_dict['train_ulb_oracle'] = train_ulb_dset_oracle
        dataset_dict['train_ulb_dset_oracle_aug'] = train_ulb_dset_oracle_aug
        ##############################

        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn("unlabeled data number: {}, labeled data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    def set_data_loader(self):
        """
        set loader_dict
        """
        if self.dataset_dict is None:
            return
            
        self.print_fn("Create train and test data loaders")
        loader_dict = {}
        loader_dict['train_lb'] = get_data_loader(self.args,
                                                  self.dataset_dict['train_lb'],
                                                  self.args.batch_size,
                                                  data_sampler=self.args.train_sampler,
                                                  num_iters=self.num_train_iter,
                                                  num_epochs=self.epochs,
                                                  num_workers=self.args.num_workers,
                                                  distributed=self.distributed)

        loader_dict['train_ulb'] = get_data_loader(self.args,
                                                   self.dataset_dict['train_ulb'],
                                                   self.args.batch_size * self.args.uratio,
                                                   data_sampler=self.args.train_sampler,
                                                   num_iters=self.num_train_iter,
                                                   num_epochs=self.epochs,
                                                   num_workers=2 * self.args.num_workers,
                                                   distributed=self.distributed)

        ############################## TODO: refactor this part
        loader_dict['train_lb_oracle'] = get_data_loader(self.args,
                                                            self.dataset_dict['train_lb_oracle'],
                                                            self.args.eval_batch_size,
                                                            data_sampler=None,
                                                            num_workers=self.args.num_workers,
                                                            drop_last=False)
        loader_dict['train_ulb_oracle'] = get_data_loader(self.args,
                                                            self.dataset_dict['train_ulb_oracle'],
                                                            self.args.eval_batch_size,
                                                            data_sampler=None,
                                                            num_workers=self.args.num_workers,
                                                            drop_last=False)
        loader_dict['train_ulb_dset_oracle_aug'] = get_data_loader(self.args,
                                                            self.dataset_dict['train_ulb_dset_oracle_aug'],
                                                            self.args.batch_size,
                                                            data_sampler=self.args.train_sampler,
                                                            num_iters=self.num_train_iter,
                                                            num_epochs=self.epochs,
                                                            num_workers=self.args.num_workers,
                                                            distributed=self.distributed)
        ##############################

        loader_dict['eval'] = get_data_loader(self.args,
                                              self.dataset_dict['eval'],
                                              self.args.eval_batch_size,
                                              # make sure data_sampler is None for evaluation
                                              data_sampler=None,
                                              num_workers=self.args.num_workers,
                                              drop_last=False)
        
        if self.dataset_dict['test'] is not None:
            loader_dict['test'] =  get_data_loader(self.args,
                                                   self.dataset_dict['test'],
                                                   self.args.eval_batch_size,
                                                   # make sure data_sampler is None for evaluation
                                                   data_sampler=None,
                                                   num_workers=self.args.num_workers,
                                                   drop_last=False)
        self.print_fn(f'[!] data loader keys: {loader_dict.keys()}')
        return loader_dict

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.num_train_iter,
                                                    num_warmup_steps=self.args.num_warmup_iter)
        return optimizer, scheduler

    def set_model(self):
        """
        initialize model
        """
        # model = self.net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrain_path, img_size=self.img_size)
        model = self.net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrained_path)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        # ema_model = self.net_builder(num_classes=self.num_classes, img_size=self.img_size)
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    # def set_ema_model(self, state_dict=None):
    #     """
    #     initialize ema model from model
    #     """
    #     ema_model = self.net_builder(num_classes=self.num_classes, img_size=self.img_size)
    #     if state_dict is None:
    #         ema_model.load_state_dict(self.model.state_dict())
    #     else:
    #         ema_model.load_state_dict(state_dict)
    #     return ema_model

    def set_hooks(self):
        """
        register necessary training hooks
        """
        # parameter update hook is called inside each train_step
        self.register_hook(ParamUpdateHook(), None, "HIGHEST")
        self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")
        if self.args.use_wandb:
            self.register_hook(WANDBHook(), None, "LOWEST")
        if self.args.use_aim:
            self.register_hook(AimHook(), None, "LOWEST")

    def process_batch(self, input_args=None, **kwargs):
        """
        process batch data, send data to cuda
        NOTE **kwargs should have the same arguments to train_step function as keys to work properly
        """
        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        input_dict = {}

        for arg, var in kwargs.items():
            if not arg in input_args:
                continue
            
            if var is None:
                continue
            
            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict
    

    def process_out_dict(self, out_dict=None, **kwargs):
        """
        process the out_dict as return of train_step
        """
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var
        
        # process res_dict, add output from res_dict to out_dict if necessary
        return out_dict


    def process_log_dict(self, log_dict=None, prefix='train', **kwargs):
        """
        process the tb_dict as return of train_step
        """
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f'{prefix}/' + arg] = var
        return log_dict

    def compute_prob(self, logits):
        return torch.softmax(logits, dim=-1)

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        """
        train_step specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model 
        # record log_dict
        # return log_dict
        raise NotImplementedError

    def valid_stage_train(self):
        """
        valid stage training
        """

        assert False, "not supported anymore. "
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        
        # ulb_loader, lb_loader, 
        # 1. Generate pseudo training labels using model
        # TODO: refactor this part

        _state_dict = copy.deepcopy(self.model.state_dict())
        self.load_checkpoint(self.args.load_path)

        ulb_oracle_eval_dict = self.evaluate(eval_dest='train_ulb_oracle', out_key='logits', return_logits=True, return_y_true=True, return_idx=True)

        self.print_fn(f"train_ulb_oracle/loss: {ulb_oracle_eval_dict['train_ulb_oracle/loss']}")
        self.print_fn(f"train_ulb_oracle/top-1-acc: {ulb_oracle_eval_dict['train_ulb_oracle/top-1-acc']}")
        self.print_fn(f"train_ulb_oracle/top-5-acc: {ulb_oracle_eval_dict['train_ulb_oracle/top-5-acc']}")
        self.print_fn(f"train_ulb_oracle/balanced_acc: {ulb_oracle_eval_dict['train_ulb_oracle/balanced_acc']}")
        self.print_fn(f"train_ulb_oracle/precision: {ulb_oracle_eval_dict['train_ulb_oracle/precision']}")
        self.print_fn(f"train_ulb_oracle/recall: {ulb_oracle_eval_dict['train_ulb_oracle/recall']}")
        self.print_fn(f"train_ulb_oracle/F1: {ulb_oracle_eval_dict['train_ulb_oracle/F1']}")

        train_ulb_logits = ulb_oracle_eval_dict['train_ulb_oracle/logits']
        # Padding train_ulb_logits
        idx = ulb_oracle_eval_dict['train_ulb_oracle/idx']
        target_len = len(self.loader_dict['train_ulb_dset_oracle_aug'].dataset.targets)
        # new_targets = np.zeros((target_len, self.num_classes))
        # for _logits, _idx in zip(train_ulb_logits, idx):
        #     new_targets[_idx] = _logits
        new_targets = np.full(target_len, -1)
        for _logits, _idx in zip(train_ulb_logits, idx):
            new_targets[_idx] = np.argmax(_logits)

        # 2. Replace 
        #   a. self.loader_dict['train_lb'] with pseudo labels 
        #   b. self.loader_dict['train_ulb'] with None
        #   c. self.loader_dict['eval'] with self.loader_dict['train_lb_oracle']

        self.loader_dict['train_lb'] = self.loader_dict['train_ulb_dset_oracle_aug']
        self.loader_dict['train_lb'].dataset.set_targets(new_targets)
        self.loader_dict['train_lb'].dataset.set_idx_list(idx)
        self.loader_dict['train_ulb'] = None
        self.loader_dict['eval'] = self.loader_dict['train_lb_oracle']

        self.print_fn(f"len(train_lb): {len(self.loader_dict['train_lb'].dataset)}")
        self.print_fn(f"len(eval): {len(self.loader_dict['eval'].dataset)}")


        # Recover the model
        self.load_state_dict(_state_dict)

        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        
        # 4. Train the model until convergence
        return self.train()


    def _check_stop(self):
        if self.it >= self.num_train_iter:
            self.print_fn("***************** Iteration limit reached")
            return True
        if self.actual_epochs > 0 and self.epoch >= self.actual_epochs:
            self.print_fn("***************** Actual epoch limit reached")
            return True
        if self.early_stop:
            if self.resume_es == False:
                print("***************** Early stopping")
                self.print_fn("***************** Early Stopped")
                return True
            else:
                self.print_fn("***************** Resume from early stopping")
        return False


    def train(self):
        """
        train function
        """

        self.model.train()

        self.call_hook("before_run")

        self.print_fn(f"Start training, start epoch: {self.start_epoch}, total epochs: {self.epochs}")


        if self.args.pretrained_path != '':
            self.print_fn(f"Using pretrained model from {self.args.pretrained_path}, performing initial evaluation")
            eval_dict = self.evaluate('eval')
            self.print_fn(f"Initial eval: {pprint.pformat(eval_dict)}")
            test_dict = self.evaluate('test')
            self.print_fn(f"Initial test: {pprint.pformat(test_dict)}")
            self.model.train()

        t_start = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            t1 = time.time()
            self.epoch = epoch
            # prevent the training iterations exceed args.num_train_iter
            # if self.it >= self.num_train_iter or self.early_stop:
            #     print("***************** Stopped")
            #     break
            if self._check_stop():
                self.print_fn("***************** Stopped")
                break

            self.print_fn(f"Epoch {self.epoch}/{self.epochs}")
            
            self.call_hook("before_train_epoch")

            lb_iter = iter(self.loader_dict['train_lb'])
            ulb_iter = iter(self.loader_dict['train_ulb'])

            lb_dset = self.loader_dict['train_lb'].dataset
            print(f'lb_dset transform: {lb_dset.transform}')
            print(f'lb_dset medium transform: {lb_dset.medium_transform}')
            print(f'lb_dset strong transform: {lb_dset.strong_transform}')
            ulb_dset = self.loader_dict['train_ulb'].dataset
            print(f'ulb_dset transform: {ulb_dset.transform}')
            print(f'ulb_dset medium transform: {ulb_dset.medium_transform}')
            print(f'ulb_dset strong transform: {ulb_dset.strong_transform}')
            eval_dset = self.loader_dict['eval'].dataset
            print(f'eval_dset transform: {eval_dset.transform}')
            print(f'eval_dset medium transform: {eval_dset.medium_transform}')
            print(f'eval_dset strong transform: {eval_dset.strong_transform}')
            test_dset = self.loader_dict['test'].dataset
            print(f'test_dset transform: {test_dset.transform}')
            print(f'test_dset medium transform: {test_dset.medium_transform}')
            print(f'test_dset strong transform: {test_dset.strong_transform}')

            try:
                for data_lb, data_ulb in zip(lb_iter, ulb_iter):
                    # prevent the training iterations exceed args.num_train_iter
                    # if self.it >= self.num_train_iter or self.early_stop:
                    #     print("***************** Stopped")
                    #     break
                    if self._check_stop():
                        print("***************** Stopped")
                        break

                    n_data_lb = len(data_lb['y_lb'])
                    n_data_ulb = len(data_ulb['x_ulb_w'])
                    n_data = n_data_lb + n_data_ulb

                    self.call_hook("before_train_step")
                    self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                    self.n_data_lb = n_data_lb
                    self.n_data_ulb = n_data_ulb
                    self.call_hook("after_train_step")
                    self.it += 1
            finally:
                del lb_iter
                del ulb_iter
            
            t7 = time.time()
            self.print_fn(f"Time for epoch {epoch}: {t7 - t1}")
            
            self.call_hook("after_train_epoch")
        t_end = time.time()
        self.print_fn(f"Time taken for training: {t_end - t_start}")

        self.call_hook("after_run")


    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False, return_y_true=False, return_idx=False):
        """
        evaluation function
        """

        self.print_fn(f"Evaluating on {eval_dest}")
        start = time.time()

        self.model.eval()
        self.ema.apply_shadow()

        if eval_dest not in self.loader_dict:
            self.print_fn(f"Loader for {eval_dest} not found, skipping evaluation")
            return
        eval_loader = self.loader_dict[eval_dest]

        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_probs = []
        y_logits = []
        y_feats = []

        # idx = []
        print(f'len(eval_loader): {len(eval_loader)}')
        print(f'len(eval_loader.dataset): {len(eval_loader.dataset)}')
        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                _start = time.time()
                idx_lb = data['idx_lb']
                # has attribute idx_list
                # _idx = [eval_loader.dataset.idx_list[i] for i in idx_lb]
                x = data['x_lb']
                y = data['y_lb']
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                out = self.model(x)
                logits = out[out_key]
                feat = out['feat']
                
                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                _y_true = y.cpu().tolist()
                _y_pred = torch.max(logits, dim=-1)[1].cpu().tolist()
                _y_logits = logits.cpu().numpy()
                _y_feats = feat.cpu().numpy()
                _y_probs = torch.softmax(logits, dim=-1).cpu().tolist()
                y_true.extend(_y_true)
                y_pred.extend(_y_pred)
                y_logits.append(_y_logits)
                y_feats.append(_y_feats)
                y_probs.extend(_y_probs)
                total_loss += loss.item() * num_batch
                _end = time.time()
                # print(f"Time for batch {i}: {_end - _start}")
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        y_feats = np.concatenate(y_feats)

        eval_dict = dict()

        # Supervised metrics
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_probs, k=5)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        
        eval_dict[f'{eval_dest}/total_num'] = total_num
        eval_dict[f'{eval_dest}/loss'] = total_loss / total_num
        eval_dict[f'{eval_dest}/top-1-acc'] = top1
        eval_dict[f'{eval_dest}/top-5-acc'] = top5
        eval_dict[f'{eval_dest}/balanced_acc'] = balanced_top1
        eval_dict[f'{eval_dest}/precision'] = precision
        eval_dict[f'{eval_dest}/recall'] = recall
        eval_dict[f'{eval_dest}/F1'] = F1

        # Unsupervised metrics
        if self.args.evaluate_unsupervised:
            self.print_fn("Performing unsupervised evaluation")
            feats = y_feats
            unsupervised_dict = unsupervised_scores(feats, y_logits, y_probs, scores=['rankme'])
            unsupervised_dict = {f'{eval_dest}/{k}': v for k, v in unsupervised_dict.items()}
            eval_dict.update(unsupervised_dict)
        
        # Additonal metrics
        if return_logits:
            eval_dict[eval_dest+'/logits'] = y_logits
        if return_y_true:
            eval_dict[eval_dest+'/y_true'] = y_true
        # if return_idx:
        #     eval_dict[eval_dest+'/idx'] = idx
        
        end = time.time()
        eval_dict[eval_dest+'/time'] = end - start
        return eval_dict

    def get_save_dict(self):
        """
        make easier for saving model when need save additional arguments
        """
        # base arguments for all models
        save_dict = {
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_scaler': self.loss_scaler.state_dict(),
            'it': self.it + 1,
            'epoch': self.epoch + 1,
            'best_it': self.best_it,
            'best_eval_acc': self.best_eval_acc,
            'best_it_es': self.best_it_es,
            'best_metric_es': self.best_metric_es,
            'early_stop': self.early_stop,
            'best_it_test': self.best_it_test,
            'best_metric_test': self.best_metric_test
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        return save_dict
    

    def save_model(self, save_name, save_path):
        """
        save model and specified parameters for resume
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_filename = os.path.join(save_path, save_name)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        self.print_fn(f"model saved: {save_filename}")


    def load_model(self, load_path):
        """
        load model and specified parameters for resume
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        self.it = checkpoint['it']
        self.start_epoch = checkpoint['epoch']
        self.epoch = self.start_epoch
        self.best_it = checkpoint['best_it']
        self.best_eval_acc = checkpoint['best_eval_acc']
        self.best_it_es = checkpoint['best_it_es']
        self.best_metric_es = checkpoint['best_metric_es']
        self.early_stop = checkpoint['early_stop']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.print_fn('Model loaded')
        return checkpoint
    
    def load_checkpoint(self, load_path):
        """
        only load checkpoint for model and optimizer (used for valid stage training)
        """
        # TODO: refactor this part
        # Load Checkpoint
        checkpoint = torch.load(load_path, map_location='cpu')
        state_dict = checkpoint['model']
        if 'backbone' in list(state_dict.keys())[0]:
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'rot_' not in k}
        self.load_state_dict(state_dict)
        self.print_fn('Checkpoint loaded')
        return checkpoint
    
    def load_state_dict(self, state_dict):
        # Load model state dict
        self.model.load_state_dict(state_dict)
        # Setup ema
        self.ema_model.load_state_dict(state_dict)
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()


    def check_prefix_state_dict(self, state_dict):
        """
        remove prefix state dict in ema model
        """
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    def register_hook(self, hook, name=None, priority='NORMAL'):
        """
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            hook_name (:str, default to None): Name of the hook to be registered. Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        
        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook
        


    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", uesed to call single hook in train_step.
        """
        
        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)
        
        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def registered_hook(self, hook_name):
        """
        Check if a hook is registered
        """
        return hook_name in self.hooks_dict


    @staticmethod
    def get_argument():
        """
        Get specificed arguments into argparse for each algorithm
        """
        return {}



class ImbAlgorithmBase(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        
        # imbalanced arguments
        self.lb_imb_ratio = self.args.lb_imb_ratio
        self.ulb_imb_ratio = self.args.ulb_imb_ratio
        self.imb_algorithm = self.args.imb_algorithm
    
    def imb_init(self, *args, **kwargs):
        """
        intiialize imbalanced algorithm parameters
        """
        pass 

    def set_optimizer(self):
        if 'vit' in self.args.net and self.args.dataset in ['cifar100', 'food101', 'semi_aves', 'semi_aves_out']:
            return super().set_optimizer() 
        elif self.args.dataset in ['imagenet', 'imagenet127']:
            return super().set_optimizer() 
        else:
            self.print_fn("Create optimizer and scheduler")
            optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, bn_wd_skip=False)
            scheduler = None
            return optimizer, scheduler
