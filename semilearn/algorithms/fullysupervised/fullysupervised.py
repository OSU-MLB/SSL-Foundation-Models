# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
import torch
from collections import defaultdict
import pprint
import time

@ALGORITHMS.register('fullysupervised')
class FullySupervised(AlgorithmBase):
    """
        Train a fully supervised model using labeled data only. This serves as a baseline for comparison.

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
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

    def train_step(self, x_lb, y_lb):
        # inference and calculate sup/unsup losses
        # assert y_lb.sum() != 0
        assert torch.sum(y_lb==-1) == 0
        with self.amp_cm():

            logits_x_lb = self.model(x_lb)['logits']
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')


        train_accu = torch.mean((torch.argmax(logits_x_lb, axis=1) == y_lb).float()).item()
        out_dict = self.process_out_dict(loss=sup_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), train_accu=train_accu)
        return out_dict, log_dict


    def train(self):
        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.call_hook("before_run")
        
        cls_n = defaultdict(int)
        idx_n = defaultdict(int)
        _t_start = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            t1 = time.time()
            self.epoch = epoch
            if self._check_stop():
                print("***************** Stopped")
                break
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break

            self.print_fn(f"Epoch {self.epoch}/{self.epochs}")

            self.call_hook("before_train_epoch")

            lb_iter = iter(self.loader_dict['train_lb'])

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
                for data_lb in lb_iter:
                    # prevent the training iterations exceed args.num_train_iter
                    if self._check_stop():
                        print("***************** Stopped")
                        break
                    _train_start = time.time()

                    y_lb = data_lb['y_lb']
                    idx_lb = data_lb['idx_lb']
                    n_data_lb = len(y_lb)
                    
                    for _y in y_lb:
                        cls_n[_y.item()] += 1
                    for _idx in idx_lb:
                        idx_n[_idx.item()] += 1

                    self.call_hook("before_train_step")
                    self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb))
                    self.n_data_lb = n_data_lb
                    self.call_hook("after_train_step")
                    self.it += 1
                    _train_end = time.time()
                    # print(f"Time taken for training step: {_train_end-_train_start}")
            finally:
                del lb_iter
            
            print(f"############ Epoch {self.epoch}:")

            self.call_hook("after_train_epoch")
            t9 = time.time()
            print(f"Time taken for epoch {self.epoch}: {t9-t1}")
        self.call_hook("after_run")
        print(f"Total time taken: {time.time()-_t_start}")


ALGORITHMS['supervised'] = FullySupervised