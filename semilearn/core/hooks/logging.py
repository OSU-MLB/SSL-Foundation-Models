# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref:https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/logger/base.py

import copy
from .hook import Hook

def is_integer(n):
    return n == int(n)

class LoggingHook(Hook):
    """
    Logging Hook for print information and log into tensorboard
    """

    def after_train_step(self, algorithm):
        """must be called after evaluation"""

        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            if not algorithm.distributed or (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                log_dict = algorithm.log_dict
                
                print_text = f"{algorithm.it + 1} iteration, USE_EMA: {algorithm.ema_m != 0}, "
                for i, (key, item) in enumerate(log_dict.items()):
                    print_text += "{:s}: {:.4f}".format(key, item)
                    if i != len(log_dict) - 1:
                        print_text += ", "
                    else:
                        print_text += " "

                
                print_text += "BEST_EVAL_ACC: {:.4f}, at {:d} iters".format(algorithm.best_eval_acc, algorithm.best_it + 1)
                # algorithm.print_fn(f"{algorithm.it + 1} iteration, USE_EMA: {algorithm.ema_m != 0}, {algorithm.log_dict}, BEST_EVAL_ACC: {algorithm.best_eval_acc}, at {algorithm.best_it + 1} iters")
                print_text += "\n BEST_METRIC_ES: {:.4f}, at {:d} iters".format(algorithm.best_metric_es, algorithm.best_it_es + 1)
                algorithm.print_fn(print_text)
            
                if not algorithm.tb_log is None:
                    print('Updating tensorboard log...')
                    algorithm.tb_log.update(log_dict, algorithm.it)
                


        
        elif self.every_n_iters(algorithm, algorithm.num_log_iter):
            if not algorithm.distributed or (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                print_text = f"{algorithm.it + 1} iteration USE_EMA: {algorithm.ema_m != 0}, "
                for i, (key, item) in enumerate(algorithm.log_dict.items()):
                    print_text += "{:s}: {:.4f}".format(key, item)
                    if i != len(algorithm.log_dict) - 1:
                        print_text += ", "
                    else:
                        print_text += " "
                algorithm.print_fn(print_text)
        
        # Every iter metrics
        if not algorithm.distributed or (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                log_dict = algorithm.log_dict
                if 'train/train_accu' in log_dict:
                    # print(f"Updating tensorboard log for train_accu: {log_dict['train/train_accu']} at {algorithm.it}")
                    algorithm.tb_log.update({'train_accu': log_dict['train/train_accu']}, algorithm.it)

                
                # PET
                if algorithm.algorithm == 'pet':
                    sup_loss = algorithm.log_dict['train/sup_loss']
                    sup_loss_w = algorithm.log_dict['train/sup_loss_w']
                    sup_loss_s = algorithm.log_dict['train/sup_loss_s']
                    sup_loss_w_kd = algorithm.log_dict['train/sup_loss_w_kd']
                    sup_loss_s_kd = algorithm.log_dict['train/sup_loss_s_kd']
                    acc_pl = algorithm.log_dict['train/acc_pl']
                    acc_gt = algorithm.log_dict['train/acc_gt']
                    algorithm.tb_log.update({
                        'sup_loss': sup_loss,
                        'sup_loss_w': sup_loss_w,
                        'sup_loss_s': sup_loss_s,
                        'sup_loss_w_kd': sup_loss_w_kd,
                        'sup_loss_s_kd': sup_loss_s_kd,
                        'acc_pl': acc_pl,
                        'acc_gt': acc_gt
                    }, algorithm.it)