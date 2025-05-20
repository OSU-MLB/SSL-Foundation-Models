# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook
from ..utils.misc import AverageMeter


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    """
    
    def __init__(self):
        super(EvaluationHook, self).__init__()
        self.avg_meters = {
            # train
            ('train/sup_loss'): AverageMeter('train/sup_loss'),
            ('train/unsup_loss'): AverageMeter('train/unsup_loss'),
            ('train/unsup_loss_overused'): AverageMeter('train/unsup_loss_overused'),
            ('train/total_loss'): AverageMeter('train/total_loss'),
            ('train/total_loss_overused'): AverageMeter('train/total_loss_overused'),
            ('train/util_ratio'): AverageMeter('train/util_ratio'),
        }

    def _save_avg_meters(self, algorithm):
        # print('Saving average meter...')

        log_dict = algorithm.log_dict
        sup_loss = algorithm.log_dict['train/sup_loss']

        n_data_lb = algorithm.n_data_lb

        self.avg_meters[('train/sup_loss')].update(sup_loss, n_data_lb)

        # Only Unsupervised method
        if 'train/unsup_loss' in algorithm.log_dict:
            unsup_loss = algorithm.log_dict['train/unsup_loss']
            n_data_ulb = algorithm.n_data_ulb
            total_loss = algorithm.log_dict['train/total_loss']
            n_data = n_data_lb + n_data_ulb

            self.avg_meters[('train/unsup_loss')].update(unsup_loss, n_data_ulb)
            self.avg_meters[('train/total_loss')].update(total_loss, n_data)

            # Only Unsupervised method but doesn't have util_ratio
            if 'train/util_ratio' in algorithm.log_dict:
                util_ratio = algorithm.log_dict['train/util_ratio']
                n_data_ulb_used = n_data_ulb*util_ratio
                n_data_used = n_data_lb + n_data_ulb_used
                self.avg_meters[('train/util_ratio')].update(util_ratio, 1)
                self.avg_meters[('train/unsup_loss_overused')].update(unsup_loss, n_data_ulb_used)
                self.avg_meters[('train/total_loss_overused')].update(total_loss, n_data_used)
        
    def _check_early_stopping(self, algorithm):
        # Perform early stopping
        print('Performing early stopping...')

        es_metric_val = algorithm.log_dict[algorithm.es_criteria]
        print(f"Current es metric: {es_metric_val}, best es metric: {algorithm.best_metric_es}")
        if algorithm.best_metric_es == None or es_metric_val < algorithm.best_metric_es:
            print(f"***************** Updating best metric to {es_metric_val} at iteration {algorithm.it + 1} from {algorithm.best_metric_es} at iteration {algorithm.best_it_es + 1}")
            algorithm.best_it_es = algorithm.it
            algorithm.best_metric_es = es_metric_val
        print(f"Counter since best metric: {algorithm.it - algorithm.best_it_es}, patience: {algorithm.es_patience_iters}")
        if algorithm.es_patience_iters <= 0:
            print('"***************** Early stopping disabled')
        if algorithm.es_patience_iters > 0 and (algorithm.it - algorithm.best_it_es > algorithm.es_patience_iters):
            print(f"***************** Early stopping at iteration {algorithm.it + 1}, best metric: {algorithm.best_metric_es} at iteration {algorithm.best_it_es + 1}")
            algorithm.early_stop = True

    def _dump_avg_meters(self):
        out_results = {}
        for key, avg_meter in self.avg_meters.items():
            out_results[f"{key}_avg"] = avg_meter.avg
            avg_meter.reset()
        return out_results
    
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("Validating...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)

            # update best metrics
            if algorithm.log_dict['eval/top-1-acc'] > algorithm.best_eval_acc:
                algorithm.best_eval_acc = algorithm.log_dict['eval/top-1-acc']
                algorithm.best_it = algorithm.it
            
            if algorithm.args.eval_on_test:
                algorithm.print_fn("Evaluating on test set...")
                test_dict = algorithm.evaluate('test')
                algorithm.log_dict.update(test_dict)
            
                if algorithm.best_metric_test is None or test_dict['test/top-1-acc'] > algorithm.best_metric_test:
                    print(f"################# Updating best test metric to {test_dict['test/top-1-acc']} at iteration {algorithm.it + 1} from {algorithm.best_metric_test} at iteration {algorithm.best_it_test + 1}")
                    algorithm.best_metric_test = test_dict['test/top-1-acc']
                    algorithm.best_it_test = algorithm.it
            
            # Dump average meters
            avg_meters = self._dump_avg_meters()
            algorithm.log_dict.update(avg_meters)
            
            self._check_early_stopping(algorithm)

        self._save_avg_meters(algorithm)
    
    def after_run(self, algorithm):
        
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            if algorithm.args.no_save:
                algorithm.print_fn("Skipping saving model...")
            else:
                save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
                algorithm.save_model('latest_model.pth', save_path)
        # early stopping related parameters
        results_dict = {'eval/best_it_acc': algorithm.best_eval_acc, 
                        'eval/best_it': algorithm.best_it, 
                        'es/best_it': algorithm.best_it_es, 
                        'es/best_it_metric': algorithm.best_metric_es, 
                        'test/best_it': algorithm.best_it_test, 
                        'test/best_it_acc': algorithm.best_metric_test}
        algorithm.results_dict = results_dict
