# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py

import os

from .hook import Hook

class CheckpointHook(Hook):
    """
    Checkpoint Hook for saving checkpoint
    """
    def after_train_step(self, algorithm):
        # must be called after evaluation for saving the best
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            
            if (not algorithm.distributed) or \
               (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                algorithm.save_model('latest_model.pth', save_path)

                # if algorithm.it == algorithm.best_it:
                #     algorithm.save_model('model_best.pth', save_path)
                
                print(f'In CheckpointHook, algorithm.best_it_es: {algorithm.best_it_es}, algorithm.it: {algorithm.it}')
                # if algorithm.best_metric_es is not None and algorithm.it == algorithm.best_it_es:
                #     print(f'***************** Saving model_best_es.pth in {save_path}/model_best_es.pth')
                #     algorithm.save_model('model_best_es.pth', save_path)

                # if algorithm.best_metric_test is not None and algorithm.it == algorithm.best_it_test:
                #     print(f'################# Saving model_best_test.pth in {save_path}/model_best_test.pth')
                #     algorithm.save_model('model_best_test.pth', save_path)
        