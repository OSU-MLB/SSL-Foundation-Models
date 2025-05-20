# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from .pet_hook import PETHook

import copy
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
import torch
import pprint
import time

from torch.nn import functional as F

@ALGORITHMS.register('pet')
class PET(AlgorithmBase):
    """
        Pseudo Label algorithm (https://arxiv.org/abs/1908.02983).

        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
        - unsup_warm_up (`float`, *optional*, defaults to 0.4):
            Ramp up for weights for unsupervised loss
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.pl = {}
        self.n_rounds = args.n_rounds
        self.conf_threshold = args.conf_threshold
        self.conf_ratio = args.conf_ratio
        self.w_alpha = args.w_alpha
        self.s_alpha = args.s_alpha
        self.kd_w_alpha = args.kd_w_alpha
        self.kd_s_alpha = args.kd_s_alpha
        self.temperature = args.temperature
        self.pretrained_path = args.pretrained_path

        self._model = copy.deepcopy(self.model)
        self._ema_model = copy.deepcopy(self.ema_model)

    def reset_model(self):
        assert self._model is not None, "Model is not initialized"
        assert self._ema_model is not None, "EMA Model is not initialized"
        self.print_fn("Resetting model and EMA model")
        # Load the initial model state
        _model_state_dict = self._model.state_dict()
        if 'module' in list(self.model.state_dict().keys())[0]:
            _model_state_dict = {f'module.{k}': v for k, v in _model_state_dict.items()}
        self.model.load_state_dict(_model_state_dict)
        # Load the initial EMA model state
        _ema_model_state_dict = self._ema_model.state_dict()
        if 'module' in list(self.ema_model.state_dict().keys())[0]:
            _ema_model_state_dict = {f'module.{k}': v for k, v in _ema_model_state_dict.items()}
        self.ema_model.load_state_dict(_ema_model_state_dict)

    def reset_optim(self):
        self.print_fn("Resetting optimizer and scheduler")
        self.optimizer, self.scheduler = self.set_optimizer()

    def set_hooks(self):
        self.register_hook(PETHook(), "PETHook")
        super().set_hooks()

    def train_step(self, idx_ulb, x_ulb, x_ulb_w, x_ulb_s, pl_ulb, distill_scores_ulb, y_true_ulb):
        # import pdb; pdb.set_trace()
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.args.w_alpha > 0 or self.args.kd_w_alpha > 0:
                logits_x_ulb_w = self.model(x_ulb_w)['logits']
            if self.args.s_alpha > 0 or self.args.kd_s_alpha > 0:
                logits_x_ulb_s = self.model(x_ulb_s)['logits']

            if self.args.w_alpha > 0:
                sup_loss_w = self.ce_loss(logits_x_ulb_w, pl_ulb, reduction='mean')
            else:
                sup_loss_w = torch.tensor(0.0).cuda(self.gpu)
            if self.args.s_alpha > 0:
                sup_loss_s = self.ce_loss(logits_x_ulb_s, pl_ulb, reduction='mean')
            else:
                sup_loss_s = torch.tensor(0.0).cuda(self.gpu)

            if self.args.kd_w_alpha > 0:
                log_probs_x_ulb_w = F.log_softmax(logits_x_ulb_w / self.temperature, dim=1)
                sup_loss_w_kd = F.kl_div(log_probs_x_ulb_w, distill_scores_ulb, reduction='batchmean')
            else:
                sup_loss_w_kd = torch.tensor(0.0).cuda(self.gpu)

            if self.args.kd_s_alpha > 0:
                log_probs_x_ulb_s = F.log_softmax(logits_x_ulb_s / self.temperature, dim=1)
                sup_loss_s_kd = F.kl_div(log_probs_x_ulb_s, distill_scores_ulb, reduction='batchmean')
            else:
                sup_loss_s_kd = torch.tensor(0.0).cuda(self.gpu)

            with torch.no_grad():                
                logits_x_ulb = self.model(x_ulb)['logits']
            acc_pl = torch.mean((torch.argmax(logits_x_ulb, axis=1) == pl_ulb).float()).item()
            acc_gt = torch.mean((torch.argmax(logits_x_ulb, axis=1) == y_true_ulb).float()).item()

            loss = (self.w_alpha * sup_loss_w) + (self.s_alpha * sup_loss_s) + (self.kd_w_alpha * sup_loss_w_kd) + (self.kd_s_alpha * sup_loss_s_kd)

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(
            sup_loss=sup_loss_w.item() + sup_loss_s.item(),
            sup_loss_w=sup_loss_w.item(),
            sup_loss_s=sup_loss_s.item(),
            sup_loss_w_kd=sup_loss_w_kd.item(),
            sup_loss_s_kd=sup_loss_s_kd.item(),
            acc_pl=acc_pl,
            acc_gt=acc_gt
        )
        return out_dict, log_dict


    def _check_stop(self):
        if self.it >= self.num_train_iter*self.n_rounds:
            self.print_fn("***************** Iteration limit reached")
            return True
        return False


    def train(self):
        if self.args.pretrained_path != '':
            self.print_fn(f"Using pretrained model from {self.args.pretrained_path}, performing initial evaluation")
            eval_dict = self.evaluate('eval')
            self.print_fn(f"Initial eval: {pprint.pformat(eval_dict)}")

        self.model.train()
        self.call_hook("before_run")
        
        _t_start = time.time()
        for i_round in range(self.n_rounds):
            self.print_fn(f"Round {i_round+1}/{self.n_rounds}")
            for epoch in range(self.start_epoch, self.epochs):
                t1 = time.time()
                self.epoch = epoch
                if self._check_stop():
                    print("***************** Stopped")
                    break


                self.print_fn(f"Epoch {self.epoch}/{self.epochs}")

                self.call_hook("before_train_epoch")
                ulb_iter = iter(self.loader_dict['train_ulb'])

                pl = self.pl[i_round]['pl']
                y_true = self.pl[i_round]['y_true']
                distill_scores = self.pl[i_round]['distill_scores']
                pl = torch.tensor(pl).cuda(self.gpu)
                y_true = torch.tensor(y_true).cuda(self.gpu)
                distill_scores = torch.tensor(distill_scores).cuda(self.gpu)


                try:
                    for data_ulb in ulb_iter:
                        # prevent the training iterations exceed args.num_train_iter
                        if self._check_stop():
                            print("***************** Stopped")
                            break

                        idx_ulb = data_ulb['idx_ulb']

                        pl_ulb = pl[idx_ulb]
                        distill_scores_ulb = distill_scores[idx_ulb]
                        y_true_ulb = y_true[idx_ulb]

                        _data_ulb = {**data_ulb, 
                                    'pl_ulb': pl_ulb,
                                    'distill_scores_ulb': distill_scores_ulb, 'y_true_ulb': y_true_ulb}
                    
                        self.call_hook("before_train_step")

                        # import pdb; pdb.set_trace()
                        self.out_dict, self.log_dict = self.train_step(**self.process_batch(**_data_ulb))

                        self.n_data_lb = 0
                        self.n_data_ulb = len(idx_ulb)
                        self.call_hook("after_train_step")
                        self.it += 1
                finally:
                    pass
                
                print(f"############ Epoch {self.epoch}:")

                self.call_hook("after_train_epoch")
                t9 = time.time()
                self.print_fn(f"Time taken for epoch {self.epoch}: {t9-t1}")
            if i_round < self.n_rounds - 1:
                self.hooks_dict['PETHook'].update_pl(self, i_round)
                self.reset_model()
                self.reset_optim()
                self.start_epoch = 0
        self.call_hook("after_run")
        self.print_fn(f"Total time taken: {time.time()-_t_start}")
