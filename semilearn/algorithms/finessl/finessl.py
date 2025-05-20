# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from .finessl_hook import FineSSLHook
from itertools import cycle

import os
import time
import datetime
import numpy as np
from torch.nn import functional 
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torchvision import transforms



from itertools import cycle

import itertools
from copy import deepcopy

from collections import Counter

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# Huy: implement the FineSSL algorithm
@ALGORITHMS.register('finessl')
class FineSSL(AlgorithmBase):

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(alpha = args.alpha, w_con = args.w_con, th = args.th, smoothing = args.smoothing)
    
    def init(self, alpha, w_con, th, smoothing):
        
        # self.classnames = self.loader_dict['train_lb'].dataset.classnames

        
        self.alpha = alpha
        self.betabase = torch.ones((self.num_classes, self.num_classes)).to("cuda")
        self.betabase[torch.arange(self.num_classes), torch.arange(self.num_classes)] = 0.0
        

        self.smoothing = smoothing
        self.th = th
        self.w_con = w_con
        
        self.ulab_len = len(self.loader_dict['train_ulb'].dataset)
        
        
    def train(self):
        """
        train function
        """

        self.call_hook("before_run")

        self.print_fn(f"Start training, start epoch: {self.start_epoch}, total epochs: {self.epochs}")

        t_start = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            t1 = time.time()
            
            self.epoch = epoch
            self.model.train()
           
            if self._check_stop():
                self.print_fn("***************** Stopped")
                break

            self.print_fn(f"Epoch {self.epoch}/{self.epochs}")
            
            self.call_hook("before_train_epoch")

            lb_iter = iter(self.loader_dict['train_lb'])
            ulb_iter = iter(self.loader_dict['train_ulb'])
            
            self.classwise_acc =torch.zeros((self.num_classes,))
            self.selected_label = torch.ones((self.ulab_len,), dtype=torch.long, ) * -1
            self.selected_label = self.selected_label.to("cuda")
        
            for data_lb, data_ulb in zip(lb_iter, ulb_iter):
                if self._check_stop():
                    print("***************** Stopped")
                    break

                self.n_data_lb = len(data_lb['y_lb'])
                self.n_data_ulb = len(data_ulb['x_ulb_w'])

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1
            
            t7 = time.time()
            print(f"Time for epoch {epoch}: {t7 - t1}")
            
            self.call_hook("after_train_epoch")
        t_end = time.time()
        print(f"Time taken for training: {t_end - t_start}")

        self.call_hook("after_run")
        
    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
                
        batch_size = x_lb.shape[0]
        y_lb = y_lb.to(torch.long)        
        pseudo_counter = Counter(self.selected_label.tolist())
    
        if max(pseudo_counter.values()) < self.ulab_len:
            wo_negative_one = deepcopy(pseudo_counter)
            if -1 in wo_negative_one.keys():
                wo_negative_one.pop(-1)
            for i in range(self.num_classes):
                self.classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())
       
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_w)) 
        
        output = self.model(inputs)
        output_x_lb = output['logits'][:batch_size]
        output_u_w, output_u_s = output['logits'][batch_size:].chunk(2)
        
        feats_x_lb = output['feat'][:batch_size]
        feats_u_w, feats_u_s = output['feat'][batch_size:].chunk(2)
        feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_u_w, 'x_ulb_s':feats_u_s}
    
        mar_diff = (self.classwise_acc.max() - self.classwise_acc.min()) * self.alpha
        alpha = (((1.0 - self.classwise_acc) / (self.classwise_acc + 1.0)) * mar_diff).unsqueeze(1).to("cuda")
    
        output_x_lb = output_x_lb + alpha[y_lb] * self.betabase[y_lb]
    
        Lx = F.cross_entropy(output_x_lb, y_lb, reduction='mean')
    
        pseu = torch.softmax(output_u_w.detach(), dim=-1)
        conf, targets_u = torch.max(pseu, dim=-1)
        mask = conf.ge(self.th)
    
        output_u_s = output_u_s + alpha[targets_u] * self.betabase[targets_u]
    
        if torch.sum(mask)> 0:
            Lu = (self.ce_loss(output_u_s, targets_u,
                               reduction='none') * conf * mask).mean()
        else:  
            Lu = 0

        select = conf.ge(0.7).long()
        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[idx_ulb[select == 1]] = targets_u[select == 1]
            
        # print("Lx: ", Lx)
        # print("Lu: ", Lu)
    
        total_loss = Lx + Lu * self.w_con

        
        train_accu = torch.mean((torch.argmax(output_x_lb, axis=1) == y_lb).float()).item()
        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=Lx.item(), 
                                        unsup_loss=Lu, 
                                        total_loss=total_loss.item(), 
                                        util_ratio=mask.float().mean().item(),
                                        train_accu=train_accu)
        
        
        return out_dict, log_dict
