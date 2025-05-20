import torch

import argparse
import copy
from dotwiz import DotWiz
from ruamel.yaml import YAML
import os

import time
import numpy as np
from semilearn.core.hooks import Hook
from semilearn.core.utils import VTAB_DSETS, get_net_builder, get_peft_config
from semilearn.algorithms.utils import smooth_targets
from semilearn.datasets.cv_datasets.vtab import get_vtab
import os
from torch.utils.data import DataLoader
import pickle
from scipy.special import softmax
from semilearn.core.utils.metrics import unsupervised_scores

_ROOT_DIR = os.path.abspath(os.curdir)
_yaml = YAML()

if __name__ == '__main__':

    eval_list = []

    _CONFIG_ROOT = './config/'
    for root, dirs, files in os.walk(_CONFIG_ROOT):
        if not 'config.yaml' in files:
            continue
        _config_path = os.path.join(root, 'config.yaml')
        with open(_config_path, 'r') as f:
            _config = _yaml.load(f)
        _load_path = _config['load_path']
        if os.path.exists(_load_path):
            eval_list.append((_load_path, _config))
    
    with open('eval_list.pkl', 'wb') as f:
        pickle.dump(eval_list, f)
    print(f'Saved {len(eval_list)} models to extract_pl_list.pkl')
