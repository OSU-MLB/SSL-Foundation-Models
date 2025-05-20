# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random
import numpy as np
import torch
from torch.utils.data import sampler, DataLoader
import torch.distributed as dist
from io import BytesIO

# TODO: better way
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def split_ssl_data(args, 
                   data, 
                   targets, 
                   num_classes,
                   lb_num_labels, 
                   ulb_num_labels=None,
                   lb_imbalance_ratio=1.0, 
                   ulb_imbalance_ratio=1.0,
                   lb_index=None, 
                   ulb_index=None, 
                   include_lb_to_ulb=True, 
                   data_dir=None, 
                   load_exist=True, 
                   save_format='npy',
                   return_idxs=False,
                   save_appendix=''):
    """
    data & target is splitted into labeled and unlabeled data.
    
    Args
        data: data to be split to labeled and unlabeled 
        targets: targets to be split to labeled and unlabeled 
        num_classes: number of total classes
        lb_num_labels: number of labeled samples. 
                       If lb_imbalance_ratio is 1.0, lb_num_labels denotes total number of samples.
                       Otherwise it denotes the number of samples in head class.
        ulb_num_labels: similar to lb_num_labels but for unlabeled data.
                        default to None, denoting use all remaining data except for labeled data as unlabeled set
        lb_imbalance_ratio: imbalance ratio for labeled data
        ulb_imbalance_ratio: imbalance ratio for unlabeled data
        lb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        ulb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeled data
    """
    data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, data, targets, 
                                                    num_classes, 
                                                    lb_num_labels, 
                                                    ulb_num_labels,
                                                    lb_imbalance_ratio, 
                                                    ulb_imbalance_ratio, 
                                                    data_dir=data_dir,
                                                    load_exist=True,
                                                    save_format=save_format,
                                                    appendix=save_appendix)
    
    # manually set lb_idx and ulb_idx, do not use except for debug
    if lb_index is not None:
        lb_idx = lb_index
    if ulb_index is not None:
        ulb_idx = ulb_index

    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
        ulb_lb_mask = np.zeros(len(ulb_idx), dtype=bool)
        ulb_lb_mask[:len(lb_idx)] = True
    else:
        ulb_lb_mask = np.zeros(len(ulb_idx), dtype=bool)
    if return_idxs:
        return lb_idx, data[lb_idx], targets[lb_idx], \
               ulb_idx, data[ulb_idx], targets[ulb_idx], ulb_lb_mask
    else:
        return data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]


def sample_labeled_data():
    pass


def _get_npy_idx_handler(lb_dump_path, ulb_dump_path):
    def load_idx():
        print(f"Loading from {lb_dump_path} and {ulb_dump_path}")
        lb_idx = None
        ulb_idx = None
        if os.path.exists(lb_dump_path):
            lb_idx = np.load(lb_dump_path)
        if os.path.exists(ulb_dump_path):
            ulb_idx = np.load(ulb_dump_path)
        return lb_idx, ulb_idx
    def save_idx(lb_idx, ulb_idx):
        print(f"Saving to {lb_dump_path} and {ulb_dump_path}")
        np.save(lb_dump_path, lb_idx)
        np.save(ulb_dump_path, ulb_idx)
    return load_idx, save_idx


def _get_list_idx_handler(data, target, lb_dump_path, ulb_dump_path):
    def load_idx():
        print(f"Loading from {lb_dump_path} and {ulb_dump_path}")
        lb_idx = None
        ulb_idx = None
        if os.path.exists(lb_dump_path):
            data = np.loadtxt(lb_dump_path, 
                              dtype={'names': ('filename', 'label', 'idx'), 
                                     'formats': ('U100', 'i4', 'i4')})
            lb_idx = data['idx']
        if os.path.exists(ulb_dump_path):
            data = np.loadtxt(ulb_dump_path, 
                              dtype={'names': ('filename', 'label', 'idx'), 
                                     'formats': ('U100', 'i4', 'i4')})
            ulb_idx = data['idx']
        return lb_idx, ulb_idx
    def save_idx(lb_idx, ulb_idx):
        print(f"Saving to {lb_dump_path} and {ulb_dump_path}")
        with open(lb_dump_path, 'w') as f:
            for idx in lb_idx:
                f.write(f"{data[idx]} {target[idx]} {idx}\n")
        with open(ulb_dump_path, 'w') as f:
            for idx in ulb_idx:
                f.write(f"{data[idx]} {target[idx]} {idx}\n")
    return load_idx, save_idx


def _get_idx_handler(dump_dir, data, target, num_labels, lb_imb_ratio, ulb_imb_ratio, seed, save_format, appendix):
    if save_format == 'npy':
        lb_dump_path = os.path.join(dump_dir, 
                                    f'lb_labels{num_labels}_{lb_imb_ratio}_seed{seed}_{appendix}idx.npy')
        ulb_dump_path = os.path.join(dump_dir, 
                                    f'ulb_labels{num_labels}_{ulb_imb_ratio}_seed{seed}_{appendix}idx.npy')
        load_idx, save_idx = _get_npy_idx_handler(lb_dump_path, ulb_dump_path)
    elif save_format == 'list':
        lb_dump_path = os.path.join(dump_dir, 
                                    f'lb_labels{num_labels}_{lb_imb_ratio}_seed{seed}_{appendix}idx.list')
        ulb_dump_path = os.path.join(dump_dir, 
                                     f'ulb_labels{num_labels}_{ulb_imb_ratio}_seed{seed}_{appendix}idx.list')
        load_idx, save_idx = _get_list_idx_handler(data, target, lb_dump_path, ulb_dump_path)
    return load_idx, save_idx


def sample_labeled_unlabeled_data(args, 
                                  data, 
                                  target, 
                                  num_classes,
                                  lb_num_labels, 
                                  ulb_num_labels=None,
                                  lb_imbalance_ratio=1.0, 
                                  ulb_imbalance_ratio=1.0, 
                                  data_dir=None, 
                                  load_exist=True, 
                                  save_format='npy',
                                  appendix=''):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    # Unpack args
    dataset = args.dataset
    num_labels = args.num_labels
    seed = args.seed

    # Check save_format
    assert save_format in ['npy', 'list'], "save_format must be either 'npy' or 'list'"

    # Set data_dir
    if data_dir is None:
        data_dir = os.path.join(base_dir, 'data', dataset)
    dump_dir = os.path.join(data_dir, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)

    # Get idx handler
    load_idx, save_idx = _get_idx_handler(dump_dir, data, target, num_labels, lb_imbalance_ratio, ulb_imbalance_ratio, seed, save_format, appendix)

    # Load idx if exists
    lb_idx, ulb_idx = load_idx()
    if load_exist and lb_idx is not None and ulb_idx is not None:
        return lb_idx, ulb_idx
    # assert False, "idx file not found!!!!!!!!!!!!!"

    # Get samples per class
    if lb_imbalance_ratio == 1.0:
        # balanced setting, lb_num_labels is total number of labels for labeled data
        assert lb_num_labels % num_classes == 0, "lb_num_labels must be dividable by num_classes in balanced setting"
        lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes
    else:
        # imbalanced setting, lb_num_labels is the maximum number of labels for class 1
        lb_samples_per_class = make_imbalance_data(lb_num_labels, num_classes, lb_imbalance_ratio)

    if ulb_imbalance_ratio == 1.0:
        # balanced setting
        if ulb_num_labels is None or ulb_num_labels == 'None':
            pass # ulb_samples_per_class = [int(len(data) / num_classes) - lb_samples_per_class[c] for c in range(num_classes)] # [int(len(data) / num_classes) - int(lb_num_labels / num_classes)] * num_classes
        else:
            assert ulb_num_labels % num_classes == 0, "ulb_num_labels must be dividable by num_classes in balanced setting"
            ulb_samples_per_class = [int(ulb_num_labels / num_classes)] * num_classes
    else:
        # imbalanced setting
        assert ulb_num_labels is not None, "ulb_num_labels must be set set in imbalanced setting"
        ulb_samples_per_class = make_imbalance_data(ulb_num_labels, num_classes, ulb_imbalance_ratio)

    # sample labeled and unlabeled data
    lb_idx = []
    ulb_idx = []
    
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])
        if ulb_num_labels is None or ulb_num_labels == 'None':
            ulb_idx.extend(idx[lb_samples_per_class[c]:])
        else:
            ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c]+ulb_samples_per_class[c]])
    
    if isinstance(lb_idx, list):
        lb_idx = np.asarray(lb_idx)
    if isinstance(ulb_idx, list):
        ulb_idx = np.asarray(ulb_idx)

    # Save idx
    save_idx(lb_idx, ulb_idx)
    
    return lb_idx, ulb_idx


def make_imbalance_data(max_num_labels, num_classes, gamma):
    """
    calculate samplers per class for imbalanced data
    """
    mu = np.power(1 / abs(gamma), 1 / (num_classes - 1))
    samples_per_class = []
    for c in range(num_classes):
        if c == (num_classes - 1):
            samples_per_class.append(int(max_num_labels / abs(gamma)))
        else:
            samples_per_class.append(int(max_num_labels * np.power(mu, c)))
    if gamma < 0:
        samples_per_class = samples_per_class[::-1]
    return samples_per_class


def get_collactor(args, net):
    if net == 'bert_base_uncased':
        from semilearn.datasets.collactors import get_bert_base_uncased_collactor
        collact_fn = get_bert_base_uncased_collactor(args.max_length)
    elif net == 'bert_base_cased':
        from semilearn.datasets.collactors import get_bert_base_cased_collactor
        collact_fn = get_bert_base_cased_collactor(args.max_length)
    elif net == 'wave2vecv2_base':
        from semilearn.datasets.collactors import get_wave2vecv2_base_collactor
        collact_fn = get_wave2vecv2_base_collactor(args.max_length_seconds, args.sample_rate)
    elif net == 'hubert_base':
        from semilearn.datasets.collactors import get_hubert_base_collactor
        collact_fn = get_hubert_base_collactor(args.max_length_seconds, args.sample_rate)
    else:
        collact_fn = None
    return collact_fn



def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = random.randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]