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
from tqdm import tqdm
from scipy.special import softmax
from semilearn.core.utils.metrics import unsupervised_scores

_ROOT_DIR = os.path.abspath(os.curdir)
_yaml = YAML()


def _eval(model, loader, eval_unsup=False):
    model.eval()
    acc = 0.0
    dset_len = len(loader.dataset)
    
    y_feats = []
    y_logits = []
    y_pred = []
    y_probs = []
    y_labels = []
    
    _n_data_processed = 0
    dset = loader.dataset
    print(f'len(dset): {len(dset)}')
    print(f'transform: {dset.transform}')
    n_processed = 0

    with torch.no_grad():
        # for data in loader:
        for data in tqdm(loader):
            image = data['x']
            target = data['y']

            _n_data_processed += len(image)

            image = image.type(torch.FloatTensor).cuda()
            
            feat = model(image, only_feat=True)
            logit = model(feat, only_fc=True)
            prob = logit.softmax(dim=-1)
            pred = prob.argmax(1)

            acc += pred.cpu().eq(target).sum().item()

            y_feats.append(feat.cpu())
            y_logits.append(logit.cpu())
            y_pred.append(pred.cpu())
            y_probs.append(prob.cpu())
            y_labels.append(target.cpu())

            n_processed += len(image)
    
    y_feats  = torch.cat(y_feats,  dim=0)
    y_logits = torch.cat(y_logits, dim=0)
    y_pred  = torch.cat(y_pred,  dim=0)
    y_probs  = torch.cat(y_probs,  dim=0)
    y_labels = torch.cat(y_labels, dim=0)

    acc = acc / dset_len
    assert n_processed == dset_len, f"n_processed: {n_processed}, dset_len: {dset_len}"
    
    if eval_unsup:
        eval_dict = unsupervised_scores(y_feats, y_logits, y_probs)
        eval_dict['acc'] = acc
    else:
        eval_dict = {'acc': acc}
    return eval_dict, y_feats, y_logits, y_pred, y_probs, y_labels


def _get_vtab(ulb_num_labels, lb_imb_ratio, ulb_imb_ratio, net, train_split, crop_ratio, img_size, alg, dset_name, num_labels, num_classes, data_dir, include_lb_to_ulb, seed, train_aug):
    args = {
        'seed': seed,
        'num_labels': num_labels,
        'dataset': dset_name,
        'ulb_num_labels': ulb_num_labels,
        'lb_imb_ratio': lb_imb_ratio,
        'ulb_imb_ratio': ulb_imb_ratio,
        'net': net,
        'train_split': train_split,
        'crop_ratio': crop_ratio,
        'img_size': img_size,
        'train_aug': train_aug
    }
    args = DotWiz(args)
    train_lb, train_ulb, val, test, ulb_lb_mask = get_vtab(args, alg, dset_name, num_labels, num_classes, data_dir, include_lb_to_ulb)
    return train_lb, train_ulb, val, test, ulb_lb_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device

    print('Loading extract_pl_list.pkl')
    with open('eval_list.pkl', 'rb') as f:
        models_to_extract_tot = pickle.load(f)

    models_to_extract = models_to_extract_tot

    print(f"Total {len(models_to_extract)} models to extract")

    extract_start = time.time()
    for i, (load_path, config) in enumerate(models_to_extract):

        print(f'##################### Evaluating {load_path}, progress: {i+1}/{len(models_to_extract)}, time elapsed: {time.time() - extract_start:.2f}s')

        if 'data_dir' in config:
            data_dir = config['data_dir']
        else:
            data_dir = './data'
        net = config['net']
        train_split = config['train_split']
        crop_ratio = config['crop_ratio']
        img_size = config['img_size']
        dataset = config['dataset']
        num_labels = config['num_labels']
        num_classes = config['num_classes']
        save_dir = config['save_dir']
        save_name = config['save_name']
        num_classes = config['num_classes']
        use_pretrain = config['use_pretrain']
        net_from_name = config['net_from_name']
        seed = config['seed']
        if 'peft_config' in config:
            _peft_config = config['peft_config']
        else:
            _peft_config = None
        peft_config = get_peft_config(_peft_config)
        if 'vit_config' in config:
            vit_config = config['vit_config']
        else:
            vit_config = {}
        if 'ulb_num_labels' in config:
            ulb_num_labels = config['ulb_num_labels']
        else:
            ulb_num_labels = None
        if 'lb_imb_ratio' in config:
            lb_imb_ratio = config['lb_imb_ratio']
        else:
            lb_imb_ratio = 1
        if 'ulb_imb_ratio' in config:
            ulb_imb_ratio = config['ulb_imb_ratio']
        else:
            ulb_imb_ratio = 1
        if 'pretrain_path' in config:
            pretrain_path = config['pretrain_path']
        else:
            pretrain_path = None
        train_aug = 'weak'

        # Sanity check
        log_path = os.path.join(_ROOT_DIR, save_dir, save_name)
        print(f'log_path: {log_path}')
        
        assert dataset in VTAB_DSETS, f"Currently only supports VTAB datasets. "

        pl_path = os.path.join(log_path, 'pl.pkl')
        val_pl_path = os.path.join(log_path, 'val_pl.pkl')
        bootstrapping_pl_path = os.path.join(log_path, 'bootstrapping_pl')

        os.makedirs(bootstrapping_pl_path, exist_ok=True)

        train_lb, train_ulb, val, test, ulb_lb_mask = _get_vtab(ulb_num_labels, lb_imb_ratio, ulb_imb_ratio, net, train_split, crop_ratio, img_size, 'extract_pl', dataset, num_labels, num_classes, data_dir, True, seed, train_aug)

        train_ulb_noaug = copy.copy(train_ulb)
        train_ulb_strongaug = copy.copy(train_ulb)
        val_transform = val.transform
        strong_transform = train_ulb.strong_transform
        train_ulb_noaug.transform = val_transform
        train_ulb_strongaug.transform = strong_transform

        # Generate pseudo labels
        print(f'Generating pseudo labels')
        
        net_builder = get_net_builder(net, net_from_name, peft_config=peft_config, vit_config=vit_config)
        model = net_builder(num_classes=num_classes, pretrained=use_pretrain, pretrained_path=pretrain_path)
        load_path = os.path.join(log_path, 'latest_model.pth')
        

        # Load model
        print(f'Loading model from {load_path}')
        checkpoint = torch.load(load_path, map_location='cpu')
        _state_dict = checkpoint['model']
        state_dict = {}
        ######## Temporary fix ########
        for k, v in _state_dict.items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        if 'backbone' in list(state_dict.keys())[0]:
                state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'rot_' not in k}
        ###############################
        model.load_state_dict(state_dict)

        # Inference
        model = model.to(device)
        model.eval()

        # Loaders
        train_ulb_loader = DataLoader(train_ulb_noaug, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)
        print(f'len(train_ulb_noaug): {len(train_ulb_noaug)}, len(train_ulb_loader): {len(train_ulb_loader)}')
        train_ulb_weakaug_loader = DataLoader(train_ulb, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)
        print(f'len(train_ulb): {len(train_ulb)}, len(train_ulb_weakaug_loader): {len(train_ulb_weakaug_loader)}')

        # Extract PL
        if '/supervised' in load_path:
            print(f'Extracting PL for {load_path}')
            eval_dict, y_feats, y_logits, y_pred, y_probs, y_true = _eval(model, train_ulb_loader, eval_unsup=False)

            y_true = y_true.numpy()
            y_pred = y_pred.numpy()
            y_logits = y_logits.numpy()
            y_feats = y_feats.numpy()

            print(f'y_true.shape: {y_true.shape}, y_pred.shape: {y_pred.shape}, y_logits.shape: {y_logits.shape}, y_feats.shape: {y_feats.shape}')
            print(f'PL accuracy: {np.mean(y_true == y_pred)}')
            print(f'eval_dict: {eval_dict}')

            # Save pickle file
            if not os.path.exists(os.path.dirname(pl_path)):
                os.makedirs(os.path.dirname(pl_path), exist_ok=True)
            print(f'Saving PL to to {pl_path}')
            with open(pl_path, 'wb') as f:
                pickle.dump({'y_true': y_true, 
                                'y_pred': y_pred, 
                                'y_logits': y_logits, 
                                'y_feats': y_feats,
                                'ulb_lb_mask': ulb_lb_mask,
                                'eval_dict': eval_dict}, f)

            # Extract bootstrapping with weak augmentation
            weak_pl_filenames = [f'weak_{i}.pkl' for i in range(10)]
            if os.path.exists(bootstrapping_pl_path):
                for f in os.listdir(bootstrapping_pl_path):
                    if f in weak_pl_filenames:
                        print(f'Existing weak PL found: {f}, removing')
                        weak_pl_filenames.remove(f)
            print(f'Extracting weak PL for {weak_pl_filenames}')
            for _weak_pl_filename in weak_pl_filenames:
                eval_dict, y_feats, y_logits, y_pred, y_probs, y_true = _eval(model, train_ulb_weakaug_loader, eval_unsup=False)

                y_true = y_true.numpy()
                y_pred = y_pred.numpy()
                y_logits = y_logits.numpy()
                y_feats = y_feats.numpy()

                print(f"Pseudo-label accuracy for weak augmentation {_weak_pl_filename}: {np.mean(y_true == y_pred)}")

                _weak_pl_path = os.path.join(bootstrapping_pl_path, _weak_pl_filename)

                print(f'Saving weak PL to to {_weak_pl_path}')
                with open(_weak_pl_path, 'wb') as f:
                    pickle.dump({'y_true': y_true,
                                    'y_pred': y_pred,
                                    'y_logits': y_logits,
                                    'y_feats': y_feats,
                                    'eval_dict': eval_dict}, f)

        # Get unsupervised metrics on validation set
        val_loader = DataLoader(val, batch_size=32, shuffle=False, num_workers=0)
        print(f'len(val): {len(val)}, len(val_loader): {len(val_loader)}')

        val_eval_dict, val_y_feats, val_y_logits, val_y_pred, val_y_probs, val_y_true = _eval(model, val_loader, eval_unsup=True)

        val_y_true = val_y_true.numpy()
        val_y_pred = val_y_pred.numpy()
        val_y_logits = val_y_logits.numpy()
        val_y_feats = val_y_feats.numpy()

        print(f'val_y_true.shape: {val_y_true.shape}, val_y_pred.shape: {val_y_pred.shape}, val_y_logits.shape: {val_y_logits.shape}, val_y_feats.shape: {val_y_feats.shape}')
        print(f'PL accuracy: {np.mean(val_y_true == val_y_pred)}')
        print(f'val_eval_dict: {val_eval_dict}')

        # Save pickle file
        print(f'Saving PL to to {val_pl_path}')
        with open(val_pl_path, 'wb') as f:
            pickle.dump({'y_true': val_y_true, 
                            'y_pred': val_y_pred, 
                            'y_logits': val_y_logits, 
                            'y_feats': val_y_feats,
                            'eval_dict': val_eval_dict}, f)
        print(f'PL extraction done for {load_path}')
