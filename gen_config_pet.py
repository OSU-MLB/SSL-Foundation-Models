import os
# import yaml
from ruamel.yaml import YAML
import os
import argparse
import sqlite3
import pickle
import numpy as np
from collections import defaultdict
import itertools
from tqdm import tqdm

yaml = YAML()

datasets = ['dtd', 'sun397', 'resisc45', 'diabetic_retinopathy', 'clevr_count', 'kitti']
_DB_PATH = './exp_result.db'

epoch_dict = {
    "dtd": {
        "3-shot": {
            "num_classes": 47,
            "epoch": 30,
            "num_labels": 141,
            "num_train_iter": 3420,
            "num_warmup_iter": 85,
            "num_log_iter": 28,
            "num_eval_iter": 228,
        },
        "6-shot": {
            "num_classes": 47,
            "epoch": 30,
            "num_labels": 282,
            "num_train_iter": 3270,
            "num_warmup_iter": 81,
            "num_log_iter": 27,
            "num_eval_iter": 218,
        }
    },
    "sun397": {
        "3-shot": {
            "num_classes": 397,
            "epoch": 30,
            "num_labels": 1191,
            "num_train_iter": 80460,
            "num_warmup_iter": 2011,
            "num_log_iter": 670,
            "num_eval_iter": 5364,
        },
        "6-shot": {
            "num_classes": 397,
            "epoch": 30,
            "num_labels": 2382,
            "num_train_iter": 79350,
            "num_warmup_iter": 1983,
            "num_log_iter": 661,
            "num_eval_iter": 5290,
        }
    },
    "resisc45": {
        "1-shot": {
            "num_classes": 45,
            "epoch": 30,
            "num_labels": 45,
            "num_train_iter": 23610,
            "num_warmup_iter": 590,
            "num_log_iter": 196,
            "num_eval_iter": 1574,
        },
        "2-shot": {
            "num_classes": 45,
            "epoch": 30,
            "num_labels": 90,
            "num_train_iter": 23550,
            "num_warmup_iter": 588,
            "num_log_iter": 196,
            "num_eval_iter": 1570,
        }
    },
    "diabetic_retinopathy": {
        "40-shot": {
            "num_classes": 5,
            "epoch": 30,
            "num_labels": 200,
            "num_train_iter": 42990,
            "num_warmup_iter": 1074,
            "num_log_iter": 358,
            "num_eval_iter": 2866,
        },
        "80-shot": {
            "num_classes": 5,
            "epoch": 30,
            "num_labels": 400,
            "num_train_iter": 42780,
            "num_warmup_iter": 1069,
            "num_log_iter": 356,
            "num_eval_iter": 2852,
        }
    },
    "clevr_count": {
        "10-shot": {
            "num_classes": 8,
            "epoch": 30,
            "num_labels": 80,
            "num_train_iter": 65550,
            "num_warmup_iter": 1638,
            "num_log_iter": 546,
            "num_eval_iter": 4370,
        },
        "20-shot": {
            "num_classes": 8,
            "epoch": 30,
            "num_labels": 160,
            "num_train_iter": 65490,
            "num_warmup_iter": 1637,
            "num_log_iter": 545,
            "num_eval_iter": 4366,
        },
        "40-shot": {
            "num_classes": 8,
            "epoch": 30,
            "num_labels": 320,
            "num_train_iter": 65340,
            "num_warmup_iter": 1633,
            "num_log_iter": 544,
            "num_eval_iter": 4356,
        }
    },
    "kitti": {
        "5-shot": {
            "num_classes": 4,
            "epoch": 30,
            "num_labels": 20,
            "num_train_iter": 6330,
            "num_warmup_iter": 158,
            "num_log_iter": 52,
            "num_eval_iter": 422,
        },
        "10-shot": {
            "num_classes": 4,
            "epoch": 30,
            "num_labels": 40,
            "num_train_iter": 6330,
            "num_warmup_iter": 158,
            "num_log_iter": 52,
            "num_eval_iter": 422,
        }
    },
    "_cub": {
        "1-shot": {
            "num_classes": 200,
            "epoch": 30,
            "num_labels": 200,
            "num_train_iter": 5460,
            "num_warmup_iter": 136,
            "num_log_iter": 45,
            "num_eval_iter": 364,
        },
        "2-shot": {
            "num_classes": 200,
            "epoch": 30,
            "num_labels": 400,
            "num_train_iter": 5250,
            "num_warmup_iter": 131,
            "num_log_iter": 43,
            "num_eval_iter": 350,
        }
    }
}


def expand_fields(d):
    keys = list(d.keys())
    values = list(d.values())
    combinations = itertools.product(*values)
    result = [
        {key: value for key, value in zip(keys, combination)}
        for combination in combinations
    ]
    return result


def get_pet_sources(strategy, dset, shot, peft, net, rows):
    sources = []
    sources_rows = []
    if strategy == 'self-training':
        candidate_rows = []
        for row in rows:
            row_peft = row[0]
            row_alg = row[1]
            row_dset = row[2]
            row_shot = row[3]
            row_net = row[4]
            if 'supervised' in row_alg and row_dset == dset and row_shot == shot and row_net == net and row_peft == peft:
                candidate_rows.append(row)
        if len(candidate_rows) == 0:
            return [], []
        metrics = []
        for _candidate in candidate_rows:
            _metrics = _candidate[6: 13]
            metrics.append(_metrics)
        metrics = np.array(metrics)
        _unsup_metrics_rank = np.argsort(np.argsort(-metrics, axis=0), axis=0) + 1
        _unsup_metrics_rank = _unsup_metrics_rank ** 2
        _avg_rank = np.mean(_unsup_metrics_rank, axis=1)
        tune_result = np.argmin(_avg_rank)
        tuned_candidate = candidate_rows[tune_result]
        tuned_candidate_peft = tuned_candidate[0]
        tuned_candidate_dset = tuned_candidate[2]
        tuned_candidate_shot = tuned_candidate[3]
        tuned_candidate_net = tuned_candidate[4]
        tuned_candidate_hyperparams = eval(tuned_candidate[5])
        config_path = f"config/{tuned_candidate_peft}/supervised/{tuned_candidate_dset}/{tuned_candidate_shot}/{tuned_candidate_net}/{'/'.join(tuned_candidate_hyperparams)}/config.yaml"
        assert os.path.exists(config_path)
        sources.append(config_path)
        sources_rows.append(tuned_candidate)
    elif strategy == 'ensembled':
        candidate_rows = defaultdict(list)
        for row in rows:
            row_peft = row[0]
            row_alg = row[1]
            row_dset = row[2]
            row_shot = row[3]
            row_net = row[4]
            if 'supervisped' in row_alg and row_dset == dset and row_shot == shot and row_net == net:
                k = str((row_peft, row_alg, row_dset, row_shot, row_net))
                candidate_rows[k].append(row)
        if len(candidate_rows) == 0:
            return [], []
        for k, _candidate_rows in candidate_rows.items():
            metrics = []
            for _candidate in _candidate_rows:
                _metrics = _candidate[6: 13]
                metrics.append(_metrics)
            metrics = np.array(metrics)
            _unsup_metrics_rank = np.argsort(np.argsort(-metrics, axis=0), axis=0) + 1
            _unsup_metrics_rank = _unsup_metrics_rank ** 2
            _avg_rank = np.mean(_unsup_metrics_rank, axis=1)
            tune_result = np.argmin(_avg_rank)
            tuned_candidate = _candidate_rows[tune_result]
            tuned_candidate_peft = tuned_candidate[0]
            tuned_candidate_dset = tuned_candidate[2]
            tuned_candidate_shot = tuned_candidate[3]
            tuned_candidate_net = tuned_candidate[4]
            tuned_candidate_hyperparams = eval(tuned_candidate[5])
            config_path = f'config/{tuned_candidate_peft}/supervised/{tuned_candidate_dset}/{tuned_candidate_shot}/{tuned_candidate_net}/{"/".join(tuned_candidate_hyperparams)}/config.yaml'
            assert os.path.exists(config_path), f'Not found: {config_path}'
            sources.append(config_path)
            sources_rows.append(tuned_candidate)
    elif strategy == 'ensembled-across-nets':
        candidate_rows = defaultdict(list)
        for row in rows:
            row_peft = row[0]
            row_alg = row[1]
            row_dset = row[2]
            row_shot = row[3]
            row_net = row[4]
            if row_alg == 'supervised' and row_dset == dset and row_shot == shot:
                k = str((row_peft, row_alg, row_dset, row_shot, row_net))
                candidate_rows[k].append(row)
        if len(candidate_rows) == 0:
            return [], []
        for k, _candidate_rows in candidate_rows.items():
            metrics = []
            for _candidate in _candidate_rows:
                _metrics = _candidate[6: 13]
                metrics.append(_metrics)
            metrics = np.array(metrics)
            _unsup_metrics_rank = np.argsort(np.argsort(-metrics, axis=0), axis=0) + 1
            _unsup_metrics_rank = _unsup_metrics_rank ** 2
            _avg_rank = np.mean(_unsup_metrics_rank, axis=1)
            tune_result = np.argmin(_avg_rank)
            tuned_candidate = _candidate_rows[tune_result]
            tuned_candidate_peft = tuned_candidate[0]
            tuned_candidate_dset = tuned_candidate[2]
            tuned_candidate_shot = tuned_candidate[3]
            tuned_candidate_net = tuned_candidate[4]
            tuned_candidate_hyperparams = eval(tuned_candidate[5])
            config_path = f'config/{tuned_candidate_peft}/supervised/{tuned_candidate_dset}/{tuned_candidate_shot}/{tuned_candidate_net}/{"/".join(tuned_candidate_hyperparams)}/config.yaml'
            assert os.path.exists(config_path), f'Not found: {config_path}'
            sources.append(config_path)
            sources_rows.append(tuned_candidate)
    else:
        raise ValueError(f'Invalid strategy: {strategy}')
    return sources, sources_rows

peft_params = {
    'adaptformer': {'adapter_scaler': 0.1, 'adapter_bottleneck': 16},
    'lora': {'lora_bottleneck': 4},
}
datasets = ['dtd', 'sun397', 'resisc45', 'diabetic_retinopathy', 'clevr_count', 'kitti']

if __name__ == '__main__':
    
    ## Load exp_result
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    # Load as table
    c.execute('SELECT * FROM exp_result')
    rows = c.fetchall()
    conn.close()

    alg = 'pet'
    template_path = f'./config/template/pet.yaml'
    strategy_space = ['self-training', 'ensembled', 'ensembled-across-nets']

    for net in ['dinov2', 'clip']:
        for dset in datasets:
            for peft in ['adaptformer', 'lora']:
                for shot, arg in epoch_dict[dset].items():
                    for strategy in strategy_space:
                        pet_sources, pet_sources_rows = get_pet_sources(strategy, dset, shot, peft, net, rows)
                            
                        _peft_params = peft_params[peft]
                        
                        template = yaml.load(open(template_path, 'r'))
                        epoch = arg['epoch']
                        num_classes = arg['num_classes']
                        num_train_iter = arg['num_train_iter']
                        num_warmup_iter = arg['num_warmup_iter']
                        num_log_iter = arg['num_log_iter']
                        num_eval_iter = arg['num_eval_iter']
                        num_labels = arg['num_labels']
                        img_size = 224
                        layer_decay = 1.0

                        if net == 'dinov2':
                            net_name = 'timm/vit_base_patch14_reg4_dinov2.lvd142m'
                        elif net == 'clip':
                            net_name = 'timm/vit_base_patch16_clip_224.openai'
                        _config = template.copy()
                        _config['lr'] = float(0.0001)
                        _config['epoch'] = epoch
                        _config['algorithm'] = alg
                        _config['dataset'] = dset
                        _config['train_split'] = 'train'
                        _config['num_classes'] = num_classes
                        _config['num_labels'] = num_labels
                        _config['num_train_iter'] = num_train_iter
                        _config['num_warmup_iter'] = num_warmup_iter
                        _config['num_log_iter'] = num_log_iter
                        _config['num_eval_iter'] = num_eval_iter
                        _config['img_size'] = img_size
                        _config['layer_decay'] = layer_decay
                        _config['net'] = net_name
                        _config['logits_ensemble'] = "voting"
                        _config['bootstrapping'] = True
                        
                        _config['w_alpha'] = 0.0
                        _config['s_alpha'] = 0.0
                        _config['kd_w_alpha'] = 1.
                        _config['kd_s_alpha'] = 1.
                        
                        if peft == 'lora':
                            _config['peft_config'] = {
                                'method_name': 'lora',
                                'lora_bottleneck': 4
                            }
                        elif peft == 'adaptformer':
                            _config['peft_config'] = {
                                'method_name': 'adaptformer',
                                'ft_mlp_module': 'adapter',
                                'ft_mlp_mode': 'parallel',
                                'ft_mlp_ln': 'before',
                                'adapter_init': 'lora_kaiming',
                                'adapter_bottleneck': 16,
                                'adapter_scaler': 0.1
                            }
                        else:
                            raise ValueError(f'Invalid peft: {peft}')

                        _config['vit_config'] = {
                            'drop_path_rate': 0.
                        }
                        if 'peft_config' in _config and _config['peft_config'] is not None:
                            _config['peft_config'].update(_peft_params)

                        alg_name = f'pet-{strategy}'

                        _config['pet_sources'] = pet_sources
                        _config['pet_sources_rows'] = pet_sources_rows
                        # Save
                        config_path = f'config/{peft}/{alg_name}/{dset}/{shot}/{net}/config.yaml'
                        save_dir = f'./saved_models/{os.path.dirname(config_path)}'
                        load_path = f'{save_dir}/log/latest_model.pth'
                        _config['save_dir'] = save_dir
                        _config['load_path'] = load_path

                        os.makedirs(os.path.dirname(config_path), exist_ok=True)
                        yaml.dump(_config, open(config_path, 'w'))
                        print(f"Saved config to {config_path}. ")
