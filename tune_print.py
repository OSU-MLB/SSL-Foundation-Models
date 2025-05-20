import os
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
import pickle
import sqlite3
import numpy as np
from scipy.stats import mode
import time
import argparse
from ruamel.yaml import YAML
_yaml = YAML()

_DB_PATH = 'exp_result.db'

_UNSUP_USE_LOGITS = np.array([False, True, True, True, True, False, True])

def mode(arr):
    freq = np.bincount(arr)
    pool = np.where(freq == np.max(freq))[0]
    if len(pool) == 1:
        result = pool[0]
    else:
        result = np.random.choice(pool)
    return result, len(pool) - 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='rank', choices=['vote', 'rank', 'oracle', 'non_squared_rank', 'rankme', 'random', 'ami', 'ari', 'v_measure', 'fmi', 'chi', 'bnm'])
    args = parser.parse_args()
    strategy = args.strategy

    # Connect to './exp_result.db'
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    # Load as table
    c.execute('SELECT * FROM exp_result')
    rows = c.fetchall()
    conn.close()

    results = defaultdict(list)

    for row in rows:
        k = row[:5]
        hyperparams = row[5]
        unsup_metrics = row[6:13]
        test_acc = row[13]

        results[k].append((hyperparams, unsup_metrics, test_acc))
    
    # Vectorize
    for k, v in results.items():
        hyperparams = np.array([vv[0] for vv in v])
        unsup_metrics = np.array([vv[1] for vv in v])
        test_acc = np.array([vv[2] for vv in v])
        results[k] = (hyperparams, unsup_metrics, test_acc)
    
    tune_results = dict()
    # Hypertune
    for k, v in results.items():
        hyperparams, unsup_metrics, test_acc = v
        if k[0] == 'linear':
            _mask =  _UNSUP_USE_LOGITS
        else:
            _mask = np.ones_like(_UNSUP_USE_LOGITS, dtype=bool)
        _unsup_metrics = unsup_metrics[:, _mask]
        # Max vote
        if strategy == 'vote':
            _unsup_metrics_vote = np.argmax(_unsup_metrics, axis=0)
            tune_result, _ = mode(_unsup_metrics_vote)
        # Average rank
        elif strategy == 'rank':
            _unsup_metrics_rank = np.argsort(np.argsort(-_unsup_metrics, axis=0), axis=0) + 1
            _unsup_metrics_rank = _unsup_metrics_rank ** 2
            _avg_rank = np.mean(_unsup_metrics_rank, axis=1)
            tune_result = np.argmin(_avg_rank)
        elif strategy == 'non_squared_rank':
            _unsup_metrics_rank = np.argsort(np.argsort(-_unsup_metrics, axis=0), axis=0) + 1
            _avg_rank = np.mean(_unsup_metrics_rank, axis=1)
            tune_result = np.argmin(_avg_rank)
        elif strategy in ['rankme', 'ami', 'ari', 'v_measure', 'fmi', 'chi', 'bnm']:
            _mask = np.zeros_like(_mask, dtype=bool)
            if strategy == 'rankme':
                _mask[0] = True
            else:
                _mask[1 + ['ami', 'ari', 'v_measure', 'fmi', 'chi', 'bnm'].index(strategy)] = True
            _unsup_metrics = unsup_metrics[:, _mask]
            _unsup_metrics_rank = np.argsort(np.argsort(-_unsup_metrics, axis=0), axis=0) + 1
            _avg_rank = np.mean(_unsup_metrics_rank, axis=1)
            tune_result = np.argmin(_avg_rank)
        elif strategy == 'oracle':
            tune_result = np.argmax(test_acc)
        elif strategy == 'random':
            tune_result = test_acc
        if strategy != 'random':
            tune_results[k] = (hyperparams, hyperparams[tune_result], test_acc[tune_result])
        else:
            tune_results[k] = (hyperparams, None, np.mean(tune_result))


    out_path = f'tune_results_{time.strftime("%m%d%H%M")}_{strategy}.txt'
    # Write _UNSUP_METRICS_MASK
    print(f'Writing to {out_path}')

    # Print
    for k, v in tune_results.items():
        peft = k[0]
        alg = k[1]
        dset = k[2]
        n_shot = int(k[3].split('-')[0])
        net = k[4]
        hyperparams = str(v[0]).replace('\n', '')
        tuned_hyperparam = str(v[1])
        test_acc = float(v[2])

        print(f'{peft}\t{alg}\t{dset}\t{n_shot}\t{net}\t{hyperparams}\t{tuned_hyperparam}\t{test_acc}')
        print(f'Writing to {out_path}')
        with open(out_path, 'a') as f:
            f.write(f'{peft}\t{alg}\t{dset}\t{n_shot}\t{net}\t{hyperparams}\t{tuned_hyperparam}\t{test_acc}\n')
