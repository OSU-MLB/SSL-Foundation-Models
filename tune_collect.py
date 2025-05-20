import os
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
import pickle
import sqlite3
import argparse

_DB_PATH = 'exp_result.db'

_DSET_VAL_LEN = {
    'dtd': 752,
    'sun397': 17401,
    'resisc45': 5040,
    'diabetic_retinopathy': 9207,
    'clevr_count': 14000,
    'kitti': 1354,
    "_cub": 1199
}


def insert_sqlite(peft, alg, dset, n_shot, net, hyperparams, _root, val_pl_path, debug):
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    # Create db if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS exp_result
                (peft text, 
                 alg text, 
                 dset text, 
                 n_shot text, 
                 net text, 
                 hyperparams text, 
                 rankme real, 
                 ami real, 
                 ari real, 
                 v_measure real, 
                 fmi real, 
                 chi real, 
                 bnm real, 
                 test_acc real,
                 insert_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()

    # Check if exists
    c.execute("SELECT * FROM exp_result WHERE peft=? AND alg=? AND dset=? AND n_shot=? AND net=? AND hyperparams=?", (peft, alg, dset, n_shot, net, hyperparams))
    res = c.fetchall()
    if (not debug) and len(res) > 0:
        return

    # Load tensorboard logs
    tb_root = os.path.join(_root, 'tensorboard')
    ea = event_accumulator.EventAccumulator(tb_root)
    ea.Reload()
    try:
        test_acc = ea.Scalars('test/top-1-acc')
        test_acc = test_acc[-1].value
    except:
        test_acc = -1
    print(f'test_acc: {test_acc}')

    # Load val pl
    with open(val_pl_path, 'rb') as f:
        val_pl = pickle.load(f)
    len_val = len(val_pl['y_pred'])
    print(f'len of val: {len_val}')

    ###### Sanity check ######
    assert len_val == _DSET_VAL_LEN[dset], f'{dset} len_val incorrect: {len_val}, should be {_DSET_VAL_LEN[dset]}'
    ##########################
    val_eval_dict = val_pl['eval_dict']
    rankme = val_eval_dict['rankme']
    ami = val_eval_dict['ami']
    ari = val_eval_dict['ari']
    v_measure = val_eval_dict['v_measure']
    fmi = val_eval_dict['fmi']
    chi = val_eval_dict['chi']
    bnm = val_eval_dict['bnm']
    acc = val_eval_dict['acc']
    print(f'rankme: {rankme}, ami: {ami}, ari: {ari}, v_measure: {v_measure}, fmi: {fmi}, chi: {chi}, bnm: {bnm}, acc: {acc}')

    # Insert
    if not debug:
        print(f'Inserting: {peft}, {alg}, {dset}, {n_shot}, {net}, {hyperparams}')
        c.execute("INSERT INTO exp_result (peft, alg, dset, n_shot, net, hyperparams, rankme, ami, ari, v_measure, fmi, chi, bnm, test_acc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (peft, alg, dset, n_shot, net, hyperparams, rankme, ami, ari, v_measure, fmi, chi, bnm, test_acc))
        conn.commit()
    else:
        print(f"I'm in debug mode, not inserting")
    conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    debug = args.debug

    exp_root = ['./saved_models/']

    for _exp_root in exp_root:
        for root, dirs, files in os.walk(_exp_root):
            # print(f'root: {root}')
            if not ('val_pl.pkl' in files):
                continue
            _root = root.split('/')
            if len(_root) < 6:
                continue
            peft = _root[2]
            alg = _root[3]
            dset = _root[4]
            n_shot = _root[5]
            net = _root[6]
            hyperparams = str(_root[7:-1])

            # Load val pl
            val_pl_path = os.path.join(root, 'val_pl.pkl')
            insert_sqlite(peft, 
                            alg, 
                            dset, 
                            n_shot, 
                            net, 
                            hyperparams, 
                            root, 
                            val_pl_path, debug)
