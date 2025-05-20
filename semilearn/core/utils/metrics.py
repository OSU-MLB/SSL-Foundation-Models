import torch
import numpy as np
from pytorch_adapt.validators import (
    BNMValidator, SNDValidator, ClassClusterValidator, ClassClusterValidatorCached
)
from torch.nn import functional as F
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             v_measure_score, fowlkes_mallows_score,
                             silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
import time
from sklearn.cluster import KMeans


_D_SCORE_FN = {
    'ami': adjusted_mutual_info_score,
    'ari': adjusted_rand_score,
    'v_measure': v_measure_score,
    'fmi': fowlkes_mallows_score,
    'silhouette': silhouette_score,
    'chi': calinski_harabasz_score,
    'dbi': davies_bouldin_score
}


_D_SCORE_FN_TYPE = {
    'ami': 'labels',
    'ari': 'labels',
    'v_measure': 'labels',
    'fmi': 'labels',
    'silhouette': 'features',
    'chi': 'features',
    'dbi': 'features'
}


def get_centroids(data, labels, num_classes):
    centroids = np.zeros((num_classes, data.shape[1]))
    for cid in range(num_classes):
        # Since we are using pseudolabels to compute centroids, some classes might not have instances according to the
        # pseudolabels assigned by the current model. In that case .mean() would return NaN causing KMeans to fail.
        # We set to 0 the missing centroids
        if (labels == cid).any():
            centroids[cid] = data[labels == cid].mean(0)
        return centroids


def cluster_scores(features, logits, score_fn, clabel_cache=None):
    assert score_fn in ['ami', 
                        'ari', 
                        'v_measure', 
                        'fmi', 
                        'silhouette', 
                        'chi', 
                        'dbi'], f"Invalid score_fn: {score_fn}"
    _score_fn = _D_SCORE_FN[score_fn]
    _score_fn_type = _D_SCORE_FN_TYPE[score_fn]
    _input = dict(features=features, logits=logits)

    if clabel_cache is not None:
        validator = ClassClusterValidatorCached(score_fn=_score_fn, score_fn_type=_score_fn_type, clabel_cache=clabel_cache)
    else:
        validator = ClassClusterValidator(score_fn=_score_fn, score_fn_type=_score_fn_type)
    _score = validator(target_train=_input)
    return _score


def rankme_score(Z):
    """
    RankMe smooth rank estimation
    from: https://arxiv.org/abs/2210.02885

    Z: (N, K), N: nb samples, K: embed dim
    N = 25000 is a good approximation in general
    """

    S = torch.linalg.svdvals(Z) # singular values
    S_norm1 = torch.linalg.norm(S, 1)

    p = S/S_norm1 + 1e-7 # normalize sum to 1
    entropy = - torch.sum(p*torch.log(p))
    return torch.exp(entropy).item()


def bnm_score(features, logits):
    _input = dict(features=features, logits=logits)
    validator = BNMValidator()
    _score = validator(target_train=_input)
    return _score


def snd_score(features, logits, probs):
    _input = dict(features=features, logits=logits, preds=probs)
    validator = SNDValidator()
    _score = validator(target_train=_input)
    return _score


def _unsupervised_scores(feats, logits, probs, k_append, scores):

    for score in scores:
        assert score in ['rankme', 'ami', 'ari', 'v_measure', 'fmi', 'silhouette', 'dbi', 'chi', 'bnm', 'snd'], f"Invalid score: {score}"

    results = {}
    t_start = time.time()

    for score in scores:
        t_score_start = time.time()
        if score == 'rankme':
            _s = rankme_score(feats)
        elif score in ['ami', 'ari', 'v_measure', 'fmi', 'silhouette', 'dbi', 'chi']:
            _s = cluster_scores(feats, logits, score)
        elif score == 'bnm':
            _s = bnm_score(feats, logits)
        elif score == 'snd':
            _s = snd_score(feats, logits, probs)
        results[f'{score}{k_append}'] = _s
        t_score_end = time.time()
        print(f'{score}{k_append} took: {t_score_end - t_score_start:.2f}s')
    t_end = time.time()
    print(f'All scores took: {t_end - t_start:.2f}s')
    return results


from collections import defaultdict

def label_to_distribution(labels):
    distribution = defaultdict(int)
    for label in labels:
        distribution[label] += 1
    return distribution

def _unsupervised_scores_cached(feats, logits, probs, k_append, scores):

    for score in scores:
        assert score in ['rankme', 'ami', 'ari', 'v_measure', 'fmi', 'silhouette', 'dbi', 'chi', 'bnm', 'snd'], f"Invalid score: {score}"

    # clabel cache
    labels = torch.argmax(logits, dim=1)
    num_classes = logits.shape[1]
    # centroids = get_centroids(feats, labels, num_classes)
    # clustering = KMeans(n_clusters=num_classes, init=centroids, n_init=1)
    clustering = KMeans(n_clusters=num_classes)
    # np.random.seed(0)
    clustering.fit(feats)
    clabels = clustering.labels_

    sorted_clabels_distribution = sorted(label_to_distribution(clabels).items(), key=lambda x: x[1], reverse=True)
    print(f'sorted_clabels_distribution: {[k[1] for k in sorted_clabels_distribution]}')

    results = {}
    t_start = time.time()
    for score in scores:
        t_score_start = time.time()
        if score == 'rankme':
            _s = rankme_score(feats)
        elif score in ['ami', 'ari', 'v_measure', 'fmi', 'silhouette', 'dbi', 'chi']:
            _s = cluster_scores(feats, logits, score, clabel_cache=clabels)
        elif score == 'bnm':
            _s = bnm_score(feats, logits)
        elif score == 'snd':
            _s = snd_score(feats, logits, probs)
        results[f'{score}{k_append}'] = _s
        t_score_end = time.time()
        print(f'{score}{k_append} took: {t_score_end - t_score_start:.2f}s')
    t_end = time.time()
    print(f'All scores took: {t_end - t_start:.2f}s')
    return results


def unsupervised_scores(feats, logits, probs, scores=['rankme', 'ami', 'ari', 'v_measure', 'fmi', 'silhouette', 'dbi', 'chi', 'bnm', 'snd']):
    if isinstance(feats, np.ndarray):
        feats = torch.tensor(feats)
        logits = torch.tensor(logits)
        probs = torch.tensor(probs)

    feats_normalized = F.normalize(feats, p=2, dim=1)
    # feats_normalized = feats

    # scores = _unsupervised_scores(t_feats, t_logits, t_probs, '_raw')
    # normalized_scores = _unsupervised_scores(t_feats_normalized, t_logits, t_probs, '')
    # scores.update(normalized_scores)

    # scores = _unsupervised_scores(feats_normalized, logits, probs, '', scores)
    scores = _unsupervised_scores_cached(feats_normalized, logits, probs, '', scores)


    print('scores: ', scores)
    # print('scores_cached: ', scores_cached)

    return scores
