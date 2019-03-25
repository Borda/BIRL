"""
Evaluate experiments

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

from itertools import chain

import numpy as np
from scipy.spatial import distance

from birl.utilities.registration import (estimate_affine_transform, get_affine_components,
                                         norm_angle)


def compute_points_dist_statistic(points_ref, points_est):
    """ compute distance as between related points in two sets
    and make a statistic on those distances - mean, std, median, min, max

    :param points_ref: np.array<nb_points, dim>
    :param points_est: np.array<nb_points, dim>
    :return: (np.array<nb_points, 1>, {str: float})

    >>> points_ref = np.array([[1, 2], [3, 4], [2, 1]])
    >>> points_est = np.array([[3, 4], [2, 1], [1, 2]])
    >>> dist, stat = compute_points_dist_statistic(points_ref, points_ref)
    >>> dist
    array([ 0.,  0.,  0.])
    >>> all(stat[k] == 0 for k in stat if k not in ['overlap points'])
    True
    >>> dist, stat = compute_points_dist_statistic(points_ref, points_est)
    >>> dist  #doctest: +ELLIPSIS
    array([ 2.828...,  3.162...,  1.414...])
    >>> import pandas as pd
    >>> pd.Series(stat).sort_index()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Max               3.16...
    Mean              2.46...
    Mean_weighted     2.52...
    Median            2.82...
    Min               1.41...
    STD               0.75...
    overlap points    1.00...
    dtype: float64

    Wrong input:
    >>> compute_points_dist_statistic(None, np.array([[1, 2], [3, 4], [2, 1]]))
    ([], {'overlap points': 0})
    """
    if not all(pts is not None and list(pts) for pts in [points_ref, points_est]):
        return [], {'overlap points': 0}

    lnd_sizes = [len(points_ref), len(points_est)]
    nb_common = min(lnd_sizes)
    assert nb_common > 0, 'no common landmarks for metric'
    points_ref = np.asarray(points_ref)[:nb_common]
    points_est = np.asarray(points_est)[:nb_common]
    diffs = np.sqrt(np.sum(np.power(points_ref - points_est, 2), axis=1))

    inter_dist = distance.cdist(points_ref, points_ref)
    # inter_dist[range(len(points_ref)), range(len(points_ref))] = np.inf
    dist = np.mean(inter_dist, axis=0)
    weights = dist / np.sum(dist)

    dict_stat = {
        'Mean': np.mean(diffs),
        'Mean_weighted': np.sum(diffs * weights),
        'STD': np.std(diffs),
        'Median': np.median(diffs),
        'Min': np.min(diffs),
        'Max': np.max(diffs),
        'overlap points': nb_common / float(max(lnd_sizes))
    }
    return diffs, dict_stat


def compute_affine_transf_diff(points_ref, points_init, points_est):
    """ compute differences between initial state and estimated results

    :param points_ref: np.array<nb_points, dim>
    :param points_init: np.array<nb_points, dim>
    :param points_est: np.array<nb_points, dim>
    :return:

    >>> points_ref = np.array([[1, 2], [3, 4], [2, 1]])
    >>> points_init = np.array([[3, 4], [1, 2], [2, 1]])
    >>> points_est = np.array([[3, 4], [2, 1], [1, 2]])
    >>> diff = compute_affine_transf_diff(points_ref, points_init, points_est)
    >>> import pandas as pd
    >>> pd.Series(diff).sort_index()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Affine rotation Diff        -8.97...
    Affine scale X Diff         -0.08...
    Affine scale Y Diff         -0.20...
    Affine shear Diff           -1.09...
    Affine translation X Diff   -1.25...
    Affine translation Y Diff    1.25...
    dtype: float64

    Wrong input:
    >>> compute_affine_transf_diff(None, np.array([[1, 2], [3, 4], [2, 1]]), None)
    {}
    """
    if not all(pts is not None and list(pts) for pts in [points_ref, points_init, points_est]):
        return {}

    points_ref = np.nan_to_num(points_ref)
    mtx_init = estimate_affine_transform(points_ref, np.nan_to_num(points_init))[0]
    affine_init = get_affine_components(np.asarray(mtx_init))
    mtx_est = estimate_affine_transform(points_ref, np.nan_to_num(points_est))[0]
    affine_estim = get_affine_components(np.asarray(mtx_est))

    diff = {'Affine %s %s Diff' % (n, c): (np.array(affine_estim[n]) - np.array(affine_init[n]))[i]
            for n in ['translation', 'scale'] for i, c in enumerate(['X', 'Y'])}
    diff.update({'Affine %s Diff' % n: norm_angle(affine_estim[n] - affine_init[n], deg=True)
                 for n in ['rotation']})
    diff.update({'Affine %s Diff' % n: affine_estim[n] - affine_init[n]
                 for n in ['shear']})
    return diff


def compute_ranking(user_cases, field, reverse=False):
    """ compute ranking over selected field

    :param {str: {}} user_cases: dictionary with measures for user and case
    :param str field: name of field to be ranked
    :param bool reverse: use reverse ordering
    :return {}: extended dictionary

    >>> user_cases = {
    ...     'karel': {1: {'rTRE': 0.04}, 2: {'rTRE': 0.25}, 3: {'rTRE': 0.1}},
    ...     'pepa': {1: {'rTRE': 0.33}, 3: {'rTRE': 0.05}},
    ...     'franta': {2: {'rTRE': 0.01}, 3: {'rTRE': 0.15}}
    ... }
    >>> user_cases = compute_ranking(user_cases, 'rTRE')
    >>> import pandas as pd
    >>> pd.DataFrame({usr: {cs: user_cases[usr][cs]['rTRE_rank'] for cs in user_cases[usr]}
    ...               for usr in user_cases})[sorted(user_cases.keys())]  # doctest: +NORMALIZE_WHITESPACE
       franta  karel  pepa
    1       3      1     2
    2       1      2     3
    3       3      2     1
    """
    users = list(user_cases.keys())
    cases = set(chain(*[user_cases[u].keys() for u in user_cases]))

    for cs in cases:
        usr_val = [(u, user_cases[u][cs].get(field, np.nan))
                   for u in users if cs in user_cases[u]]
        usr_val = sorted(usr_val, key=lambda x: x[1], reverse=reverse)
        usr_rank = dict((usr, i + 1) for i, (usr, _) in enumerate(usr_val))
        for usr in users:
            if cs not in user_cases[usr]:
                user_cases[usr][cs] = {}
            user_cases[usr][cs][field + '_rank'] = usr_rank.get(usr, len(users))

    return user_cases
