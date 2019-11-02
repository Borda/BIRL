"""
Evaluate experiments

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import

from itertools import chain
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial import distance

from .registration import estimate_affine_transform, get_affine_components, norm_angle


def compute_tre(points_1, points_2):
    """ computing Target Registration Error for each landmark pair

    :param ndarray points_1: set of points
    :param ndarray points_2: set of points
    :return ndarray: list of errors of size min nb of points

    >>> np.random.seed(0)
    >>> compute_tre(np.random.random((6, 2)),
    ...             np.random.random((9, 2)))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([ 0.21...,  0.70...,  0.44...,  0.34...,  0.41...,  0.41...])
    """
    nb_common = min([len(pts) for pts in [points_1, points_2]
                     if pts is not None])
    assert nb_common > 0, 'no common landmarks for metric'
    points_1 = np.asarray(points_1)[:nb_common]
    points_2 = np.asarray(points_2)[:nb_common]
    diffs = np.sqrt(np.sum(np.power(points_1 - points_2, 2), axis=1))
    return diffs


def compute_target_regist_error_statistic(points_ref, points_est):
    """ compute distance as between related points in two sets
    and make a statistic on those distances - mean, std, median, min, max

    :param ndarray points_ref: final landmarks in target image of  np.array<nb_points, dim>
    :param ndarray points_est: warped landmarks from source to target of np.array<nb_points, dim>
    :return tuple(ndarray,dict): (np.array<nb_points, 1>, dict)

    >>> points_ref = np.array([[1, 2], [3, 4], [2, 1]])
    >>> points_est = np.array([[3, 4], [2, 1], [1, 2]])
    >>> dist, stat = compute_target_regist_error_statistic(points_ref, points_ref)
    >>> dist
    array([ 0.,  0.,  0.])
    >>> all(stat[k] == 0 for k in stat if k not in ['overlap points'])
    True
    >>> dist, stat = compute_target_regist_error_statistic(points_ref, points_est)
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

    >>> # Wrong input:
    >>> compute_target_regist_error_statistic(None, np.array([[1, 2], [3, 4], [2, 1]]))
    ([], {'overlap points': 0})
    """
    if not all(pts is not None and list(pts) for pts in [points_ref, points_est]):
        return [], {'overlap points': 0}

    lnd_sizes = [len(points_ref), len(points_est)]
    assert min(lnd_sizes) > 0, 'no common landmarks for metric'
    diffs = compute_tre(points_ref, points_est)

    inter_dist = distance.cdist(points_ref[:len(diffs)], points_ref[:len(diffs)])
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
        'overlap points': min(lnd_sizes) / float(max(lnd_sizes))
    }
    return diffs, dict_stat


def compute_tre_robustness(points_target, points_init, points_warp):
    """ compute robustness as improvement for each TRE

    :param ndarray points_target: final landmarks in target image
    :param ndarray points_init: initial landmarks in source image
    :param ndarray points_warp: warped landmarks from source to target
    :return bool: improvement

    >>> np.random.seed(0)
    >>> compute_tre_robustness(np.random.random((10, 2)),
    ...                        np.random.random((9, 2)),
    ...                        np.random.random((8, 2)))
    0.375
    >>> compute_tre_robustness(np.random.random((10, 2)),
    ...                        np.random.random((9, 2)) + 5,
    ...                        np.random.random((8, 2)) + 2)
    1.0
    """
    assert all(pts is not None for pts in [points_init, points_target, points_warp])
    nb_common = min([len(pts) for pts in [points_init, points_target, points_warp]])
    tre_init = compute_tre(points_init[:nb_common], points_target[:nb_common])
    tre_final = compute_tre(points_warp[:nb_common], points_target[:nb_common])
    robust = np.sum(tre_final < tre_init) / float(len(tre_final))
    return robust


def compute_affine_transf_diff(points_ref, points_init, points_est):
    """ compute differences between initial state and estimated results

    :param ndarray points_ref: np.array<nb_points, dim>
    :param ndarray points_init: np.array<nb_points, dim>
    :param ndarray points_est: np.array<nb_points, dim>
    :return ndarray: list of errors

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

    >>> # Wrong input:
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

    :param dict(dict) user_cases: dictionary with measures for user and case
    :param str field: name of field to be ranked
    :param bool reverse: use reverse ordering
    :return dict(dict(dict)): extended dictionary

    >>> user_cases = {
    ...     'karel': {1: {'rTRE': 0.04}, 2: {'rTRE': 0.25}, 3: {'rTRE': 0.1}},
    ...     'pepa': {1: {'rTRE': 0.33}, 3: {'rTRE': 0.05}},
    ...     'franta': {2: {'rTRE': 0.01}, 3: {'rTRE': 0.15}}
    ... }
    >>> user_cases = compute_ranking(user_cases, 'rTRE')
    >>> import pandas as pd
    >>> df = pd.DataFrame({usr: {cs: user_cases[usr][cs]['rTRE_rank']
    ...                          for cs in user_cases[usr]}
    ...                    for usr in user_cases})[sorted(user_cases.keys())]
    >>> df  # doctest: +NORMALIZE_WHITESPACE
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


def compute_matrix_user_ranking(df_stat, higher_better=False):
    """ compute ranking matrix over features in columns
    sorting per column and unique colour per user

    :param DF df_stat: table where index are users and columns are scoring
    :param bool higher_better: ranking such that larger value is better
    :return ndarray: ranking with features in columns

    >>> np.random.seed(0)
    >>> df = pd.DataFrame(np.random.random((5, 3)), columns=list('abc'))
    >>> compute_matrix_user_ranking(df)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 3.,  1.,  4.],
           [ 2.,  0.,  3.],
           [ 1.,  3.,  0.],
           [ 0.,  2.,  1.],
           [ 4.,  4.,  2.]])
    """
    ranking = np.zeros(df_stat.as_matrix().shape)
    nan = -np.inf if higher_better else np.inf
    for i, col in enumerate(df_stat.columns):
        vals = [v if not np.isnan(v) else nan for v in df_stat[col]]
        idx_vals = list(zip(range(len(df_stat)), vals))
        # sort according second element - vale
        idx_vals = sorted(idx_vals, key=lambda iv: iv[1], reverse=higher_better)
        # if values are NaN keep index as NaN
        ranking[:, i] = [idx if val != nan else np.nan for idx, val in idx_vals]
    return ranking


def grouping_cumulative(df, col_index, col_column):
    """ compute histogram statistic over selected column and in addition group this histograms

    :param DataFrame df: rich table
    :param str col_index: column which will be used s index in resulting table
    :param str col_column: column used for computing a histogram
    :return DF:

    >>> np.random.seed(0)
    >>> df = pd.DataFrame()
    >>> df['result'] = np.random.randint(0, 2, 50)
    >>> df['user'] = np.array(list('abc'))[np.random.randint(0, 3, 50)]
    >>> grouping_cumulative(df, 'user', 'result').astype(int)  # doctest: +NORMALIZE_WHITESPACE
           0   1
    user
    a     10  12
    b      4   9
    c      6   9
    """
    df_counts = pd.DataFrame()
    for idx, dfg in df[[col_index, col_column]].groupby(col_index):
        counts = dict(Counter(dfg[col_column]))
        counts[col_index] = idx
        df_counts = df_counts.append(counts, ignore_index=True)
    df_counts.set_index(col_index, inplace=True)
    return df_counts


def aggregate_user_score_timeline(df, col_aggreg, col_user, col_score,
                                  lower_better=True, top_down=True, interp=False):
    """ compute some cumulative statistic over given table, assuming col_aggreg is continues
    first it is grouped by col_aggreg and chose min/max (according to lower_better) of col_score
    assuming that col_aggreg is sortable like a timeline do propagation of min/max
    from past values depending on top_down (which reverse the order)

    :param df: rich table containing col_aggreg, col_user, col_score
    :param str col_aggreg: used for grouping assuming to be like a timeline
    :param str col_user: by this column the scores are assumed to be independent
    :param str col_score: the scoring value for selecting the best
    :param bool lower_better: taking min/max of scoring value
    :param bool top_down: reversing the order according to col_aggreg
    :param bool interp: in case some scores for col_aggreg are missing, interpolate from past
    :return DF: table

    >>> np.random.seed(0)
    >>> df = pd.DataFrame()
    >>> df['day'] = np.random.randint(0, 5, 50)
    >>> df['user'] = np.array(list('abc'))[np.random.randint(0, 3, 50)]
    >>> df['score'] = np.random.random(50)
    >>> df_agg = aggregate_user_score_timeline(df, 'day', 'user', 'score')
    >>> df_agg.round(3)  # doctest: +NORMALIZE_WHITESPACE
           b      c      a
    4  0.447  0.132  0.567
    0  0.223  0.005  0.094
    3  0.119  0.005  0.094
    1  0.119  0.005  0.094
    2  0.119  0.005  0.020
    """
    users = df[col_user].unique().tolist()
    aggrs = df[col_aggreg].unique().tolist()
    mtx = np.full((len(aggrs), len(users)), fill_value=np.nan)
    fn_best = np.nanmin if lower_better else np.nanmax
    # for each user
    for usr, dfg in df.groupby(col_user):
        # find the best over particular time unit - day
        for agg, dfgg in dfg.groupby(col_aggreg):
            mtx[aggrs.index(agg), users.index(usr)] = fn_best(dfgg[col_score])
    # for each user
    for j in range(len(users)):
        # depending on the schema invert timeline
        vrange = range(len(aggrs)) if top_down else range(len(aggrs))[::-1]
        # if the particular value is not NaN or interpolate missing values
        for i in (i for i in vrange if interp or not np.isnan(mtx[i, j])):
            vals = mtx[:i + 1, j] if top_down else mtx[i:, j]
            mtx[i, j] = fn_best(vals)
    df_agg = pd.DataFrame(mtx, columns=users, index=aggrs)
    return df_agg
