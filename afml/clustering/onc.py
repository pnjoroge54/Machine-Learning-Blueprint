"""
Optimal Number of Clusters (ONC Algorithm)
Detection of False Investment Strategies using Unsupervised Learning Methods
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017

Revision notes:
- Fixed Python 3 lazy map bug in _improve_clusters.
- Added deterministic random_state to KMeans and set n_init=10, matching the
  reproducibility convention used in the article's base clustering snippet.
"""


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def _improve_clusters(
    corr_mat: pd.DataFrame, clusters: dict, top_clusters: dict
) -> pd.DataFrame | dict | pd.Series:
    """
    Improve number clusters using silhouette scores.

    :param corr_mat: (pd.DataFrame) Correlation matrix.
    :param clusters: (dict) Current clusters.
    :param top_clusters: (dict) Improved clusters.
    :return: (tuple) [ordered correlation matrix, clusters, silh scores].
    """
    clusters_new, new_idx = {}, []

    for i in clusters:
        clusters_new[len(clusters_new.keys())] = list(clusters[i])

    for i in top_clusters:
        clusters_new[len(clusters_new.keys())] = list(top_clusters[i])

    # Python 3 map is lazy; list() forces the side effect.
    list(map(new_idx.extend, clusters_new.values()))

    corr_new = corr_mat.loc[new_idx, new_idx]

    dist = ((1 - corr_mat.fillna(0)) / 2.0) ** 0.5

    kmeans_labels = np.zeros(len(dist.columns))
    for i in clusters_new:
        idxs = [dist.index.get_loc(k) for k in clusters_new[i]]
        kmeans_labels[idxs] = i

    silh_scores_new = pd.Series(silhouette_samples(dist, kmeans_labels), index=dist.index)

    return corr_new, clusters_new, silh_scores_new


def _cluster_kmeans_base(
    corr_mat: pd.DataFrame,
    max_num_clusters: int = 10,
    repeat: int = 10,
    random_state: int = 42,
    is_distance_matrix: bool = False,
) -> pd.DataFrame | dict | pd.Series:
    """
    Initial clustering step using KMeans.

    :param corr_mat: (pd.DataFrame) Correlation matrix.
    :param max_num_clusters: (int) Maximum number of clusters to search for.
    :param repeat: (int) Number of clustering algorithm repetitions.
    :param random_state: (int) Random seed for KMeans reproducibility.
    :param is_distance_matrix: (bool) If True, corr_mat is already a distance matrix.
    :return: (tuple) [ordered correlation matrix, clusters, silh scores].
    """
    if not is_distance_matrix:
        # Both statements below deliberately avoid writing through the caller's
        # frame. clip(inplace=True) would mutate the correlation matrix the
        # caller passed in, and np.fill_diagonal(corr_mat.values, 1) writes
        # through a buffer that pandas marks read-only under copy-on-write, so
        # it raises instead of setting the diagonal. Work on an owned array and
        # rebind the local name.
        corr_mat = corr_mat.clip(-1, 1)
        _arr = corr_mat.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(_arr, 1)
        corr_mat = pd.DataFrame(_arr, index=corr_mat.index, columns=corr_mat.columns)
        distance = ((1 - corr_mat.fillna(0)) / 2.0) ** 0.5
    else:
        distance = corr_mat

    silh = pd.Series(dtype="float64")

    for repeat_idx in range(repeat):
        for num_clusters in range(2, max_num_clusters + 1):
            kmeans_ = KMeans(
                n_clusters=num_clusters,
                n_init=10,
                random_state=random_state + repeat_idx,
            )
            kmeans_ = kmeans_.fit(distance)
            silh_ = silhouette_samples(distance, kmeans_.labels_)

            stat = (silh_.mean() / silh_.std(), silh.mean() / silh.std())

            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh = silh_
                kmeans = kmeans_

    new_idx = np.argsort(kmeans.labels_)

    corr1 = corr_mat.iloc[new_idx]
    corr1 = corr1.iloc[:, new_idx]

    clusters = {
        i: corr_mat.columns[np.where(kmeans.labels_ == i)[0]].tolist()
        for i in np.unique(kmeans.labels_)
    }

    silh = pd.Series(silh, index=distance.index)

    return corr1, clusters, silh


def _check_improve_clusters(
    new_tstat_mean: float,
    mean_redo_tstat: float,
    old_cluster: tuple,
    new_cluster: tuple,
) -> tuple:
    """
    Checks cluster improvement condition based on t-statistic.

    :param new_tstat_mean: (float) T-statistic of new clusters.
    :param mean_redo_tstat: (float) Average t-statistic for cluster improvement.
    :param old_cluster: (tuple) Old cluster result.
    :param new_cluster: (tuple) New cluster result.
    :return: (tuple) Selected cluster result.
    """
    if new_tstat_mean > mean_redo_tstat:
        return old_cluster
    return new_cluster


def cluster_kmeans_top(
    corr_mat: pd.DataFrame,
    repeat: int = 10,
    random_state: int = 42,
    is_distance_matrix: bool = False,
) -> pd.DataFrame | dict | pd.Series | bool:
    """
    Improve initial clustering by leaving high-quality clusters unchanged and
    modifying below-average clusters.

    :param corr_mat: (pd.DataFrame) Correlation matrix.
    :param repeat: (int) Number of clustering algorithm repetitions.
    :param random_state: (int) Random seed for reproducibility.
    :param is_distance_matrix: (bool) If True, corr_mat is already a distance matrix.
    :return: (tuple) [correlation matrix, optimized clusters, silh scores].
    """
    max_num_clusters = (
        min(
            corr_mat.drop_duplicates().shape[0],
            corr_mat.drop_duplicates().shape[1],
        )
        - 1
    )

    corr1, clusters, silh = _cluster_kmeans_base(
        corr_mat,
        max_num_clusters=max_num_clusters,
        repeat=repeat,
        random_state=random_state,
        is_distance_matrix=is_distance_matrix,
    )

    cluster_quality = {
        i: (
            float("inf")
            if np.std(silh[clusters[i]]) == 0
            else np.mean(silh[clusters[i]]) / np.std(silh[clusters[i]])
        )
        for i in clusters.keys()
    }

    avg_quality = np.mean(list(cluster_quality.values()))
    redo_clusters = [i for i in cluster_quality if cluster_quality[i] < avg_quality]

    if len(redo_clusters) <= 2:
        return corr1, clusters, silh

    keys_redo = []
    for i in redo_clusters:
        keys_redo.extend(clusters[i])

    corr_tmp = corr_mat.loc[keys_redo, keys_redo]
    mean_redo_tstat = np.mean([cluster_quality[i] for i in redo_clusters])

    _, top_clusters, _ = cluster_kmeans_top(
        corr_tmp,
        repeat=repeat,
        random_state=random_state,
    )

    corr_new, clusters_new, silh_new = _improve_clusters(
        corr_mat,
        {i: clusters[i] for i in clusters.keys() if i not in redo_clusters},
        top_clusters,
    )

    new_tstat_mean = np.mean(
        [
            np.mean(silh_new[clusters_new[i]]) / np.std(silh_new[clusters_new[i]])
            for i in clusters_new
        ]
    )

    return _check_improve_clusters(
        new_tstat_mean,
        mean_redo_tstat,
        (corr1, clusters, silh),
        (corr_new, clusters_new, silh_new),
    )


def get_onc_clusters(
    corr_mat: pd.DataFrame,
    repeat: int = 10,
    random_state: int = 42,
    is_distance_matrix: bool = False,
) -> pd.DataFrame | dict | pd.Series:
    """
    Optimal Number of Clusters (ONC) algorithm.

    :param corr_mat: (pd.DataFrame) Correlation matrix of features.
    :param repeat: (int) Number of clustering algorithm repetitions.
    :param random_state: (int) Random seed for reproducibility.
    :param is_distance_matrix: (bool) If False, corr_mat is treated as correlation.
    :return: (tuple) [correlation matrix, optimized clusters, silh scores].
    """
    return cluster_kmeans_top(
        corr_mat,
        repeat=repeat,
        random_state=random_state,
        is_distance_matrix=is_distance_matrix,
    )
