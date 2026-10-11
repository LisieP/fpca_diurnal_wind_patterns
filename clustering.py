"""
Clustering computations for the J3 FPCA manuscript.

All K-means calls use n_init=50, random_state=42, init='k-means++'
exactly matching cell 109 of the original notebook.
"""

import gc

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


# ─────────────────────────────────────────────────────────────────────────────
# Feature construction
# ─────────────────────────────────────────────────────────────────────────────

def build_cf_features(wind_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the capacity-factor feature matrix for clustering.

    Replicates cell 137: mean_df = wind_df.T.mean(axis=1).to_frame('Mean_Generation').

    Returns
    -------
    mean_df : DataFrame, shape (N_locations, 1)
        Index = location IDs (index.name = 'ID').
    """
    mean_df = wind_df.T.mean(axis=1).to_frame(name="Mean_Generation")
    mean_df.index.name = "ID"
    return mean_df


def build_fpca_features(mean_pc_df: pd.DataFrame, n_modes: int = 4) -> pd.DataFrame:
    """
    Select the first n_modes PC mean-score columns for clustering.

    Replicates cell 130: X = mean_pc_df[["PC1","PC2","PC3","PC4"]].

    Parameters
    ----------
    mean_pc_df : DataFrame  from fpca.mean_pc_scores (index = location IDs)
    n_modes : int           number of PC columns to use (default 4)

    Returns
    -------
    DataFrame, shape (N_locations, n_modes)
    """
    cols = [f"PC{i + 1}" for i in range(n_modes)]
    return mean_pc_df[cols]


# ─────────────────────────────────────────────────────────────────────────────
# K-means clustering
# ─────────────────────────────────────────────────────────────────────────────

def perform_clustering(
    data: pd.DataFrame,
    n_clusters: int,
    id_df: pd.DataFrame = None,
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    K-means clustering on data with the manuscript's fixed settings.

    Exact port of the function in cell 109 of the original notebook.
    Uses KMeans(init='k-means++', n_init=50, random_state=42).

    Parameters
    ----------
    data : DataFrame
        Input data; index must match id_df if provided.
    n_clusters : int
    id_df : DataFrame, optional
        DataFrame whose index receives the 'Cluster' column.
    feature_cols : list, optional
        Columns of data to use; all columns used when None.

    Returns
    -------
    DataFrame : id_df with 'Cluster' column (1-based labels), or a
                bare cluster-label DataFrame when id_df is None.
    """
    if isinstance(data, pd.DataFrame):
        X = data[feature_cols].values if feature_cols else data.values
        data_index = data.index.copy()
    else:
        X = data
        data_index = None

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=50,
        random_state=42,
    )
    cluster_labels = kmeans.fit_predict(X) + 1  # 1-based

    if id_df is not None:
        result_df = id_df.copy()
        if data_index is not None:
            if not result_df.index.equals(data_index):
                cluster_series = pd.Series(index=data_index, data=cluster_labels)
                common = result_df.index.intersection(data_index)
                result_df["Cluster"] = cluster_series.loc[common]
            else:
                result_df["Cluster"] = cluster_labels
        else:
            result_df["Cluster"] = cluster_labels

        if feature_cols and isinstance(data, pd.DataFrame):
            for col in feature_cols:
                result_df[f"clustered_on {col}"] = data[col].values
        return result_df

    return pd.DataFrame(
        {"Cluster": cluster_labels},
        index=data_index if data_index is not None else range(len(cluster_labels)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cluster statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_mean_gen_per_cluster(id_df: pd.DataFrame, wind_data: pd.DataFrame) -> pd.DataFrame:
    """
    Mean normalised wind power per cluster.

    Exact port from cell 109.

    Parameters
    ----------
    id_df : DataFrame with columns 'ID' and 'Cluster'
    wind_data : DataFrame (N_hours, N_locations)

    Returns
    -------
    DataFrame with columns: Cluster, Mean_Generation, Count, IDs
    """
    results = []
    for cluster in sorted(id_df["Cluster"].unique()):
        cluster_ids = id_df[id_df["Cluster"] == cluster]["ID"].tolist()
        available = [c for c in cluster_ids if c in wind_data.columns]
        mean_gen = wind_data[available].mean().mean() if available else np.nan
        results.append(
            {
                "Cluster": cluster,
                "Mean_Generation": mean_gen,
                "Count": len(cluster_ids),
                "IDs": cluster_ids,
            }
        )
    return pd.DataFrame(results)


def sort_clusters(
    id_df: pd.DataFrame,
    cluster_stats_df: pd.DataFrame,
    sort_by: str = "Mean_Generation",
) -> tuple:
    """
    Relabel clusters in ascending order of sort_by.

    Exact port from cell 109.

    Returns
    -------
    (reordered_id_df, reordered_cluster_stats_df)
    """
    id_df_copy = id_df.copy()
    stats_copy = cluster_stats_df.copy()

    sorted_stats = stats_copy.sort_values(by=sort_by).reset_index(drop=True)
    cluster_mapping = {
        old: new
        for new, old in enumerate(sorted_stats["Cluster"].tolist(), start=1)
    }

    id_df_copy["Cluster"] = id_df_copy["Cluster"].map(cluster_mapping)
    sorted_stats["Cluster"] = sorted_stats["Cluster"].map(cluster_mapping)
    sorted_stats = sorted_stats.sort_values("Cluster")

    mean_columns = [
        c for c in sorted_stats.columns if c not in ("Cluster", "Count", "IDs")
    ]
    cluster_means = {
        row["Cluster"]: {col: row[col] for col in mean_columns}
        for _, row in sorted_stats.iterrows()
    }

    for mean_col in mean_columns:
        feat = mean_col.replace("Mean_", "") if mean_col.startswith("Mean_") else mean_col
        new_col = f"cluster_mean_{feat}"
        id_df_copy[new_col] = id_df_copy["Cluster"].map(
            lambda x, mc=mean_col: cluster_means[x][mc]
        )

    return id_df_copy, sorted_stats


# ─────────────────────────────────────────────────────────────────────────────
# Stability study
# ─────────────────────────────────────────────────────────────────────────────

def cluster_stability(
    data,
    n_clusters: int,
    feature_cols=None,
    n_runs: int = 200,
    n_init_values=(1, 5, 10, 20, 50),
    seed_offset: int = 1000,
) -> tuple:
    """
    Pairwise ARI stability study over multiple n_init values.

    Exact port from cell 109 of the original notebook.
    Default n_runs=200, seed_offset=1000, n_init_values=(1,5,10,20,50).

    Parameters
    ----------
    data : DataFrame or ndarray
    n_clusters : int
    feature_cols : list, optional
    n_runs : int               number of independent runs per n_init
    n_init_values : tuple      values of n_init to compare
    seed_offset : int          first run uses random_state = seed_offset + 0

    Returns
    -------
    summary : DataFrame  one row per n_init value with ARI and inertia stats
    labels : dict        {n_init: (n_runs, n_samples) label array}
    """
    if isinstance(data, pd.DataFrame):
        X = data[feature_cols].values if feature_cols else data.values
    else:
        X = np.asarray(data)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    records, labels = [], {}

    for n_init in n_init_values:
        run_labels = np.empty((n_runs, X.shape[0]), dtype=int)
        inertias = np.empty(n_runs)

        for r in range(n_runs):
            km = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=n_init,
                random_state=seed_offset + r,
            )
            run_labels[r] = km.fit_predict(X)
            inertias[r] = km.inertia_

        aris = [
            adjusted_rand_score(run_labels[i], run_labels[j])
            for i in range(n_runs)
            for j in range(i + 1, n_runs)
        ]
        labels[n_init] = run_labels
        records.append(
            {
                "n_init": n_init,
                "ari_mean": np.mean(aris),
                "ari_min": np.min(aris),
                "ari_frac_above_0.99": np.mean(np.array(aris) > 0.99),
                "inertia_mean": inertias.mean(),
                "inertia_std": inertias.std(),
                "n_distinct_solutions": len(np.unique(np.round(inertias, 6))),
            }
        )

    return pd.DataFrame(records), labels


# ─────────────────────────────────────────────────────────────────────────────
# Silhouette over mean daily profiles
# ─────────────────────────────────────────────────────────────────────────────

def get_distances_between_daily_profiles(
    id_df: pd.DataFrame,
    wind_df: pd.DataFrame,
) -> dict:
    """
    Intra-cluster distances and silhouette scores on mean daily profiles.

    Exact port from cell 115 of the original notebook.

    Parameters
    ----------
    id_df : DataFrame with columns 'ID' and 'Cluster'
    wind_df : DataFrame (N_hours, N_locations)

    Returns
    -------
    dict with keys:
        cluster_stats   DataFrame
        silhouette_df   DataFrame (per-location silhouette values)
        global_silhouette  dict {'euclidean': float, 'cosine': float}
        n_locations, n_clusters : int
    """
    print("Calculating daily profiles for each location...")
    daily_profiles = {}
    for loc_id in wind_df.columns:
        if loc_id not in id_df["ID"].values:
            continue
        daily_profiles[loc_id] = (
            wind_df[loc_id].groupby(wind_df.index.hour).mean().values
        )

    all_ids, all_profiles, all_clusters = [], [], []
    for loc_id, profile in daily_profiles.items():
        cluster = id_df[id_df["ID"] == loc_id]["Cluster"].values[0]
        all_ids.append(loc_id)
        all_profiles.append(profile)
        all_clusters.append(cluster)

    X = np.array(all_profiles)
    y = np.array(all_clusters)

    print("\nCalculating intra-cluster distances...")
    cluster_stats = {}
    for cluster in np.unique(y):
        mask = y == cluster
        cluster_profiles = X[mask]
        n = len(cluster_profiles)

        if n < 2:
            cluster_stats[cluster] = {
                "count": n,
                "avg_euclidean_distance": np.nan,
                "std_euclidean_distance": np.nan,
                "avg_cosine_distance": np.nan,
                "std_cosine_distance": np.nan,
                "avg_pairwise_correlation": np.nan,
            }
            continue

        euc_mat = euclidean_distances(cluster_profiles)
        uidx = np.triu_indices_from(euc_mat, k=1)
        euc_flat = euc_mat[uidx]
        del euc_mat
        gc.collect()

        cos_mat = cosine_distances(cluster_profiles)
        cos_flat = cos_mat[uidx]
        del cos_mat
        gc.collect()

        from scipy.stats import pearsonr
        corrs = [
            pearsonr(cluster_profiles[i], cluster_profiles[j])[0]
            for i in range(n)
            for j in range(i + 1, n)
        ]

        cluster_stats[cluster] = {
            "count": n,
            "avg_euclidean_distance": np.mean(euc_flat),
            "std_euclidean_distance": np.std(euc_flat),
            "avg_cosine_distance": np.mean(cos_flat),
            "std_cosine_distance": np.std(cos_flat),
            "avg_pairwise_correlation": np.mean(corrs),
            "std_pairwise_correlation": np.std(corrs),
        }
        print(
            f"Cluster {cluster}: {n} profiles, "
            f"Avg Euclidean dist: {cluster_stats[cluster]['avg_euclidean_distance']:.4f}, "
            f"Avg Cosine dist: {cluster_stats[cluster]['avg_cosine_distance']:.4f}"
        )

    print("\nCalculating silhouette scores...")
    sil_euc = silhouette_score(X, y, metric="euclidean")
    sil_cos = silhouette_score(X, y, metric="cosine")
    sil_samples_euc = silhouette_samples(X, y, metric="euclidean")
    sil_samples_cos = silhouette_samples(X, y, metric="cosine")

    silhouette_df = pd.DataFrame(
        {
            "ID": all_ids,
            "Cluster": y,
            "Silhouette_Euclidean": sil_samples_euc,
            "Silhouette_Cosine": sil_samples_cos,
        }
    )
    for cluster in np.unique(y):
        m = silhouette_df["Cluster"] == cluster
        silhouette_df.loc[m, "Cluster_Avg_Silhouette_Euclidean"] = (
            silhouette_df.loc[m, "Silhouette_Euclidean"].mean()
        )
        silhouette_df.loc[m, "Cluster_Avg_Silhouette_Cosine"] = (
            silhouette_df.loc[m, "Silhouette_Cosine"].mean()
        )

    results = {
        "cluster_stats": pd.DataFrame(cluster_stats).T,
        "silhouette_df": silhouette_df,
        "global_silhouette": {"euclidean": sil_euc, "cosine": sil_cos},
        "n_locations": len(all_ids),
        "n_clusters": len(np.unique(y)),
    }

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Number of locations: {results['n_locations']}")
    print(f"Number of clusters: {results['n_clusters']}")
    print(f"\nGlobal Silhouette Scores:")
    print(f"  - Euclidean: {sil_euc:.4f}")
    print(f"  - Cosine: {sil_cos:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stability cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_or_load_stability(
    data,
    n_clusters: int,
    feature_cols,
    cache_path,
    run: bool = False,
    n_runs: int = 200,
    n_init_values=(1, 5, 10, 20, 50),
    seed_offset: int = 1000,
) -> tuple:
    """
    Return (summary_df, labels_dict) from cache or by running the study.

    Parameters
    ----------
    cache_path : Path   base path; summary saved as <cache_path>_summary.csv
    run : bool          if True, recompute even when cache exists
    """
    import pickle
    from pathlib import Path

    summary_path = Path(str(cache_path) + "_summary.csv")
    labels_path = Path(str(cache_path) + "_labels.pkl")

    if not run and summary_path.exists() and labels_path.exists():
        summary = pd.read_csv(summary_path)
        with open(labels_path, "rb") as fh:
            labels = pickle.load(fh)
        return summary, labels

    summary, labels = cluster_stability(
        data,
        n_clusters=n_clusters,
        feature_cols=feature_cols,
        n_runs=n_runs,
        n_init_values=n_init_values,
        seed_offset=seed_offset,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    with open(labels_path, "wb") as fh:
        import pickle as _pkl
        _pkl.dump(labels, fh)

    return summary, labels
