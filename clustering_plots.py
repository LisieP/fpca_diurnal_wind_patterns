"""
Clustering figure functions for the J3 FPCA manuscript.

Produces:
  Fig. 14a  plot_elbow_silhouette_with_piecewise  (FPCA features, k_range 2-20)
  Fig. 14b  plot_elbow_silhouette_with_piecewise  (CF feature,   k_range 2-20)
  Fig. 15   plot_daily_profiles_grid              (FPCA clusters)
  Fig. 16   plot_silhouettes_daily                (FPCA clusters)
  Fig. 18   plot_daily_profiles_grid              (CF clusters)
  Fig. 19   plot_silhouettes_daily                (CF clusters)
  + plot_cluster_feature_boxplots (cells 132 / 142)

Spatial cluster-assignment maps (Figs. 17, 20) are produced separately.
"""

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 14a / 14b — Elbow + silhouette with piecewise linear fit
# ─────────────────────────────────────────────────────────────────────────────

def plot_elbow_silhouette_with_piecewise(
    X,
    k_range=range(2, 15),
    save_path=None,
    figsize=(10, 6),
    show_piecewise: bool = True,
    sil_cluster: int = None,
) -> dict:
    """
    Elbow curve + silhouette bars with optional piecewise linear annotation.

    Exact port from cell 111 of the original notebook.

    Parameters
    ----------
    X : array-like  (N_samples, N_features)
    k_range : range
    save_path : Path or str, optional  — saved as PNG when provided
    figsize : tuple
    show_piecewise : bool  draw red dashed lines and print elbow diagnostics
    sil_cluster : int, optional  highlight this k with a black triangle + hline

    Returns
    -------
    dict with keys: wcss, silhouette, k_range, and optionally
        elbow_k, intersection_k, reduced_range, optimal_k, reduced_silhouette
    """
    wcss, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(X)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    max_wcss = max(wcss)
    wcss_norm = np.array(wcss) / max_wcss

    # ── piecewise elbow detection ────────────────────────────────────────────
    if show_piecewise:
        k_arr = np.array(list(k_range))

        def _elbow(k_vals, wcss_vals):
            p1, p2 = np.array([k_vals[0], wcss_vals[0]]), np.array([k_vals[-1], wcss_vals[-1]])
            dists = [
                np.abs(np.cross(p2 - p1, p1 - np.array([k_vals[i], wcss_vals[i]])))
                / np.linalg.norm(p2 - p1)
                for i in range(len(k_vals))
            ]
            return np.argmax(dists), dists

        elbow_idx, _ = _elbow(k_arr, wcss_norm)
        elbow_k = k_arr[elbow_idx]

        lk, lw = k_arr[: elbow_idx + 1], wcss_norm[: elbow_idx + 1]
        rk, rw = k_arr[elbow_idx:], wcss_norm[elbow_idx:]
        ls, li, *_ = stats.linregress(lk, lw)
        rs, ri, *_ = stats.linregress(rk, rw)

        if ls != rs:
            ix = (ri - li) / (ls - rs)
            iy = ls * ix + li
        else:
            ix, iy = elbow_k, wcss_norm[elbow_idx]

        rl = max(k_range.start, int(np.floor(ix)) - 2)
        rh = min(k_range.stop - 1, int(np.ceil(ix)) + 2)
        reduced_range = range(rl, rh + 1)

        red_sil = []
        for k in reduced_range:
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
            km.fit(X)
            red_sil.append(silhouette_score(X, km.labels_))

        opt_k = list(reduced_range)[int(np.argmax(red_sil))]

        print(f"Elbow point detected at k = {elbow_k}")
        print(f"Line intersection at k = {ix:.2f}")
        print(f"Reduced range: {rl} – {rh}")
        print(f"Optimal k (silhouette within reduced range): {opt_k}")
        print(f"Silhouette scores within reduced range: {dict(zip(reduced_range, red_sil))}")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(
        k_range, wcss_norm,
        color="black", linestyle="-", marker="o",
        markerfacecolor="blue", markeredgecolor="blue",
        markersize=8, linewidth=2, zorder=3,
        label="Within cluster sum of squares",
    )
    ax1.set_xlabel("Number of clusters", fontsize=20)
    ax1.set_ylabel("Normalised within cluster sum of squares [pu]", fontsize=19.2)
    ax1.tick_params(axis="both", labelsize=16)
    ax1.set_ylim(0, 1.05)
    ax1.spines["bottom"].set_linewidth(2)
    ax1.spines["left"].set_linewidth(2)
    ax1.grid(True, linestyle="-", linewidth=0.5)

    if show_piecewise:
        lx = np.linspace(k_range.start, ix + 2, 100)
        ax1.plot(lx, ls * lx + li, color="red", linestyle="--", linewidth=2, alpha=0.7, zorder=2)
        rx = np.linspace(ix - 2, k_range.stop - 1, 100)
        ax1.plot(rx, rs * rx + ri, color="red", linestyle="--", linewidth=2, alpha=0.7, zorder=2)

    ax1.bar(
        np.array(k_range), np.array(sil_scores),
        width=0.6, color="gray", alpha=0.4, label="Average silhouette width", zorder=1,
    )

    ax2 = ax1.twinx()
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Average silhouette width", fontsize=20)
    ax2.tick_params(axis="y", labelsize=16)

    if sil_cluster is not None:
        k_list = list(k_range)
        if sil_cluster in k_list:
            idx = k_list.index(sil_cluster)
            sv = sil_scores[idx]
            ax1.scatter([sil_cluster], [sv], color="black", s=150, zorder=5,
                        marker="^", edgecolor="black", linewidth=0)
            ax1.axhline(y=sv, color="grey", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
            print(f"Highlighted cluster {sil_cluster} with silhouette score: {sv:.4f}")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    seen, all_h, all_l = set(), [], []
    for h, l in zip(h1 + h2, l1 + l2):
        if l not in seen and l != "":
            seen.add(l)
            all_h.append(h)
            all_l.append(l)
    ax1.legend(all_h, all_l, loc="upper right", fontsize=16, frameon=True)

    plt.xticks(np.arange(k_range.start, k_range.stop, 1))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    print("Silhouette scores:", sil_scores)

    results = {"wcss": wcss_norm, "silhouette": sil_scores, "k_range": list(k_range)}
    if show_piecewise:
        results.update(
            {
                "elbow_k": elbow_k,
                "intersection_k": ix,
                "reduced_range": list(reduced_range),
                "optimal_k": opt_k,
                "reduced_silhouette": dict(zip(reduced_range, red_sil)),
            }
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 15 / 18 — Daily profiles grid per cluster
# ─────────────────────────────────────────────────────────────────────────────

def plot_daily_profiles_grid(
    wind_df,
    id_df,
    clusters_per_row: int = 3,
    figsize=(18, 12),
    colors=None,
    save_path=None,
):
    """
    Grid of mean daily profiles, one subplot per cluster.

    Exact port from cell 113 of the original notebook.

    Parameters
    ----------
    wind_df : DataFrame (N_hours, N_locations)
    id_df : DataFrame with columns 'ID' and 'Cluster'
    clusters_per_row : int
    figsize : tuple
    colors : ignored (kept for API compat); palette is Spectral_r internally
    save_path : Path or str, optional

    Returns
    -------
    fig, axes
    """
    clusters = sorted(id_df["Cluster"].unique())
    n_clusters = len(clusters)
    n_cols = min(clusters_per_row, n_clusters)
    n_rows = ceil(n_clusters / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else np.array([axes]).flatten()

    cluster_colors = plt.cm.Spectral_r(np.linspace(0.1, 0.8, n_clusters))
    c2col = {cl: cluster_colors[i] for i, cl in enumerate(clusters)}

    y_min, y_max = 1.0, 0.0

    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        cluster_ids = id_df[id_df["Cluster"] == cluster]["ID"].tolist()

        all_vals = []
        for loc_id in cluster_ids:
            if loc_id not in wind_df.columns:
                continue
            hp = wind_df[loc_id].groupby(wind_df.index.hour).mean()
            ax.plot(hp.index, hp.values, color=c2col[cluster], linewidth=0.8, alpha=1)
            all_vals.extend(hp.values)

        if all_vals:
            y_min = min(y_min, np.min(all_vals))
            y_max = max(y_max, np.max(all_vals))

        sample_ids = cluster_ids[:50]
        sample_profiles = [
            wind_df[lid].groupby(wind_df.index.hour).mean().values
            for lid in sample_ids
            if lid in wind_df.columns
        ]
        if sample_profiles:
            ax.plot(
                range(24), np.mean(sample_profiles, axis=0),
                color="black", linewidth=2.5, linestyle="--", label="Mean",
            )

        ax.set_title(f"Cluster {cluster}", fontsize=26, fontweight="bold")
        ax.grid(True, alpha=0.5, axis="y", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.2, axis="x", linestyle="--", linewidth=0.5)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(0, 23)
        ax.tick_params(axis="both", labelsize=21)
        ax.legend(loc="upper left", fontsize=22, handlelength=1.2)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)
        ax.spines["top"].set_linewidth(0)
        ax.spines["right"].set_linewidth(0)

    y_margin = (y_max - y_min) * 0.05
    for i in range(len(clusters)):
        axes[i].set_ylim(y_min - y_margin, y_max + y_margin)

    for i in range(len(clusters), len(axes)):
        axes[i].set_visible(False)

    fig.text(0.02, 0.5, "Normalised wind power [pu]",
             va="center", rotation="vertical", fontsize=34)
    fig.text(0.5, 0.02, "Time of day [Hr]", ha="center", fontsize=34)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.98])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 16 / 19 — Silhouette plot over mean daily profiles
# ─────────────────────────────────────────────────────────────────────────────

def plot_silhouettes_daily(
    results: dict,
    figsize=(10, 6),
    xlim=(-0.6, 0.7),
    save_path=None,
):
    """
    Horizontal silhouette-width plot (grey bars + per-cluster red dot + global line).

    Exact port from cell 116 of the original notebook.

    Parameters
    ----------
    results : dict  from clustering.get_distances_between_daily_profiles
    figsize, xlim : matplotlib params
    save_path : Path or str, optional

    Returns
    -------
    fig, ax
    """
    silhouette_df = results["silhouette_df"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    y_lower = 10
    unique_clusters = sorted(silhouette_df["Cluster"].unique())

    for cluster in unique_clusters:
        c_sil = silhouette_df[silhouette_df["Cluster"] == cluster][
            "Silhouette_Euclidean"
        ].values.copy()
        c_sil.sort()
        c_mean = c_sil.mean()
        size_c = len(c_sil)
        y_upper = y_lower + size_c

        ax.barh(
            range(y_lower, y_upper), c_sil,
            height=1.0, color="#cccccc", edgecolor="#666666", linewidth=0.5,
        )
        ax.plot(c_mean, (y_lower + y_upper) / 2, "ro", markersize=5, markerfacecolor="red")
        print(f"Mean for cluster {cluster}: {c_mean}")
        ax.text(-0.02, y_lower + 0.5 * size_c, str(cluster),
                va="center", ha="right", fontsize=16)
        y_lower = y_upper + 10

    global_sil = results["global_silhouette"]["euclidean"]
    ax.axvline(x=global_sil, color="#d62728", linestyle="--", linewidth=2,
               label=f"Overall average: {global_sil:.3f}")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Silhouette width", fontsize=20)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.1, 0.1))
    ax.set_ylabel("Cluster", fontsize=20)
    ax.set_xlim(xlim)
    ax.set_ylim([0, y_lower])
    ax.grid(True, alpha=0.8, axis="both", linestyle="--", linewidth=0.8)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=8, label="Cluster mean"),
        Line2D([0], [0], color="#d62728", linestyle="--", linewidth=2,
               label=f"Overall mean: {global_sil:.3f}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True,
              fontsize=17, framealpha=0.6)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# TOU feature boxplots (cells 132 / 142)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_feature_boxplots(
    df,
    feature_cols: list,
    feature_labels: list = None,
    colors=None,
    save_path=None,
    filename: str = "cluster_boxplots.png",
    ylabel: str = "Value",
    figsize=(14, 6),
    tick_fontsize: int = 12,
    legend_fontsize: int = 16,
    label_fontsize: int = 20,
):
    """
    Side-by-side boxplots per cluster for each feature in feature_cols.

    Exact port from cell 126 of the original notebook.

    Parameters
    ----------
    df : DataFrame with 'Cluster' column + feature columns
    feature_cols : list of str
    feature_labels : list of str, optional (defaults to feature_cols)
    colors : list, optional
    save_path : Path or str, optional  (file saved as save_path/filename)
    ylabel : str
    figsize, tick_fontsize, legend_fontsize, label_fontsize : matplotlib params

    Returns
    -------
    fig, ax
    """
    clusters = sorted(df["Cluster"].unique())
    n_clusters = len(clusters)
    n_features = len(feature_cols)

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_features))
    if feature_labels is None:
        feature_labels = feature_cols

    fig, ax = plt.subplots(figsize=figsize)

    for f_idx, (feature, color) in enumerate(zip(feature_cols, colors)):
        positions = [
            cl + (f_idx - (n_features - 1) / 2) * 0.3
            for cl in range(1, n_clusters + 1)
        ]
        feature_data = [
            df[df["Cluster"] == cl][feature].dropna().values for cl in clusters
        ]
        bp = ax.boxplot(
            feature_data,
            positions=positions,
            widths=0.2,
            boxprops=dict(color=color, linewidth=1.5),
            whiskerprops=dict(color=color, linestyle="--", linewidth=1.5),
            capprops=dict(color=color, linewidth=1.5),
            medianprops=dict(color=color, linewidth=2),
            meanprops=dict(marker="o", markersize=5, markeredgewidth=1.5,
                           markeredgecolor="red", markerfacecolor="red"),
            showmeans=True,
            patch_artist=False,
            showfliers=False,
        )

    ax.set_xlabel("Cluster", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xticks(range(1, n_clusters + 1))
    ax.set_xticklabels(clusters, fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.7)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["top"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)

    legend_elements = [
        Line2D([0], [0], color=c, lw=2, label=l)
        for c, l in zip(colors, feature_labels)
    ]
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markeredgecolor="red", markersize=8, label="Mean")
    )
    ax.legend(handles=legend_elements, fontsize=legend_fontsize,
              loc="upper left", frameon=True, framealpha=0.95)

    plt.tight_layout()

    if save_path:
        fig.savefig(Path(save_path) / filename, bbox_inches="tight", dpi=300)

    plt.show()
    return fig, ax
