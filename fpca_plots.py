"""
FPCA figures for the J3 manuscript.

Each function corresponds to one or more manuscript figures.
No top-level execution; call from main.ipynb.
"""

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 4 – Scree plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_scree(
    r2_ind: np.ndarray,
    r2_cum: np.ndarray,
    n_plot: int = 10,
    save_path=None,
):
    """
    Plot individual and cumulative R² about the mean daily profile.

    Manuscript Fig. 4.  Implements the live code of cell 20.

    Parameters
    ----------
    r2_ind : ndarray  individual R² increments (from fpca.r2_about_mean)
    r2_cum : ndarray  cumulative R²
    n_plot : int  number of components to show (10 in the paper)
    save_path : path-like or None

    Returns
    -------
    fig, ax
    """
    comp = np.arange(1, n_plot + 1)
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(comp, r2_cum[:n_plot], marker="s", label="Cumulative")
    ax.plot(comp, r2_ind[:n_plot], marker="o", label="Individual contribution")

    ax.set_xlabel("Number of components", fontsize=24)
    ax.set_ylabel("$R^2$", fontsize=24)
    ax.set_xticks(comp)
    ax.set_ylim(-0.01, 1.0)
    ax.legend(loc="center right", fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 5 – PC loadings
# ─────────────────────────────────────────────────────────────────────────────

def plot_pc_loadings(
    components: np.ndarray,
    modes=(1, 2, 3, 4),
    colors=None,
    markers=None,
    ylim=(-0.38, 0.38),
    linewidth=1.7,
    markersize=7,
    ncols_legend=4,
    save_path=None,
):
    """
    Plot selected PC loading vectors against hour of day.

    Manuscript Fig. 5.  Wraps pca.plot_pca_modes with the same arguments
    used in cell 23 of the original notebook.

    Parameters
    ----------
    components : ndarray, shape (K, 24)
    modes : sequence of int  1-based PC numbers to plot
    save_path : path-like or None

    Returns
    -------
    fig, ax
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from pca import plot_pca_modes

    fig, ax = plot_pca_modes(
        components=components,
        modes=list(modes),
        colors=colors or plt.rcParams["axes.prop_cycle"].by_key()["color"],
        markers=markers or ["P", "^", "o", "D", "s", "p", "*"],
        ylim=ylim,
        linewidth=linewidth,
        markersize=markersize,
        ncols_legend=ncols_legend,
    )
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 6 – Mean ± σ_q · PC_q  (Ramsay & Silverman style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pc_perturbations(
    X: np.ndarray,
    scores: np.ndarray,
    components: np.ndarray,
    explained=None,
    n_plot: int = 4,
    c: float = 1.0,
    hours=None,
    colors=None,
    save_path=None,
):
    """
    Plot the mean daily profile with ± c * SD(score_k) * PC_k for each
    retained mode (Ramsay & Silverman convention).

    Manuscript Fig. 6.  Exact port of the function defined and called in
    cell 27 of the original notebook.

    Parameters
    ----------
    X : ndarray, shape (N*D, 24)  stacked daily profiles
    scores : ndarray, shape (N*D, K)  PC scores
    components : ndarray, shape (K, 24) or (24, K)
    explained : ndarray or None  individual R² in % (used for subplot titles)
    n_plot : int  number of PCs to show
    c : float  multiplier on the SD
    save_path : path-like or None

    Returns
    -------
    fig, axes
    """
    X = np.asarray(X)
    scores = np.asarray(scores)
    comps = np.asarray(components)
    if comps.shape[1] != X.shape[1]:
        comps = comps.T

    mean_curve = X.mean(axis=0)
    sd = scores.std(axis=0, ddof=1)
    hours = np.arange(X.shape[1]) if hours is None else hours
    _colors = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ncols = 2
    nrows = int(np.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows),
                             sharex=True, sharey=True)

    for k, ax in enumerate(np.atleast_1d(axes).flat):
        if k >= n_plot:
            ax.axis("off")
            continue
        delta = c * sd[k] * comps[k]
        ax.plot(hours, mean_curve, "k-", lw=1.2, label="Mean")
        ax.plot(hours, mean_curve + delta, ls="none", marker="+",
                ms=10, mew=2.5, color=_colors[k], label=f"Mean + {c:g} SD")
        ax.plot(hours, mean_curve - delta, ls="none", marker="_",
                ms=10, mew=2.5, color=_colors[k], label=f"Mean - {c:g} SD")

        title = f"PC{k + 1}"
        if explained is not None:
            title += f" ({explained[k]:.1f}%)"
        ax.set_title(title, fontsize=26)
        ax.set_xlabel("Time of day [Hr]", fontsize=24)
        ax.set_xticks(np.arange(0, 24, 2))
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(True, color="grey", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    fig.supylabel("Normalised wind power [pu]", fontsize=28)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 7 – Per-year PCs vs pooled PCs
# ─────────────────────────────────────────────────────────────────────────────

def plot_components_by_year(
    pooled: np.ndarray,
    year_components: dict,
    year_r2=None,
    pooled_r2=None,
    n_plot: int = 4,
    hours=None,
    colors=None,
    sharey: bool = True,
    save_path=None,
):
    """
    2×2 grid: each panel shows one PC; pooled (dashed black) plus each year
    coloured line.

    Manuscript Fig. 7.  Exact port of the function defined and called in
    cell 26 of the original notebook.

    Parameters
    ----------
    pooled : ndarray, shape (K, 24)  five-year components
    year_components : dict  {year: (K, 24)}  from fpca.per_year_decomposition
    year_r2 : dict or None  {year: (K,)} individual R² per year
    pooled_r2 : ndarray or None  individual R² for the pooled record
    n_plot : int
    save_path : path-like or None

    Returns
    -------
    fig, axes
    """
    def _unit(C):
        C = np.asarray(C)
        return C / np.linalg.norm(C, axis=1, keepdims=True)

    P = _unit(np.asarray(pooled))
    years = sorted(year_components)
    hours = np.arange(P.shape[1]) if hours is None else hours
    _colors = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["^", "o", "D", "s", "x", "H", "D", "d", "P", "X"]

    ncols = 2
    nrows = int(np.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows),
                             sharex=True, sharey=sharey)

    for k, ax in enumerate(np.atleast_1d(axes).flat):
        if k >= n_plot:
            ax.axis("off")
            continue
        ax.axhline(0, lw=0.6, color="0.7")
        ax.plot(hours, P[k], color="k", ls="--", lw=2.2, zorder=5,
                label="Five-year")
        for i, y in enumerate(years):
            ax.plot(
                hours, year_components[y][k],
                color=_colors[i], lw=1.8, alpha=0.9,
                label=str(y), marker=markers[i],
                markersize=6, markevery=3,
            )
        ax.set_title(f"PC{k + 1}", fontsize=26)
        ax.set_xlabel("Time of day [Hr]", fontsize=24)
        ax.set_xticks(np.arange(0, 24, 2))
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(True, color="grey", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        if k == 0:
            ax.legend(fontsize=19, ncol=2, frameon=True, framealpha=0.9,
                      loc="lower right", handlelength=1.5, handletextpad=0.5)

    fig.supylabel("Component loading", fontsize=28)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 8 – Residual std map
# ─────────────────────────────────────────────────────────────────────────────

def plot_residual_std_map(
    residuals_std,
    id_df,
    upper_cmap,
    rsa_shp_path,
    vmin: float = 0.088,
    markersize: float = 5.8,
    save_path=None,
):
    """
    Spatial map of the std of the Q=4 reconstruction residual per location.

    Manuscript Fig. 8.  Calls map_plotter.plot_spatial_squares with the
    exact arguments from cell 100.

    Parameters
    ----------
    residuals_std : Series, index = location IDs
    id_df : DataFrame with columns [ID, Longitude, Latitude]
    upper_cmap : Colormap  (truncated viridis from cell 99)
    rsa_shp_path : str  path to the Western Cape shapefile
    save_path : path-like or None

    Returns
    -------
    fig, ax
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from map_plotter import plot_spatial_squares

    save_dir = str(save_path.parent) if save_path else None
    save_fname = save_path.name if save_path else None

    fig, ax = plot_spatial_squares(
        values=residuals_std,
        sa_ids_df=id_df,
        cmap=upper_cmap,
        markersize=markersize,
        vmin=vmin,
        cbar_label=r"Standard deviation [pu]",
        plot_path=save_dir,
        fname=save_fname,
        rsa_shp_path=rsa_shp_path,
        rsa_edgecolor="black",
        rsa_linewidth=3.5,
    )
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 9 – Mean score maps PC1–PC4
# ─────────────────────────────────────────────────────────────────────────────

def plot_mean_score_maps(
    centered_pc1_scores_dict: dict,
    id_df,
    rsa_shp_path,
    save_path=None,
):
    """
    2×2 map grid: overall mean PC score per location.  PC1 is centred by the
    regional mean; PC2–PC4 shown on their natural scale.

    Manuscript Fig. 9.  Calls map_plotter.plot_pc_overall_means_own_scale
    with the exact arguments from cell 38.

    Returns fig, axes
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from map_plotter import plot_pc_overall_means_own_scale

    fig, axes = plot_pc_overall_means_own_scale(
        centered_pc1_scores_dict,
        id_df,
        components=[1, 2, 3, 4],
        n_cols=2,
        markersize=0.5,
        cmap="Spectral_r",
        plot_path=str(save_path) if save_path else None,
        rsa_shp_path=rsa_shp_path,
        rsa_edgecolor="black",
        rsa_linewidth=1.5,
        rsa_facecolor="none",
        vmax=3,
        scales_vmax=[1, 1, 1, 1],
        cbar_dict={
            "PC1": "Centered mean score",
            "PC2": "Mean score",
            "PC3": "Mean score",
            "PC4": "Mean score",
        },
    )
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 10 – Seasonal mean score maps (summer DJF / winter JJA)
# ─────────────────────────────────────────────────────────────────────────────

def plot_seasonal_score_maps(
    centered_scores_dict: dict,
    scores_dict: dict,
    id_df,
    rsa_shp_path,
    years=(2014, 2015, 2016, 2017, 2019),
    save_path_pc1=None,
    save_path_template=None,
):
    """
    Summer/winter mean score maps for PC1 (centred) and PC2–PC4.

    Panels (a)+(b) = PC1 summer/winter (centred_scores_dict).
    Panels (c)+(d) = PC2 summer/winter (scores_dict, vmax=1).

    Manuscript Fig. 10.  Implements cell 43 exactly.

    Returns list of (fig, ax) tuples, one per PC.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from map_plotter import plot_summer_winter

    results = []
    years_list = list(years)

    # PC1 centred
    fig, ax = plot_summer_winter(
        centered_scores_dict,
        id_df,
        pc=1,
        years=years_list,
        rsa_shp_path=rsa_shp_path,
        rsa_edgecolor="black",
        rsa_linewidth=1.4,
        rsa_facecolor="none",
        rsa_alpha=1.0,
        markersize=1,
        color_ramp="Spectral_r",
        cbar_label="Centered mean PC1 score",
    )
    if save_path_pc1:
        fig.savefig(save_path_pc1, dpi=300, bbox_inches="tight")
    results.append((fig, ax))

    # PC2, 3, 4 (uncentred)
    for i in [2, 3, 4]:
        fig, ax = plot_summer_winter(
            scores_dict,
            id_df,
            pc=i,
            years=years_list,
            rsa_shp_path=rsa_shp_path,
            rsa_edgecolor="black",
            rsa_linewidth=1.4,
            rsa_facecolor="none",
            rsa_alpha=1.0,
            vmax=1,
            markersize=1,
            color_ramp="Spectral_r",
            cbar_label=f"Mean PC{i} score",
        )
        if save_path_template:
            p = str(save_path_template).replace("{pc}", str(i))
            fig.savefig(p, dpi=300, bbox_inches="tight")
        results.append((fig, ax))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figs. 11a, 11b, 12, 13 – Scatter / regression plots
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_cf(
    cf, y_vals, color_vals,
    xlabel, ylabel, cbar_label,
    save_path=None,
):
    """Internal helper: scatter plot with CF on x-axis, coloured by a PC score."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(cf, y_vals, alpha=0.8, s=10, c=color_vals, cmap="Spectral_r")
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    cbar = fig.colorbar(sc)
    cbar.solids.set_alpha(1)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_fig11a(tou_means, mean_pc_df, save_path=None):
    """
    Evening-peak mean power vs CF, coloured by mean PC1 score.  Fig. 11(a).
    Implements cell 70.
    """
    return _scatter_cf(
        tou_means["capacity_factor"],
        tou_means["overall_e_peak"],
        mean_pc_df["PC1"],
        xlabel="Capacity factor [pu]",
        ylabel="Mean wind power in evening peak [pu]",
        cbar_label="PC1 score",
        save_path=save_path,
    )


def plot_fig11b(tou_means, mean_pc_df, save_path=None):
    """
    Evening-peak mean power vs CF, coloured by mean PC2 score.  Fig. 11(b).
    Implements the scatter part of cell 71.
    """
    return _scatter_cf(
        tou_means["capacity_factor"],
        tou_means["overall_e_peak"],
        mean_pc_df["PC2"],
        xlabel="Capacity factor [pu]",
        ylabel="Mean wind power in evening peak [pu]",
        cbar_label="Mean PC2 score",
        save_path=save_path,
    )


def plot_fig12(tou_means, mean_pc_df, save_path=None):
    """
    Morning (00–06) mean power vs CF, coloured by mean PC2 score.  Fig. 12.
    Implements cell 72.
    """
    return _scatter_cf(
        tou_means["capacity_factor"],
        tou_means["morning_mean"],
        mean_pc_df["PC2"],
        xlabel="Capacity factor [pu]",
        ylabel="Mean wind power in morning periods [pu]",
        cbar_label="PC2 score",
        save_path=save_path,
    )


def plot_fig13(tou_means, mean_pc_df, save_path=None):
    """
    Midday (10–14) mean power vs CF, coloured by mean PC3 score.  Fig. 13.
    Implements cell 74.
    """
    return _scatter_cf(
        tou_means["capacity_factor"],
        tou_means["midday"],
        mean_pc_df["PC3"],
        xlabel="Capacity factor [pu]",
        ylabel="Mean wind power in midday [pu]",
        cbar_label="PC3 score",
        save_path=save_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 2 – Mean CF map  (no flagged cell; written fresh using plot_field)
# ─────────────────────────────────────────────────────────────────────────────

def plot_mean_cf_map(
    id_df_means,
    rsa_shp_path,
    save_path=None,
):
    """
    Spatial map of mean normalised wind power (capacity factor) per location.

    Manuscript Fig. 2.  Uses map_plotter.plot_field; no flagged cell existed
    in the original notebook — this was produced externally and reconstructed
    here using the same plotting infrastructure.

    Parameters
    ----------
    id_df_means : DataFrame  with columns [ID, Longitude, Latitude, Mean_Generation]
    rsa_shp_path : str
    save_path : path-like or None

    Returns
    -------
    fig, ax
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from map_plotter import plot_field

    fig, ax = plot_field(
        values=id_df_means.set_index("ID")["Mean_Generation"],
        sa_ids_df=id_df_means[["ID", "Longitude", "Latitude"]],
        vmin=0,
        vmax=None,
        markersize=1,
        color_ramp="Spectral_r",
        cbar_label="Mean normalised wind power [pu]",
        rsa_shp_path=rsa_shp_path,
        rsa_edgecolor="black",
        rsa_linewidth=1.5,
        rsa_facecolor="none",
        plot_path=str(save_path) if save_path else None,
    )
    return fig, ax
