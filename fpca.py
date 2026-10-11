"""
Functional PCA computations for the J3 manuscript.

All SVD operations are uncentred (the mean daily profile is retained as the
first singular direction rather than subtracted beforehand), following the
approach described in the manuscript.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# SVD / decomposition
# ─────────────────────────────────────────────────────────────────────────────

def fit_uncentred_svd(X: np.ndarray, n_components: int = 6) -> tuple:
    """
    Fit full uncentred SVD of the daily-profile matrix X.

    Parameters
    ----------
    X : ndarray, shape (N_locations * N_days, 24)
        Stacked daily profiles (location-major order).
    n_components : int
        Number of leading components to return explicitly.

    Returns
    -------
    svd_model : dict
        Keys: U, S, Vt, explained_variance (S^2/sum(S^2)), cumulative_variance.
    scores : ndarray, shape (N_locations * N_days, n_components)
        F * Lambda  (= U[:, :k] * S[:k]).
    components : ndarray, shape (n_components, 24)
        Right singular vectors (PC loadings), Vt[:k].
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    total = np.sum(S ** 2)
    exp_var = (S ** 2) / total

    k = min(n_components, len(S))
    scores = U[:, :k] * S[:k]
    components = Vt[:k, :]

    svd_model = {
        "U": U,
        "S": S,
        "Vt": Vt,
        "explained_variance": exp_var,
        "cumulative_variance": np.cumsum(exp_var),
    }
    return svd_model, scores, components


def invert_component(scores: np.ndarray, components: np.ndarray, k: int):
    """
    Flip the sign of component k (0-based) so that its loading is positive.

    Implements the sign convention used in the original notebook
    (``invert_component(scores, components, 0)`` ensures PC1 is positive).
    """
    scores = scores.copy()
    components = components.copy()
    scores[:, k] *= -1
    components[k, :] *= -1
    return scores, components


# ─────────────────────────────────────────────────────────────────────────────
# R² about the mean daily profile  (Eq. used in Fig. 4 / scree plot)
# ─────────────────────────────────────────────────────────────────────────────

def r2_about_mean(X: np.ndarray, svd_model: dict) -> tuple:
    """
    Compute individual and cumulative R² about the mean daily profile for an
    uncentred SVD decomposition.

    The zero model is the mean daily profile x̄ (average over all N*D rows).
    The baseline sum of squares is SS_total − SS_mean = sum(S²) − N*||x̄||².

    Implements the formula used in cells 20 and 24 of the original notebook.

    Parameters
    ----------
    X : ndarray, shape (N*D, 24)
    svd_model : dict returned by fit_uncentred_svd

    Returns
    -------
    r2_ind : ndarray, shape (n_S,)  individual R² increments
    r2_cum : ndarray, shape (n_S,)  cumulative R²
    mean_share : float              fraction of SS_total explained by x̄ alone
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    x_bar = X.mean(axis=0)
    S = np.asarray(svd_model["S"], dtype=float)

    total_ss = np.sum(X ** 2)
    mean_share = N * np.sum(x_bar ** 2) / total_ss
    baseline = 1.0 - mean_share

    ss_ratio = (S ** 2) / total_ss
    r2_cum = 1.0 - (1.0 - np.cumsum(ss_ratio)) / baseline
    r2_ind = np.diff(np.concatenate([[0.0], r2_cum]))
    return r2_ind, r2_cum, mean_share


# ─────────────────────────────────────────────────────────────────────────────
# Score dictionary (location × day DataFrames, one per PC)
# ─────────────────────────────────────────────────────────────────────────────

def build_scores_dict(
    scores: np.ndarray,
    n_locations: int,
    dates: pd.DatetimeIndex,
    location_ids,
    n_modes: int = 6,
) -> dict:
    """
    Reshape the flat scores array into a dict of (D, N_locations) DataFrames.

    Parameters
    ----------
    scores : ndarray, shape (N_locations * N_days, n_modes)
    n_locations : int
    dates : DatetimeIndex  (length N_days)
    location_ids : array-like  (length N_locations)
    n_modes : int  number of PCs to include

    Returns
    -------
    scores_dict : dict  {"PC1": DataFrame(index=dates, columns=IDs), ...}
    """
    scores_dict = {}
    for i in range(n_modes):
        scores_i = scores[:, i].reshape(n_locations, -1)   # (N_locs, N_days)
        scores_dict[f"PC{i + 1}"] = pd.DataFrame(
            scores_i.T,
            index=dates,
            columns=location_ids,
        )
    return scores_dict


def mean_pc_scores(scores_dict: dict, n_modes: int = 4) -> pd.DataFrame:
    """
    Mean PC score per location over all days.

    Parameters
    ----------
    scores_dict : dict from build_scores_dict
    n_modes : int  number of PCs to include

    Returns
    -------
    mean_pc_df : DataFrame, shape (N_locations, n_modes)
        Columns "PC1" … "PCn_modes"; index = location IDs.
    """
    data = {f"PC{i+1}": scores_dict[f"PC{i+1}"].mean(axis=0) for i in range(n_modes)}
    df = pd.DataFrame(data)
    df.index.name = "ID"
    return df


def centre_scores(scores_dict: dict, pc: str = "PC1") -> dict:
    """
    Centre one PC by subtracting the regional mean (mean over all locations
    and all days) so that the zero point is the study-area average.

    Returns a copy of scores_dict with the specified PC centred; all other PCs
    are passed through unchanged.
    """
    out = {}
    for k, df in scores_dict.items():
        if k == pc:
            out[k] = df - df.mean().mean()
        else:
            out[k] = df
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-year decomposition + Tucker congruence
# ─────────────────────────────────────────────────────────────────────────────

def _unit_rows(C: np.ndarray) -> np.ndarray:
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def _comp_rows(C: np.ndarray, H: int = 24) -> np.ndarray:
    C = np.asarray(C)
    return C if C.shape[1] == H else C.T


def match_components(C_year: np.ndarray, C_pooled: np.ndarray) -> tuple:
    """
    Match each pooled mode to a year mode using the Hungarian algorithm,
    resolving arbitrary sign and ordering.

    Returns (idx, signs, cosines) each of length K_pooled.
    """
    P = _unit_rows(np.asarray(C_pooled))
    C = _unit_rows(np.asarray(C_year))
    M = P @ C.T                          # (K_pooled, K_year)

    rows, cols = linear_sum_assignment(-np.abs(M))
    idx = np.zeros(P.shape[0], dtype=int)
    idx[rows] = cols

    cos = M[np.arange(P.shape[0]), idx]
    signs = np.where(cos >= 0, 1.0, -1.0)
    return idx, signs, np.abs(cos)


def per_year_decomposition(
    wind_df: pd.DataFrame,
    pooled_components: np.ndarray,
    n_modes: int = 4,
    n_return: int = 8,
) -> tuple:
    """
    Fit an uncentred SVD for each year in wind_df and align modes to the
    pooled five-year components via Tucker congruence (cosine similarity).

    Parameters
    ----------
    wind_df : DataFrame, shape (N_hours, N_locations)
    pooled_components : ndarray, shape (K, 24)  from five-year SVD
    n_modes : int  number of modes to track
    n_return : int  modes to fit per year (>= n_modes for matching room)

    Returns
    -------
    year_components : dict  {year: (n_modes, 24) unit-norm components}
    year_r2 : dict          {year: (n_modes,) individual R² for matched modes}
    match_df : DataFrame    full cosine table for manuscript Table / Fig. 7
    """
    from data import to_daily_matrix

    pooled = _unit_rows(_comp_rows(pooled_components)[:n_modes])
    years = sorted(wind_df.index.year.unique())

    year_components, year_r2 = {}, {}
    rows = []

    for y in years:
        X_y = to_daily_matrix(wind_df.loc[wind_df.index.year == y])
        svd_y, _, comps_y = fit_uncentred_svd(X_y, n_components=n_return)
        r2_ind_y, _, _ = r2_about_mean(X_y, svd_y)

        C_y = _unit_rows(_comp_rows(comps_y))
        idx, signs, cos = match_components(C_y, pooled)

        year_components[y] = signs[:, None] * C_y[idx]
        year_r2[y] = r2_ind_y[idx]

        for k in range(n_modes):
            rows.append({
                "year": y,
                "pooled_PC": k + 1,
                "year_PC": int(idx[k]) + 1,
                "sign": int(signs[k]),
                "tucker_congruence": round(float(cos[k]), 4),
                "r2_ind": round(float(r2_ind_y[idx[k]]), 4),
            })

    match_df = pd.DataFrame(rows)
    return year_components, year_r2, match_df


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction & residuals
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct(
    svd_model: dict,
    Q: int,
    original_index: pd.DatetimeIndex,
    original_columns,
) -> pd.DataFrame:
    """
    Reconstruct the wind-power time series using the first Q SVD modes.

    Parameters
    ----------
    svd_model : dict from fit_uncentred_svd
    Q : int  number of modes
    original_index : DatetimeIndex  of wind_df (hourly)
    original_columns : array-like  location IDs

    Returns
    -------
    reconstructed_df : DataFrame, shape (N_hours, N_locations)
    """
    U, S, Vt = svd_model["U"], svd_model["S"], svd_model["Vt"]
    X_flat = (U[:, :Q] * S[:Q]) @ Vt[:Q, :]

    n_locs = len(original_columns)
    n_days = len(original_index) // 24
    X_3d = X_flat.reshape(n_locs, n_days, 24)
    X_2d = X_3d.reshape(n_locs, -1).T

    return pd.DataFrame(X_2d, index=original_index, columns=original_columns)


# ─────────────────────────────────────────────────────────────────────────────
# Demand-period regressions
# ─────────────────────────────────────────────────────────────────────────────

def linear_r2(X_cols: list, y: np.ndarray) -> float:
    """OLS R² for y regressed on the columns in X_cols."""
    from sklearn.linear_model import LinearRegression
    X = np.column_stack(X_cols)
    return LinearRegression().fit(X, y).score(X, y)


def evening_peak_regressions(tou_means: pd.DataFrame, mean_pc_df: pd.DataFrame) -> dict:
    """
    R² for evening-peak mean power regressed on CF alone and CF + mean PC2.

    Returns dict with keys r2_cf, r2_cf_pc2.
    Implements the regression in cell 71 of the original notebook.
    """
    df = tou_means[["capacity_factor", "overall_e_peak"]].join(
        mean_pc_df[["PC2"]], how="inner"
    )
    cf = df["capacity_factor"].to_numpy()
    ep = df["overall_e_peak"].to_numpy()
    f2 = df["PC2"].to_numpy()

    return {
        "r2_cf":     linear_r2([cf], ep),
        "r2_cf_pc2": linear_r2([cf, f2], ep),
    }


def midday_regressions(tou_means: pd.DataFrame, mean_pc_df: pd.DataFrame) -> dict:
    """
    R² for midday mean power regressed on CF alone and CF + mean PC3.

    Returns dict with keys r2_cf, r2_cf_pc3.
    Implements the regression in cell 73 of the original notebook.
    """
    df = tou_means[["capacity_factor", "midday"]].join(
        mean_pc_df[["PC3"]], how="inner"
    )
    cf = df["capacity_factor"].to_numpy()
    md = df["midday"].to_numpy()
    f3 = df["PC3"].to_numpy()

    return {
        "r2_cf":     linear_r2([cf], md),
        "r2_cf_pc3": linear_r2([cf, f3], md),
    }


def cf_pc1_correlation(tou_means: pd.DataFrame, mean_pc_df: pd.DataFrame) -> dict:
    """
    Pearson r and R² between CF and mean PC1 score.

    Implements cell 75 of the original notebook.
    """
    corr = mean_pc_df["PC1"].corr(tou_means["capacity_factor"])
    return {"pearson_r": corr, "r2": corr ** 2}
