"""
Data loading and preprocessing for the J3 FPCA manuscript.

Raw data: WASA 3x3 km normalised wind-power capacity factors (.npz).
Coordinates: WC_3x3.csv  (columns: ID, Longitude, Latitude).
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

_NPZ_FILES = {
    2014: "sam_power_wc_3x3_2014.npz",
    2015: "sam_power_wc_3x3_2015.npz",
    2016: "sam_power_wc_3x3_2016.npz",
    2017: "sam_power_wc_3x3_2017.npz",
    2019: "sam_power_wc_and_ipps_3x3_2019.npz",
}


def _load_npz(path: Path) -> pd.DataFrame:
    """
    Load one WASA .npz file.

    Returns a DataFrame with a DatetimeIndex (hourly) and integer location IDs
    as columns (rows = hours, columns = locations).
    """
    d = np.load(path, allow_pickle=True)
    df = pd.DataFrame(
        d["power_values"],
        index=d["id_array"],
        columns=d["time_array"],
    )
    df.columns = df.columns.astype(int)
    df = df.T                        # transpose to (time, locations)
    df.columns = df.columns.astype(int)
    df.index = pd.to_datetime(df.index)
    return df


def load_wind(data_dir: Path, years=(2014, 2015, 2016, 2017, 2019)) -> pd.DataFrame:
    """
    Load and concatenate WASA wind-power data for the requested years.

    Parameters
    ----------
    data_dir : Path
        Folder containing the .npz files (DATA_DIR from config).
    years : sequence of int
        Years to load; default matches the five-year study period.

    Returns
    -------
    wind_df : DataFrame, shape (N_hours_total, N_locations)
        Hourly normalised wind power [pu].  DatetimeIndex, int columns = IDs.
        29 February 2016 is excluded.
    """
    frames = [_load_npz(Path(data_dir) / _NPZ_FILES[y]) for y in years]
    df = pd.concat(frames, axis=0).sort_index()
    df = df.loc[~((df.index.month == 2) & (df.index.day == 29))]   # drop leap day
    return df


def load_coords(data_dir: Path) -> pd.DataFrame:
    """
    Load location coordinates from WC_3x3.csv.

    Returns
    -------
    id_df : DataFrame, columns [ID, Longitude, Latitude]
    """
    return pd.read_csv(Path(data_dir) / "WC_3x3.csv")


def align_locations(wind_df: pd.DataFrame, id_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns of wind_df that appear in id_df, in id_df order.

    This matches the original notebook's ``wind_df = wind_df.loc[:, w_points["ID"]]``.
    """
    return wind_df.loc[:, id_df["ID"]]


# ─────────────────────────────────────────────────────────────────────────────
# Reshaping
# ─────────────────────────────────────────────────────────────────────────────

def to_daily_matrix(wind_df: pd.DataFrame) -> np.ndarray:
    """
    Reshape (N_hours, N_locations) → (N_locations * N_days, 24).

    Stacking is location-major: all days of location 0 come first, then all
    days of location 1, etc.  This matches the SVD input X used throughout
    the manuscript.

    Parameters
    ----------
    wind_df : DataFrame, shape (N_locations * 24, N_days) effectively stored
              as (N_hours, N_locations) with DatetimeIndex.

    Returns
    -------
    X : ndarray, shape (N_locations * N_days, 24)
    """
    return wind_df.T.values.flatten().reshape(-1, 24)


def daily_dates(wind_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Unique calendar dates in wind_df (one per day)."""
    return wind_df.index.normalize().unique()


# ─────────────────────────────────────────────────────────────────────────────
# TOU mean-power features
# ─────────────────────────────────────────────────────────────────────────────

def compute_tou_means(
    wind_df: pd.DataFrame,
    hds_months,
    lds_months,
    hds_tou: dict,
    lds_tou: dict,
    morning_hours,
    midday_hours,
) -> pd.DataFrame:
    """
    Compute per-location mean wind power for each TOU period and season.

    Replicates cell 6–7 of the original notebook exactly: all TOU masks
    use demand-season-appropriate hours; capacity_factor = overall mean.

    Parameters
    ----------
    wind_df : DataFrame, shape (N_hours, N_locations)
    hds_months, lds_months : lists of int  (season month numbers)
    hds_tou, lds_tou : dicts  {period_name: [hours]}
    morning_hours, midday_hours : lists of int

    Returns
    -------
    tou_means : DataFrame, index = location IDs
        Columns include capacity_factor, overall_e_peak, morning_mean, midday, etc.
    """
    wind_hour = wind_df.copy()
    wind_hour["hour"] = wind_hour.index.hour
    wind_hour["month"] = wind_hour.index.month

    hds_mask = wind_hour["month"].isin(hds_months)
    lds_mask = wind_hour["month"].isin(lds_months)

    tou_means = pd.DataFrame(index=wind_df.columns)

    # Season-specific period means + std
    for season, mask, tou in [("hds", hds_mask, hds_tou), ("lds", lds_mask, lds_tou)]:
        season_data = wind_hour.loc[mask, wind_df.columns]
        for period, hours in tou.items():
            hour_mask = wind_hour.loc[mask, "hour"].isin(hours)
            col = f"{season}_{period}"
            tou_means[col] = season_data.loc[hour_mask].mean()
            tou_means[f"{col}_std"] = season_data.loc[hour_mask].std()

    # Overall (combined-season) period means + std
    for period in hds_tou:
        hds_p = wind_hour.loc[hds_mask & wind_hour["hour"].isin(hds_tou[period]), wind_df.columns]
        lds_p = wind_hour.loc[lds_mask & wind_hour["hour"].isin(lds_tou[period]), wind_df.columns]
        name = f"overall_{period}"
        tou_means[name] = pd.concat([hds_p, lds_p]).mean()
        tou_means[f"{name}_std"] = pd.concat([hds_p, lds_p]).std()

    # Midday (all-season)
    midday_mask = wind_hour["hour"].isin(midday_hours)
    for season, mask in [("hds", hds_mask), ("lds", lds_mask)]:
        tou_means[f"{season}_midday"] = wind_hour.loc[mask & midday_mask, wind_df.columns].mean()
        tou_means[f"{season}_midday_std"] = wind_hour.loc[mask & midday_mask, wind_df.columns].std()
    tou_means["midday"] = wind_hour.loc[midday_mask, wind_df.columns].mean()
    tou_means["midday_std"] = wind_hour.loc[midday_mask, wind_df.columns].std()

    # Morning (all-season)
    morning_mask = wind_hour["hour"].isin(morning_hours)
    tou_means["morning_mean"] = wind_hour.loc[morning_mask, wind_df.columns].mean()

    # Capacity factor = overall mean normalised power
    tou_means["capacity_factor"] = wind_df.mean()

    return tou_means


def compute_mean_power_map(wind_df: pd.DataFrame, id_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean normalised wind power per location merged with coordinates.

    Returns
    -------
    id_df_means : DataFrame, columns [ID, Longitude, Latitude, Mean_Generation]
    """
    means = wind_df.mean().rename("Mean_Generation")
    return pd.merge(id_df, means, left_on="ID", right_index=True)
