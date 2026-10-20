"""
Hierarchical imputation, fitted on the training period only.

1. Rainfall (PREC): missing -> 0 mm, so no artificial rain events are created.
2. Other variables: gaps of up to INTERP_LIMIT consecutive days are linearly
   interpolated inside each station's own series.
3. Longer gaps: the station's mean for the same month and day, computed from
   the training years.
4. Fallbacks: the station's overall training mean, then the global training mean.
"""
import pandas as pd

import config


def fit_seasonal_means(train_df, cols, station_col=config.STATION_COL,
                       date_col=config.DATE_COL):
    tmp = train_df.copy()
    tmp["month"] = tmp[date_col].dt.month
    tmp["day"] = tmp[date_col].dt.day
    means = tmp.groupby([station_col, "month", "day"])[cols].mean()
    fallback_station = tmp.groupby(station_col)[cols].mean()
    fallback_global = tmp[cols].mean()
    return means, fallback_station, fallback_global


def apply_hybrid_impute(df, cols, means, fallback_station, fallback_global,
                        station_col=config.STATION_COL, date_col=config.DATE_COL,
                        target_col=config.TARGET_COL, interp_limit=config.INTERP_LIMIT):
    tmp = df.sort_values([station_col, date_col]).copy()

    for c in cols:
        if c == target_col:
            tmp[c] = tmp[c].fillna(0.0)
        else:
            tmp[c] = tmp.groupby(station_col)[c].transform(
                lambda s: s.interpolate(method="linear", limit=interp_limit,
                                        limit_direction="both"))

    tmp["month"] = tmp[date_col].dt.month
    tmp["day"] = tmp[date_col].dt.day
    tmp = tmp.join(means, on=[station_col, "month", "day"], rsuffix="_seasonal")

    for c in cols:
        seasonal = f"{c}_seasonal"
        if seasonal in tmp.columns:
            tmp[c] = tmp[c].fillna(tmp[seasonal])
            tmp = tmp.drop(columns=[seasonal])
        tmp[c] = tmp[c].fillna(tmp[station_col].map(fallback_station[c]))
        tmp[c] = tmp[c].fillna(fallback_global[c])

    return tmp.drop(columns=["month", "day"])


def imputation_share(raw_df, cols, station_col=config.STATION_COL):
    """Share of values per station and variable that had to be imputed."""
    return raw_df.groupby(station_col)[cols].apply(lambda g: g.isna().mean())
