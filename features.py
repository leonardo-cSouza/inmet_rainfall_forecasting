"""Calendar encodings, rainfall lags and per-station design matrices."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def add_calendar_features(df, date_col=config.DATE_COL):
    """Sine/cosine encoding of day of year and month (Eq. 1 of the paper)."""
    df = df.copy()
    doy = df[date_col].dt.dayofyear
    month = df[date_col].dt.month
    df["DAY_OF_YEAR"] = doy
    df["MONTH"] = month
    df["DAY_SIN"] = np.sin(2 * np.pi * doy / 365.25)
    df["DAY_COS"] = np.cos(2 * np.pi * doy / 365.25)
    df["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * month / 12)
    return df


def lag_name(lag):
    return f"{config.TARGET_COL}_LAG_{lag}"


def build_station_features(train_df, test_df, station,
                           same_day_covariates=config.SAME_DAY_COVARIATES):
    """
    Feature table for one station: 6 meteorological + 4 calendar + 10 rain lags.

    Train and test are concatenated before shifting, so the first test days get
    their lags from the end of the training period. Rows whose lags reach before
    the start of the record are dropped.
    """
    s_col, d_col, y_col = config.STATION_COL, config.DATE_COL, config.TARGET_COL
    tr = train_df[train_df[s_col] == station].assign(__split__="train")
    te = test_df[test_df[s_col] == station].assign(__split__="test")
    full = pd.concat([tr, te]).sort_values(d_col).reset_index(drop=True)

    met_cols = list(config.MET_VARS)
    if not same_day_covariates:
        for c in config.MET_VARS:
            full[f"{c}_LAG_1"] = full[c].shift(1)
        met_cols = [f"{c}_LAG_1" for c in config.MET_VARS]

    for lag in config.RAIN_LAGS:
        full[lag_name(lag)] = full[y_col].shift(lag)

    feature_cols = met_cols + config.CALENDAR_VARS + [lag_name(l) for l in config.RAIN_LAGS]
    full = full.dropna(subset=feature_cols + [y_col]).reset_index(drop=True)

    train = full[full["__split__"] == "train"].drop(columns="__split__")
    test = full[full["__split__"] == "test"].drop(columns="__split__")
    return train, test, feature_cols


class StationData:
    """Scaled arrays for one station: standardized X, standardized log1p(y)."""

    def __init__(self, train, test, feature_cols):
        self.train, self.test, self.feature_cols = train, test, feature_cols
        self.dates_test = test[config.DATE_COL].values
        self.y_train = train[config.TARGET_COL].to_numpy(float)
        self.y_test = test[config.TARGET_COL].to_numpy(float)

        self.x_scaler = StandardScaler()
        self.X_train = self.x_scaler.fit_transform(train[feature_cols].to_numpy(float))
        self.X_test = self.x_scaler.transform(test[feature_cols].to_numpy(float))

        self.y_train_log = np.log1p(self.y_train)
        self.y_scaler = StandardScaler()
        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train_log.reshape(-1, 1)).ravel()

    def to_mm(self, y_scaled):
        """Undo the target scaling and the log transform; clip to [0, training maximum]."""
        y_log = self.y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
        return self.clip_mm(np.expm1(y_log))

    def clip_mm(self, y_mm):
        """Physical bounds: no negative rain, nothing above the station's training record.
        Guards against the exponential blowing up rare extreme outputs of the log model."""
        return np.clip(y_mm, 0, self.y_train.max())

    def lag_free_columns(self):
        """Indices of non-lag features (ARIMAX models its own autoregression)."""
        return [i for i, c in enumerate(self.feature_cols)
                if not c.startswith(f"{config.TARGET_COL}_LAG_")]
