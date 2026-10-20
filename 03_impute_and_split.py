"""
Step 3 - Chronological train/test split, calendar features and imputation.

Train: START_DATE up to the day before SPLIT_DATE. Test: SPLIT_DATE to END_DATE.
The seasonal means used to fill long gaps are fitted on the training period
only, then applied to both sets.

    python scripts/03_impute_and_split.py [--all-clean-stations]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from inmet.features import add_calendar_features
from inmet.imputation import apply_hybrid_impute, fit_seasonal_means, imputation_share
from inmet.io import read_dataset, write_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--all-clean-stations", action="store_true",
                        help="impute every station that passed the missing-data filter, "
                             "not only the selected ones")
    args = parser.parse_args()

    df = read_dataset(config.CLEAN_FILE)
    if not args.all_clean_stations:
        selected = read_dataset_codes(config.SELECTED_STATIONS_FILE)
        df = df[df[config.CODE_COL].isin(selected)]
    print(f"Imputing {df[config.CODE_COL].nunique()} stations")

    cols = [config.TARGET_COL] + config.MET_VARS
    split = pd.Timestamp(config.SPLIT_DATE)
    train = df[df[config.DATE_COL] < split].copy()
    test = df[df[config.DATE_COL] >= split].copy()

    share = imputation_share(df, cols)
    share.to_csv(config.DATA_OUT / "imputation_share.csv", sep=";")
    print("Mean share of imputed values per variable:")
    print(share.mean().round(3).to_string())

    means, fb_station, fb_global = fit_seasonal_means(train, cols)
    train_imp = add_calendar_features(apply_hybrid_impute(train, cols, means, fb_station, fb_global))
    test_imp = add_calendar_features(apply_hybrid_impute(test, cols, means, fb_station, fb_global))

    write_dataset(train_imp, config.TRAIN_FILE)
    write_dataset(test_imp, config.TEST_FILE)
    print(f"Train: {train_imp[config.DATE_COL].min():%Y-%m-%d} to "
          f"{train_imp[config.DATE_COL].max():%Y-%m-%d}, {len(train_imp):,} rows -> {config.TRAIN_FILE}")
    print(f"Test:  {test_imp[config.DATE_COL].min():%Y-%m-%d} to "
          f"{test_imp[config.DATE_COL].max():%Y-%m-%d}, {len(test_imp):,} rows -> {config.TEST_FILE}")


def read_dataset_codes(path):
    return set(pd.read_csv(path, sep=";", encoding="utf-8-sig")[config.CODE_COL])


if __name__ == "__main__":
    main()
