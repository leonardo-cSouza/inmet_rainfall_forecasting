"""Missing-data report, station filtering and per-state selection."""
import numpy as np
import pandas as pd

import config


def missing_report(df, target_col=config.TARGET_COL):
    """
    Share of missing values per station.

    MISSING_PCT is computed over the rows each station has, as in the paper.
    COVERAGE_PCT is the share of the full study calendar with a valid rainfall
    value, which also penalises stations that started late or closed early.
    """
    n_days = len(pd.date_range(config.START_DATE, config.END_DATE, freq="D"))
    keys = [config.CODE_COL, config.STATION_COL, "LATITUDE", "LONGITUDE"]
    report = (df.groupby(keys, dropna=False)
                .agg(N_ROWS=(target_col, "size"),
                     N_VALID=(target_col, "count"),
                     FIRST_DATE=(config.DATE_COL, "min"),
                     LAST_DATE=(config.DATE_COL, "max"))
                .reset_index())
    report["MISSING_PCT"] = 1 - report["N_VALID"] / report["N_ROWS"]
    report["COVERAGE_PCT"] = report["N_VALID"] / n_days
    return report.sort_values("MISSING_PCT").reset_index(drop=True)


def filter_by_missing(df, report, threshold=config.MISSING_THRESHOLD):
    keep = report.loc[report["MISSING_PCT"] < threshold, config.CODE_COL]
    return df[df[config.CODE_COL].isin(keep)].copy()


def assign_states(report, states_geojson=None):
    """
    Add UF and REGION columns to a station report.

    With a states GeoJSON (for example the IBGE state boundaries) each station
    is located by point-in-polygon. Without one, the curated table in
    data/station_metadata.csv is used, which covers only the 46 study stations.
    """
    report = report.copy()
    if states_geojson is not None:
        import geopandas as gpd

        states = gpd.read_file(states_geojson)
        uf_col = next(c for c in ("sigla", "SIGLA_UF", "UF", "abbrev_state")
                      if c in states.columns)
        points = gpd.GeoDataFrame(
            report, geometry=gpd.points_from_xy(report["LONGITUDE"], report["LATITUDE"]),
            crs="EPSG:4326").to_crs(states.crs)
        joined = gpd.sjoin(points, states[[uf_col, "geometry"]], how="left",
                           predicate="within")
        report["UF"] = joined[uf_col].values
    else:
        meta = pd.read_csv(config.STATION_METADATA)
        report = report.merge(meta[[config.CODE_COL, "UF"]], on=config.CODE_COL, how="left")
    report["REGION"] = report["UF"].map(config.STATE_TO_REGION)
    return report


def select_per_state(report, k=config.STATIONS_PER_STATE):
    """
    Keep the k most complete stations of each state. States with fewer than k
    eligible stations are dropped, so every state contributes equally.
    """
    eligible = report.dropna(subset=["UF"])
    # A station must reach the test period to be evaluated at all
    eligible = eligible[eligible["LAST_DATE"] >= pd.Timestamp(config.SPLIT_DATE)]
    ranked = eligible.sort_values(["UF", "MISSING_PCT", "COVERAGE_PCT"],
                                  ascending=[True, True, False])
    chosen = ranked.groupby("UF").head(k)
    counts = chosen["UF"].value_counts()
    full_states = counts[counts == k].index
    dropped = sorted(set(counts.index) - set(full_states))
    return chosen[chosen["UF"].isin(full_states)].reset_index(drop=True), dropped
