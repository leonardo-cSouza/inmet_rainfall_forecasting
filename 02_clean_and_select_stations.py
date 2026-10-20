"""
Step 2 - Missing-data filter and selection of two stations per state.

1. Computes the share of missing rainfall per station (missing_report.csv).
2. Keeps stations with less than MISSING_THRESHOLD missing (Fig. 1 of the paper).
3. Assigns each station to a state and keeps the STATIONS_PER_STATE most
   complete per state (the 46 study stations).

State assignment: pass --states-geojson with a Brazilian states boundary file
(needs geopandas) to locate every station by its coordinates. Without it, the
curated table data/station_metadata.csv (the 46 study stations) is used.

    python scripts/02_clean_and_select_stations.py [--states-geojson FILE]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

import config
from inmet.cleaning import assign_states, filter_by_missing, missing_report, select_per_state
from inmet.io import read_dataset, write_dataset
from inmet.plotting import plot_params, save


def plot_missing_map(report, path, states_geojson=None):
    """Stations kept by the filter, shaded by % missing rainfall (Fig. 1)."""
    plot_params()
    fig, ax = plt.subplots(figsize=(8, 8))
    if states_geojson is not None:
        import geopandas as gpd
        gpd.read_file(states_geojson).plot(ax=ax, color="#f5f5f5",
                                           edgecolor="#bdc3c7", linewidth=0.6)
    sc = ax.scatter(report["LONGITUDE"], report["LATITUDE"],
                    c=report["MISSING_PCT"] * 100, cmap="Greys", s=35,
                    edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(sc, ax=ax, shrink=0.8).set_label("% of (NaN)")
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--states-geojson", default=None,
                        help="Brazilian states boundaries (GeoJSON/shapefile)")
    parser.add_argument("--threshold", type=float, default=config.MISSING_THRESHOLD)
    args = parser.parse_args()

    df = read_dataset(config.CONSOLIDATED_FILE)
    report = missing_report(df)
    write_dataset(report, config.MISSING_REPORT_FILE)

    clean = filter_by_missing(df, report, args.threshold)
    kept = report[report["MISSING_PCT"] < args.threshold]
    print(f"Stations analysed: {len(report)}")
    print(f"Stations kept (< {args.threshold:.0%} missing rainfall): {len(kept)}")
    write_dataset(clean, config.CLEAN_FILE)
    plot_missing_map(kept, config.FIG_OUT / "NaN_map.png", args.states_geojson)

    with_states = assign_states(kept, args.states_geojson)
    n_unknown = with_states["UF"].isna().sum()
    if n_unknown:
        print(f"{n_unknown} kept stations have no state assigned "
              f"(pass --states-geojson to locate all of them)")

    selected, dropped = select_per_state(with_states)
    write_dataset(selected, config.SELECTED_STATIONS_FILE)
    print(f"Selected {len(selected)} stations in {selected['UF'].nunique()} federative units "
          f"-> {config.SELECTED_STATIONS_FILE}")
    if dropped:
        print(f"States with fewer than {config.STATIONS_PER_STATE} eligible stations "
              f"(excluded): {', '.join(dropped)}")


if __name__ == "__main__":
    main()
