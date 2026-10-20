"""
Step 1 - Consolidate the raw INMET daily files into one dataset.

Reads every dados_<CODE>_D_<start>_<end>.csv in RAW_DIR, standardises column
names, attaches station metadata and restricts the records to the study period.

    python scripts/01_build_dataset.py [--raw-dir PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from inmet.io import read_raw_station_file, write_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    args = parser.parse_args()

    files = sorted(args.raw_dir.glob("dados_*_D_*.csv"))
    print(f"Found {len(files)} raw station files in {args.raw_dir}")

    frames, failed = [], []
    for path in files:
        try:
            frames.append(read_raw_station_file(path))
        except Exception as exc:          # one bad download must not stop the run
            failed.append((path.name, str(exc)))
    for name, err in failed:
        print(f"  skipped {name}: {err}")

    df = pd.concat(frames, ignore_index=True)
    in_period = df[config.DATE_COL].between(config.START_DATE, config.END_DATE)
    df = df[in_period].sort_values([config.CODE_COL, config.DATE_COL])
    df = df.drop_duplicates([config.CODE_COL, config.DATE_COL])

    write_dataset(df, config.CONSOLIDATED_FILE)
    print(f"{df[config.CODE_COL].nunique()} stations, {len(df):,} station-days "
          f"-> {config.CONSOLIDATED_FILE}")


if __name__ == "__main__":
    main()
