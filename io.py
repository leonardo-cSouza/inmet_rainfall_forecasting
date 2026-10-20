"""
Reading raw INMET daily files and the intermediate CSVs of the pipeline.

A raw BDMEP download looks like this (header block, blank line, data table):

    Nome: BRASILIA
    Codigo Estacao: A001
    Latitude: -15.78944444
    Longitude: -47.92583332
    Altitude: 1160.96
    ...
    Data Medicao;PRECIPITACAO TOTAL, DIARIO (AUT)(mm);PRESSAO ATMOSFERICA ...
    2015-01-01;0;887;30.4;16.9;60.1;30;1.8;
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import config

# Substring of each raw INMET column header -> short name used in the study.
# Matching on substrings avoids problems with the degree sign, whose encoding
# varies between downloads.
RAW_COLUMN_KEYS = {
    "DATA MEDICAO": "DATA",
    "PRECIPITACAO TOTAL": "PREC",
    "PRESSAO ATMOSFERICA MEDIA": "PMED",
    "TEMPERATURA MAXIMA": "TMAX",
    "TEMPERATURA MINIMA": "TMIN",
    "UMIDADE RELATIVA DO AR, MEDIA": "UMED",
    "UMIDADE RELATIVA DO AR, MINIMA": "UMIN",
    "VENTO, VELOCIDADE MEDIA": "VEL",
}

METADATA_KEYS = {
    "NOME": "ESTACAO_NOME",
    "CODIGO ESTACAO": "ESTACAO_CODIGO",
    "LATITUDE": "LATITUDE",
    "LONGITUDE": "LONGITUDE",
    "ALTITUDE": "ALTITUDE",
}


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def _normalise(text):
    return strip_accents(text).upper().strip()


def read_raw_station_file(path):
    """Parse one raw INMET daily file into a tidy DataFrame with metadata columns."""
    path = Path(path)
    with open(path, "r", encoding="latin-1") as fh:
        lines = fh.readlines()

    metadata, header_idx = {}, None
    for i, line in enumerate(lines[:20]):
        clean = line.strip()
        if clean.count(";") > 5:                  # first row of the data table
            header_idx = i
            break
        if ":" in clean:
            key, value = clean.split(":", 1)
            key = _normalise(key)
            if key in METADATA_KEYS:
                metadata[METADATA_KEYS[key]] = value.replace(";", "").strip()
    if header_idx is None:
        raise ValueError(f"No data header found in {path.name}")

    df = pd.read_csv(path, sep=";", skiprows=header_idx, encoding="latin-1",
                     dtype=str)
    df = df.dropna(how="all", axis=1)

    rename = {}
    for col in df.columns:
        norm = _normalise(col)
        for key, short in RAW_COLUMN_KEYS.items():
            if key in norm:
                rename[col] = short
    df = df.rename(columns=rename)[[c for c in RAW_COLUMN_KEYS.values() if c in rename.values()]]

    for col in df.columns:
        if col == "DATA":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            values = df[col].str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(values, errors="coerce")
    df = df.replace(-9999, np.nan).dropna(subset=["DATA"])

    df["ESTACAO_CODIGO"] = metadata.get("ESTACAO_CODIGO", path.stem.split("_")[1])
    df["ESTACAO_NOME"] = metadata.get("ESTACAO_NOME", "DESCONHECIDO")
    for key in ("LATITUDE", "LONGITUDE", "ALTITUDE"):
        df[key] = pd.to_numeric(metadata.get(key, "").replace(",", "."), errors="coerce")
    return df


def read_dataset(path):
    """Read a CSV written by this pipeline (semicolon-separated, UTF-8 with BOM)."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    return df


def write_dataset(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")


def load_station_metadata():
    """Station name -> state (UF) and region, curated for the 46 study stations."""
    return pd.read_csv(config.STATION_METADATA)
