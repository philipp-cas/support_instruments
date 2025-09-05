# src/data_store.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from .config import DATA_CSV
from .data_import import build_data_from_sources

def save_data(df: pd.DataFrame, csv_path: Path = DATA_CSV):
    """Speichert den DataFrame als CSV im base/data-Verzeichnis."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=True)

def load_data(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    """Lädt den DataFrame aus CSV, falls vorhanden."""
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["DateTime"], index_col="DateTime")
    raise FileNotFoundError("Kein CSV-Cache gefunden in base/data/. Bitte get_data(force_refresh=True) nutzen.")

def get_data(force_refresh: bool = False,
             rebap_csv: Path | None = None,
             id1_xlsx: Path | None = None) -> pd.DataFrame:
    """
    Liefert den DataFrame:
      - Falls CSV-Cache vorhanden & !force_refresh: laden
      - Sonst: aus Quellen neu bauen, speichern (CSV) und zurückgeben.
    Für Neuaufbau sind rebap_csv und id1_xlsx erforderlich.
    """
    if not force_refresh:
        try:
            return load_data()
        except FileNotFoundError:
            pass

    if rebap_csv is None or id1_xlsx is None:
        raise ValueError("Zum Neuaufbau bitte Pfade zu rebap_csv und id1_xlsx angeben.")

    df = build_data_from_sources(rebap_csv=rebap_csv, id1_xlsx=id1_xlsx)
    save_data(df)
    return df
