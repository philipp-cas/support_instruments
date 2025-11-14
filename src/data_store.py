# src/data_store.py
# =============================================================================
# Daten-Cache-Handling (Laden, Speichern, Orchestrierung)
# -----------------------------------------------------------------------------
# Zweck
# -----
# Dieses Modul verwaltet den CSV-Cache für die Simulation.
#   - Speichert den aus Quellen gebauten DataFrame als CSV
#   - Lädt den Cache wieder, falls vorhanden
#   - Bietet eine zentrale `get_data()`-Funktion, die automatisch
#     entscheidet: Laden oder Neuaufbau
#
# Workflow
# --------
# 1. Beim ersten Mal (oder wenn die Quellen aktualisiert wurden):
#       get_data(force_refresh=True,
#                rebap_csv=DATA_DIR/"reBAP_utc.csv",
#                id1_xlsx=DATA_DIR/"id1_price_utc.xlsx")
#    → baut den DataFrame mit `build_data_from_sources()`, reichert ihn mit
#      `process_data()` (z. B. Asset-/Referenzspalten, MV-Schätzer) an,
#      speichert ihn als CSV unter `DATA_CSV` (z. B. `data/data_complete.csv`)
#      und gibt ihn zurück.
#
# 2. Danach (Standard):
#       df = get_data()
#    → lädt direkt den Cache aus `DATA_CSV` (z. B. `data/data_complete.csv`).
#
# Quellen
# -------
# - reBAP-CSV (UTC, deutsch formatiert, mit ";"-Trennung)
# - ID1-Excel (UTC-Stempel, Spalte "id1")
# - ENTSO-E-Datenbank (über src/data_import.build_data_from_sources)
#
# Rückgabe
# --------
# - `pd.DataFrame` mit Index=DateTime (15min, tz-naiv)
#
# Typische Nutzung
# ----------------
# from src.data_store import get_data
# df = get_data()  # lädt Cache oder baut neu, je nach Zustand
#
# =============================================================================


# src/data_store.py
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd

from .config import DATA_CSV, DATA_DIR
from .data_import import build_data_from_sources
from .data_processing import process_data  # Es werden die Spalten aus data_processing gemerged


# ---------- Speichern ----------
def save_data(df: pd.DataFrame, csv_path: Path = DATA_CSV) -> None:
    """
    Speichert den DataFrame als CSV.
    - schreibt zuerst in *.tmp und ersetzt dann atomar -> robuster
    - gibt Debug-Infos aus (Zielpfad, exists, Dateigröße)
    """
    csv_path = Path(csv_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("[save_data] Ziel:", csv_path)
    print("[save_data] Ordner existiert?", csv_path.parent.exists())

    try:
        tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
        df.to_csv(tmp, index=True)
        os.replace(tmp, csv_path)
    except Exception as e:
        print("[save_data] FEHLER beim Schreiben:", repr(e))
        raise

    ok = csv_path.exists()
    print("[save_data] exists() ->", ok)
    if ok:
        try:
            print("[save_data] Größe:", csv_path.stat().st_size, "Bytes")
        except Exception as e:
            print("[save_data] stat() Fehler:", repr(e))


# ---------- Laden ----------
def load_data(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    """
    Lädt den DataFrame aus dem CSV-Cache.
    Erwartet eine 'DateTime'-Spalte, die als Index geparst wird.
    """
    csv_path = Path(csv_path).resolve()
    print("[load_data] Lade:", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Kein CSV-Cache gefunden unter {csv_path}. "
            "Bitte zuerst get_data(force_refresh=True, ...) ausführen."
        )

    df = pd.read_csv(csv_path, parse_dates=["DateTime"], index_col="DateTime")
    print("[load_data] geladen:", df.shape, "Zeilen x Spalten")
    return df


# ---------- Orchestrierung ----------
def get_data(
    force_refresh: bool = False,
    rebap_csv: Path | None = None,
    id1_xlsx: Path | None = None,
    be_csv: Path | None = None,
    mstr_csv: Path | None = None,
    hochrechnung_csv: Path | None = None,
    futures_dir: Path | None = None,
    years = (2021, 2022, 2023, 2024, 2025),
) -> pd.DataFrame:
    """
    Liefert den Arbeits-DataFrame.
    - Wenn der CSV-Cache existiert und force_refresh=False: lade ihn.
    - Sonst: baue neu aus Quellen (reBAP, ID1, BE, MStR, Futures) und speichere.
    """
    cache = Path(DATA_CSV).resolve()

    if not force_refresh and cache.exists():
        print("[get_data] Cache gefunden -> lade:", cache)
        return load_data(cache)

    print("[get_data] Baue neu aus Quellen… (force_refresh =", force_refresh, ")")

    # Defaults für Quellen (DATA_DIR)
    rebap_csv   = Path(rebap_csv   or (DATA_DIR / "reBAP_utc.csv")).resolve()
    id1_xlsx    = Path(id1_xlsx    or (DATA_DIR / "id1_price_utc.xlsx")).resolve()
    be_csv      = Path(be_csv      or (DATA_DIR / "Belgium_elia.csv")).resolve()
    mstr_csv    = Path(mstr_csv    or (DATA_DIR / "marktstammdatenregister_windleistung_transnetbw.csv")).resolve()
    hochrechnung_csv = Path(hochrechnung_csv or (DATA_DIR / "Netztransparenz_WindOnshore_Hochrechnungen.csv")).resolve()
    futures_dir = Path(futures_dir or (DATA_DIR / "futures")).resolve()

    # Existenz prüfen
    missing = []
    for p in [rebap_csv, id1_xlsx, be_csv, mstr_csv, hochrechnung_csv]:
        if not p.exists():
            missing.append(str(p))
    if not futures_dir.exists():
        missing.append(str(futures_dir) + " (Ordner)")
    if missing:
        raise FileNotFoundError(
            "Quell-Dateien/Ordner fehlen für Neuaufbau:\n  - " + "\n  - ".join(missing)
        )

    print("[get_data] Quellen:")
    print("  rebap_csv  :", rebap_csv)
    print("  id1_xlsx   :", id1_xlsx)
    print("  be_csv     :", be_csv)
    print("  mstr_csv   :", mstr_csv)
    print("  hochrechnung_csv:", hochrechnung_csv)
    print("  futures_dir:", futures_dir)

    # Neu bauen
    df = build_data_from_sources(
        rebap_csv=rebap_csv,
        id1_xlsx=id1_xlsx,
        be_csv=be_csv,
        mstr_csv=mstr_csv,
        hochrechnung_csv=hochrechnung_csv,
        futures_dir=futures_dir,
        years=years,
    )
    print("[get_data] gebaut:", df.shape)

    # Wir importieren die Spalten die gemäß der data_processing.py erstellt werden
    df = process_data(df, DATA_DIR)
    print("[get_data] nach process_data:", df.shape)
    
    # Speichern
    save_data(df, cache)
    return df
