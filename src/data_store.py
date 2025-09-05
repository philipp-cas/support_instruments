<<<<<<< HEAD
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
#    → baut den DataFrame mit `build_data_from_sources()`, speichert ihn
#      als `data/data_final.csv` und gibt ihn zurück.
#
# 2. Danach (Standard):
#       df = get_data()
#    → lädt direkt den Cache `data/data_final.csv`.
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
) -> pd.DataFrame:
    """
    Liefert den Arbeits-DataFrame.

    - Wenn der CSV-Cache (DATA_CSV) existiert und force_refresh=False: lade ihn.
    - Andernfalls: baue neu aus Quellen (rebap_csv, id1_xlsx), speichere, gib zurück.

    Hinweise:
    - rebap_csv und id1_xlsx sind nur nötig, wenn neu gebaut wird.
    - Standardmäßig liegen die Dateien im DATA_DIR (siehe config.py).
    """
    cache = Path(DATA_CSV).resolve()

    if not force_refresh and cache.exists():
        print("[get_data] Cache gefunden -> lade:", cache)
        return load_data(cache)

    print("[get_data] Baue neu aus Quellen… (force_refresh =", force_refresh, ")")

    # Defaults für Quellen, falls nicht explizit übergeben
    if rebap_csv is None:
        rebap_csv = Path(DATA_DIR) / "reBAP_utc.csv"
    if id1_xlsx is None:
        id1_xlsx = Path(DATA_DIR) / "id1_price_utc.xlsx"

    rebap_csv = Path(rebap_csv).resolve()
    id1_xlsx = Path(id1_xlsx).resolve()

    # Vorher prüfen und klare Fehlermeldungen geben
    missing = []
    if not rebap_csv.exists():
        missing.append(str(rebap_csv))
    if not id1_xlsx.exists():
        missing.append(str(id1_xlsx))
    if missing:
        raise FileNotFoundError(
            "Quell-Dateien fehlen für Neuaufbau:\n  - " + "\n  - ".join(missing)
        )

    print("[get_data] Quellen:")
    print("  rebap_csv:", rebap_csv)
    print("  id1_xlsx :", id1_xlsx)

    # Neu bauen
    df = build_data_from_sources(rebap_csv=rebap_csv, id1_xlsx=id1_xlsx)
    print("[get_data] gebaut:", df.shape)

    # Speichern
    save_data(df, cache)
    return df
=======
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
#    → baut den DataFrame mit `build_data_from_sources()`, speichert ihn
#      als `data/data_final.csv` und gibt ihn zurück.
#
# 2. Danach (Standard):
#       df = get_data()
#    → lädt direkt den Cache `data/data_final.csv`.
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
) -> pd.DataFrame:
    """
    Liefert den Arbeits-DataFrame.

    - Wenn der CSV-Cache (DATA_CSV) existiert und force_refresh=False: lade ihn.
    - Andernfalls: baue neu aus Quellen (rebap_csv, id1_xlsx), speichere, gib zurück.

    Hinweise:
    - rebap_csv und id1_xlsx sind nur nötig, wenn neu gebaut wird.
    - Standardmäßig liegen die Dateien im DATA_DIR (siehe config.py).
    """
    cache = Path(DATA_CSV).resolve()

    if not force_refresh and cache.exists():
        print("[get_data] Cache gefunden -> lade:", cache)
        return load_data(cache)

    print("[get_data] Baue neu aus Quellen… (force_refresh =", force_refresh, ")")

    # Defaults für Quellen, falls nicht explizit übergeben
    if rebap_csv is None:
        rebap_csv = Path(DATA_DIR) / "reBAP_utc.csv"
    if id1_xlsx is None:
        id1_xlsx = Path(DATA_DIR) / "id1_price_utc.xlsx"

    rebap_csv = Path(rebap_csv).resolve()
    id1_xlsx = Path(id1_xlsx).resolve()

    # Vorher prüfen und klare Fehlermeldungen geben
    missing = []
    if not rebap_csv.exists():
        missing.append(str(rebap_csv))
    if not id1_xlsx.exists():
        missing.append(str(id1_xlsx))
    if missing:
        raise FileNotFoundError(
            "Quell-Dateien fehlen für Neuaufbau:\n  - " + "\n  - ".join(missing)
        )

    print("[get_data] Quellen:")
    print("  rebap_csv:", rebap_csv)
    print("  id1_xlsx :", id1_xlsx)

    # Neu bauen
    df = build_data_from_sources(rebap_csv=rebap_csv, id1_xlsx=id1_xlsx)
    print("[get_data] gebaut:", df.shape)

    # Speichern
    save_data(df, cache)
    return df
>>>>>>> 6bc1d56 (general update)
