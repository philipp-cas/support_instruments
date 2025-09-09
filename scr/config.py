# src/config.py
from __future__ import annotations
from pathlib import Path
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# .env laden (falls vorhanden)
load_dotenv()

# -------------------------------------------------------------------
# Projektpfade
# -------------------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parents[1]   # Projekt-Root (enthält src/, data/, scripts/, ...)
DATA_DIR   = ROOT_DIR / "data"                     # <-- Daten liegen direkt in project-root/data
RESULTS_DIR= ROOT_DIR / "results"                  # Ausgabe-Verzeichnis (CSV/Plots/etc.)
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CSV-Cache
DATA_CSV = DATA_DIR / "data_final.csv"

# -------------------------------------------------------------------
# Zeitraum
# -------------------------------------------------------------------
START_DATE = "2022-03-01 00:00:00"
END_DATE   = "2024-12-31 23:59:59"

# -------------------------------------------------------------------
# Spaltenkonventionen
# -------------------------------------------------------------------
FORECAST_DA_COL  = "asset_Wind Onshore_da"
ACTUAL_COL       = "asset_Wind Onshore_act"
DA_PRICE_COL     = "da_price"
REBAP_COL        = "rebap"
FORECAST_ID_COL  = "asset_Wind Onshore_id"
ID_PRICE_COL     = "id1_price"
MV_REAL_COL      = "Wind Onshore_marketvalue"

# -------------------------------------------------------------------
# Regime-Parameter (Defaults)
# -------------------------------------------------------------------
P_FIT    = 70.0
CFD_K    = 70.0
CFD_VIEW = "standard"   # "standard" | "transparent"
MPM_AW   = 70.0

# -------------------------------------------------------------------
# (Optional) DB-Credentials aus .env / Umgebung
# -------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASS", "")

def get_engine():
    """
    SQLAlchemy Engine für die ENTSO-E Datenbank (nur nötig, wenn du echte
    DB-Queries nutzt). Wirft einen aussagekräftigen Fehler, wenn Variablen fehlen.
    """
    missing = [k for k, v in {
        "DB_HOST": DB_HOST, "DB_NAME": DB_NAME, "DB_USER": DB_USER, "DB_PASS": DB_PASS
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"Fehlende DB-Variablen: {', '.join(missing)}. "
            "Bitte .env ergänzen oder get_engine() nicht verwenden."
        )
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True)
