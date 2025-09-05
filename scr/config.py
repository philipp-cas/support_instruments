from pathlib import Path
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# .env laden (falls vorhanden)
load_dotenv()

# Projektstruktur
ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT_DIR / "base"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CSV-Cache
DATA_CSV = DATA_DIR / "data_final.csv"

# Zeitraum
START_DATE = "2022-01-01 00:00:00"
END_DATE   = "2024-12-31 23:59:59"

# Spalten
FORECAST_DA_COL  = "asset_Wind Onshore_da"
ACTUAL_COL       = "asset_Wind Onshore_act"
DA_PRICE_COL     = "da_price"
REBAP_COL        = "rebap"
FORECAST_ID_COL  = "asset_Wind Onshore_id"
ID_PRICE_COL     = "id1_price"
MV_REAL_COL      = "Wind Onshore_marketvalue"

# Parameter Defaults
P_FIT   = 70.0
CFD_K   = 70.0
CFD_VIEW = "standard"
MPM_AW  = 70.0

# Datenbank-Credentials (aus .env oder Umgebung)
DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASS", "")

def get_engine():
    """SQLAlchemy Engine für die ENTSO-E Datenbank."""
    missing = [k for k, v in {
        "DB_HOST": DB_HOST, "DB_NAME": DB_NAME, "DB_USER": DB_USER, "DB_PASS": DB_PASS
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Fehlende DB-Variablen: {', '.join(missing)}. Bitte .env ergänzen.")
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True)
