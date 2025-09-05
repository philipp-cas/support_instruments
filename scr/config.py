# src/config.py
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT_DIR / "base"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Nur CSV (optional komprimiert)
DATA_CSV = DATA_DIR / "data_final.csv"

# Zeitraum & Spalten/Parameter (Beispiele)
START_DATE = "2022-01-01 00:00:00"
END_DATE   = "2024-12-31 23:59:59"

FORECAST_DA_COL  = "asset_Wind Onshore_da"
ACTUAL_COL       = "asset_Wind Onshore_act"
DA_PRICE_COL     = "da_price"
REBAP_COL        = "rebap"
FORECAST_ID_COL  = "asset_Wind Onshore_id"
ID_PRICE_COL     = "id1_price"
MV_REAL_COL      = "Wind Onshore_marketvalue"

P_FIT   = 70.0
CFD_K   = 70.0
CFD_VIEW = "standard"
MPM_AW  = 70.0

# DB/Secrets aus Umgebungsvariablen (oder .env)
DB_HOST = os.getenv("DB_HOST", "132.252.60.112")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "ENTSOE")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASS", "")
