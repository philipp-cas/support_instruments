# scripts/build_cache.py
from pathlib import Path
from src.data_store import get_data

# Pfade innerhalb von base/data
rebap_path = Path("base/data/reBAP_utc.csv")
id1_path   = Path("base/data/id1_price_utc.xlsx")

df = get_data(force_refresh=True,
              rebap_csv=rebap_path,
              id1_xlsx=id1_path)

print("Cache gebaut:", df.shape, df.index.min(), "→", df.index.max())
