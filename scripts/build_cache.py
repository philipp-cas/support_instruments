# scripts/build_cache.py
from pathlib import Path
from src.data_store import get_data

def main():
    # Rohdaten-Pfade (innerhalb von base/data abgelegt)
    rebap_path = Path("base/data/reBAP_utc.csv")
    id1_path   = Path("base/data/id1_price_utc.xlsx")

    # Cache neu erzeugen
    df = get_data(force_refresh=True,
                  rebap_csv=rebap_path,
                  id1_xlsx=id1_path)

    print("Cache gebaut")
    print("Shape:", df.shape)
    print("Zeitraum:", df.index.min(), "→", df.index.max())
    print("Gespeichert unter: base/data/data_final.csv")

if __name__ == "__main__":
    main()
