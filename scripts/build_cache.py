<<<<<<< HEAD
# scripts/build_cache.py
from __future__ import annotations
from pathlib import Path
import sys, os

# --- Projektroot robust ermitteln (Ordner, der 'src' und 'data' enthält) ---
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))
# Optional: CWD auf Projektroot setzen, damit relative Pfade wie "data/..." sicher sind
os.chdir(ROOT)

from src.data_store import get_data
from src.config import DATA_DIR

def main() -> None:
    rebap_path = DATA_DIR / "reBAP_utc.csv"
    id1_path   = DATA_DIR / "id1_price_utc.xlsx"

    # Debug-Ausgaben helfen, falls es doch mal klemmt
    print("CWD:", Path.cwd())
    print("ReBAP:", rebap_path, "exists:", rebap_path.exists())
    print("ID1:  ", id1_path,   "exists:", id1_path.exists())

    if not rebap_path.exists() or not id1_path.exists():
        raise FileNotFoundError(
            "Rohdaten fehlen:\n"
            f"  - {rebap_path}\n"
            f"  - {id1_path}\n"
            "Bitte Dateien in den data/ Ordner legen."
        )

    df = get_data(force_refresh=True, rebap_csv=rebap_path, id1_xlsx=id1_path)

    print("Cache gebaut ✅")
    print("Shape:", df.shape)
    print("Zeitraum:", df.index.min(), "→", df.index.max())
    print(f"Gespeichert unter: {DATA_DIR / 'data_final.csv'}")

if __name__ == "__main__":
    main()
=======
# scripts/build_cache.py
from __future__ import annotations
from pathlib import Path
import sys, os

# --- Projektroot robust ermitteln (Ordner, der 'src' und 'data' enthält) ---
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))
# Optional: CWD auf Projektroot setzen, damit relative Pfade wie "data/..." sicher sind
os.chdir(ROOT)

from src.data_store import get_data
from src.config import DATA_DIR

def main() -> None:
    rebap_path = DATA_DIR / "reBAP_utc.csv"
    id1_path   = DATA_DIR / "id1_price_utc.xlsx"

    # Debug-Ausgaben helfen, falls es doch mal klemmt
    print("CWD:", Path.cwd())
    print("ReBAP:", rebap_path, "exists:", rebap_path.exists())
    print("ID1:  ", id1_path,   "exists:", id1_path.exists())

    if not rebap_path.exists() or not id1_path.exists():
        raise FileNotFoundError(
            "Rohdaten fehlen:\n"
            f"  - {rebap_path}\n"
            f"  - {id1_path}\n"
            "Bitte Dateien in den data/ Ordner legen."
        )

    df = get_data(force_refresh=True, rebap_csv=rebap_path, id1_xlsx=id1_path)

    print("Cache gebaut ✅")
    print("Shape:", df.shape)
    print("Zeitraum:", df.index.min(), "→", df.index.max())
    print(f"Gespeichert unter: {DATA_DIR / 'data_final.csv'}")

if __name__ == "__main__":
    main()
>>>>>>> 6bc1d56 (general update)
