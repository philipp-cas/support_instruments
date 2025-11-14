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
from src.config import DATA_CSV

def main() -> None:
    print("CWD:", Path.cwd())
    df = get_data(force_refresh=True)  # nutzt DATA_DIR/reBAP_utc.csv, ... und DATA_DIR/futures
    print("Cache gebaut ✅")
    print("Shape:", df.shape)
    print("Zeitraum:", df.index.min(), "→", df.index.max())
    print(f"Gespeichert unter: {DATA_CSV}")

if __name__ == "__main__":
    main()