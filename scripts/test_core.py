<<<<<<< HEAD
# scripts/test_core.py
# =============================================================================
# Smoke Test für das Simulation Framework
# =============================================================================
# Zweck
# -----
# - Schnellprüfung, ob das Framework lauffähig ist:
#   * Import der Projektmodule funktioniert (src/*, scripts/*).
#   * Daten-Cache (`base/data/data_final.csv`) kann geladen werden.
#   * Eine Beispielsimulation (CFD, DA_only) läuft durch.
#   * Totals stimmen mit den Details überein (Summencheck).
#
# Kontext
# -------
# - Dieser Test ist KEIN Ersatz für Unit- oder Integrationstests, sondern ein
#   pragmatischer "Smoke Test" – d. h. er prüft, ob der Pipeline-End-to-End-
#   Durchlauf ohne Fehler möglich ist.
# - Typische Nutzung:
#     python scripts/test_core.py
#
# Erwartetes Ergebnis
# -------------------
# - Ausgabe von Basisinformationen zum Daten-Cache (Zeilenanzahl, Indexbereich).
# - Totals-Dict einer Beispielsimulation (CFD, DA_only).
# - Erste Zeilen des Details-DataFrames.
# - Erfolgsnachricht mit [OK].
# =============================================================================

# --- Bootstrap: Projekt-Root auf sys.path legen (für VS Code / Direktaufruf) ---
from pathlib import Path
import sys, os

ROOT = Path(__file__).resolve().parents[1]  # Ordner, der 'src/' und 'scripts/' enthält
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))
# -------------------------------------------------------------------------------

# Import der Projektmodule
from src.data_store import get_data
from src.reduced_strategies_call import run_one
from src.config import (
    FORECAST_DA_COL, ACTUAL_COL, DA_PRICE_COL, REBAP_COL,
    FORECAST_ID_COL, ID_PRICE_COL, START_DATE, END_DATE
)

# ---------------------------------------------------------------------------
def main():
    print("=== Smoke Test gestartet ===")

    # 1) Daten laden (aus Cache oder neu bauen)
    # Hinweis: Falls data_final.csv noch nicht existiert:
    #   python scripts/build_cache.py
    df = get_data(force_refresh=False)

    print(f"[OK] Daten geladen: {len(df):,} Zeilen, "
          f"Index von {df.index.min()} bis {df.index.max()}")

    # 2) Einfache Simulation: CFD, DA_only
    details, totals = run_one("CFD", use_da_id=False)

    print("\n[OK] Beispielsimulation (CFD, DA_only) erfolgreich.")
    print("Totals (Ausschnitt):")
    for k, v in list(totals.items())[:10]:  # nur erste 10 Keys zeigen
        print(f"  {k:20s}: {v}")

    print("\nDetails (erste 5 Zeilen):")
    print(details.head())

    # 3) Plausiprüfung
    assert "Netto_€" in details.columns, "Spalte 'Netto_€' fehlt in Details!"
    assert abs(details["Netto_€"].sum() - totals["Netto_€"]) < 1e-6, \
        "Totals stimmen nicht mit Details überein!"

    print("\n[OK] Smoke Test abgeschlossen – alles sieht gut aus.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
=======
# scripts/test_core.py
# =============================================================================
# Smoke Test für das Simulation Framework
# =============================================================================
# Zweck
# -----
# - Schnellprüfung, ob das Framework lauffähig ist:
#   * Import der Projektmodule funktioniert (src/*, scripts/*).
#   * Daten-Cache (`base/data/data_final.csv`) kann geladen werden.
#   * Eine Beispielsimulation (CFD, DA_only) läuft durch.
#   * Totals stimmen mit den Details überein (Summencheck).
#
# Kontext
# -------
# - Dieser Test ist KEIN Ersatz für Unit- oder Integrationstests, sondern ein
#   pragmatischer "Smoke Test" – d. h. er prüft, ob der Pipeline-End-to-End-
#   Durchlauf ohne Fehler möglich ist.
# - Typische Nutzung:
#     python scripts/test_core.py
#
# Erwartetes Ergebnis
# -------------------
# - Ausgabe von Basisinformationen zum Daten-Cache (Zeilenanzahl, Indexbereich).
# - Totals-Dict einer Beispielsimulation (CFD, DA_only).
# - Erste Zeilen des Details-DataFrames.
# - Erfolgsnachricht mit [OK].
# =============================================================================

# --- Bootstrap: Projekt-Root auf sys.path legen (für VS Code / Direktaufruf) ---
from pathlib import Path
import sys, os

ROOT = Path(__file__).resolve().parents[1]  # Ordner, der 'src/' und 'scripts/' enthält
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))
# -------------------------------------------------------------------------------

# Import der Projektmodule
from src.data_store import get_data
from src.reduced_strategies_call import run_one
from src.config import (
    FORECAST_DA_COL, ACTUAL_COL, DA_PRICE_COL, REBAP_COL,
    FORECAST_ID_COL, ID_PRICE_COL, START_DATE, END_DATE
)

# ---------------------------------------------------------------------------
def main():
    print("=== Smoke Test gestartet ===")

    # 1) Daten laden (aus Cache oder neu bauen)
    # Hinweis: Falls data_final.csv noch nicht existiert:
    #   python scripts/build_cache.py
    df = get_data(force_refresh=False)

    print(f"[OK] Daten geladen: {len(df):,} Zeilen, "
          f"Index von {df.index.min()} bis {df.index.max()}")

    # 2) Einfache Simulation: CFD, DA_only
    details, totals = run_one("CFD", use_da_id=False)

    print("\n[OK] Beispielsimulation (CFD, DA_only) erfolgreich.")
    print("Totals (Ausschnitt):")
    for k, v in list(totals.items())[:10]:  # nur erste 10 Keys zeigen
        print(f"  {k:20s}: {v}")

    print("\nDetails (erste 5 Zeilen):")
    print(details.head())

    # 3) Plausiprüfung
    assert "Netto_€" in details.columns, "Spalte 'Netto_€' fehlt in Details!"
    assert abs(details["Netto_€"].sum() - totals["Netto_€"]) < 1e-6, \
        "Totals stimmen nicht mit Details überein!"

    print("\n[OK] Smoke Test abgeschlossen – alles sieht gut aus.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
>>>>>>> 6bc1d56 (general update)
