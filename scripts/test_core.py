# scripts/test_core.py
# =============================================================================
# Allgemeiner Smoke Test für das Simulation Framework
# =============================================================================
# Zweck
# -----
# - Schnellprüfung, ob das Framework end-to-end funktionsfähig ist:
#   * Projektmodule (src/*, scripts/*) können importiert werden.
#   * Daten-Cache (`data/data_final.csv`) kann geladen oder neu gebaut werden.
#   * Basisdaten besitzen plausiblen Zeitindex (15min oder 60min).
#   * Eine Beispielsimulation (CFD, Day-Ahead only) läuft fehlerfrei durch.
#   * Gesamtergebnisse (Totals) stimmen mit den Detailwerten überein.
#
# Kontext
# -------
# - Dieser Test dient als pragmatische Systemprüfung („Smoke Test“),
#   um sicherzustellen, dass alle zentralen Komponenten – Datenimport,
#   Caching, Simulation, Berechnung – ohne Fehlermeldungen funktionieren.
# - Er ersetzt keine Unit- oder Integrationstests, sondern prüft lediglich,
#   ob die vollständige Pipeline lauffähig ist.
#
# Typische Nutzung
# ----------------
#     python scripts/test_core.py
#
# Erwartetes Ergebnis
# -------------------
# - Ausgabe grundlegender Informationen zum geladenen Datensatz
#   (Zeilenanzahl, Indexbereich, Spaltenauszug, Zeitraster).
# - Kurze Übersicht über die berechneten Gesamtergebnisse der Simulation.
# - Anzeige der ersten Zeilen der Detailtabelle.
# - Abschlussmeldung:
#       [OK] Smoke Test abgeschlossen – alles sieht gut aus.
# =============================================================================

# scripts/test_core.py
# =============================================================================
# Allgemeiner Smoke Test fürs Framework (ohne futures-spezifische Checks)
# =============================================================================
from pathlib import Path
import sys, os
import pandas as pd
import numpy as np

# --- Projektroot auf sys.path setzen ---
try:
    # Wenn Skript ausgeführt wird → nimm Ordner eine Ebene über der Datei
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Wenn __file__ fehlt (z. B. Interactive Window) → nimm Arbeitsverzeichnis
    ROOT = Path.cwd().resolve()

# Root dem Python-Pfad hinzufügen, falls noch nicht vorhanden
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Projektmodule ---
from src.data_store import get_data, load_data
from src.reduced_strategies_call import run_one
from src.config import (
    FORECAST_DA_COL, ACTUAL_COL, DA_PRICE_COL, REBAP_COL,
    FORECAST_ID_COL, ID_PRICE_COL, START_DATE, END_DATE, DATA_CSV
)

def _ensure_cache_or_build() -> pd.DataFrame:
    """Lädt Cache, sonst baut neu mit DATA_DIR-Defaults."""
    cache = Path(DATA_CSV)
    if cache.exists():
        print(f"[Info] Cache gefunden: {cache}")
        return load_data(cache)
    print("[Info] Kein Cache -> baue neu (DATA_DIR-Defaults)…")
    return get_data(force_refresh=True)

def main():
    print("=== Smoke Test gestartet ===")

    # 1) Daten laden (Cache oder Neuaufbau)
    df = _ensure_cache_or_build()
    print(f"[OK] Daten geladen: {len(df):,} Zeilen, "
          f"Index von {df.index.min()} bis {df.index.max()}")
    print(f"[OK] Spalten (Auszug): {', '.join(df.columns[:12])} …")

    # 2) Grober Frequenz-Check (nur Info/Heuristik)
    inferred = pd.infer_freq(df.index[:400])
    if inferred:
        print(f"[Info] erkannte Frequenz (Stichprobe): {inferred}")
    else:
        # Fallback-Heuristik: häufigster Abstand
        most_common_delta = df.index.to_series().diff().dropna().value_counts().idxmax()
        print(f"[Info] häufigster Zeitabstand: {most_common_delta}")
        assert most_common_delta in {pd.Timedelta(minutes=15), pd.Timedelta(hours=1)}, \
            "Unerwartetes Zeitraster (weder 15min noch 60min)."

    # 3) Kernspalten (nur warnen, nicht hart failen)
    expected_candidates = [DA_PRICE_COL, ID_PRICE_COL, REBAP_COL]
    missing_soft = [c for c in expected_candidates if c not in df.columns]
    if missing_soft:
        print("⚠️  Hinweis: folgende erwartete Spalten fehlen (Test läuft weiter):",
              ", ".join(missing_soft))

    # --- Zusatz: neue Spalten/Estimatoren (nur Info) --------------------------
    new_cols = [
        "Wind Onshore_marketvalue",              # realer Monats-MV (€/MWh)
        "Wind Onshore_marketvalue_est",          # Monats-Schätzer (as of 11:00)
        "Wind Onshore_marketvalue_year_est",     # Jahres-Schätzer (as of 11:00)
        "future_current_month",                  # Alias zu monthly_future_base
        "monthly_future_base",
        "DE_Wind Onshore_hochrechnung",
    ]
    have_new = [c for c in new_cols if c in df.columns]
    if have_new:
        na_share = df[have_new].isna().mean().sort_values(ascending=False)
        print("\n[Info] Anteil fehlender Werte (neue Spalten – Auszug):")
        print(na_share.head(10).to_string())

        # Kurzer Plausi-Vergleich Monats-MV: Schätzer vs. real (täglich)
        if {"Wind Onshore_marketvalue", "Wind Onshore_marketvalue_est"} <= set(df.columns):
            mv_true_d = df["Wind Onshore_marketvalue"].resample("D").mean()
            mv_est_d  = df["Wind Onshore_marketvalue_est"].resample("D").mean()
            overlap = pd.concat([mv_true_d, mv_est_d], axis=1).dropna()
            if not overlap.empty:
                mae = float(np.mean(np.abs(overlap.iloc[:,1] - overlap.iloc[:,0])))
                print(f"[Info] Monats-MV: Tages-MAE (est vs. real) = {mae:.2f} €/MWh")

    # 4) Beispielsimulation (CFD, DA_only)
    details, totals = run_one("CFD", use_da_id=False)
    print("\n[OK] Beispielsimulation (CFD, DA_only) erfolgreich.")
    print("Totals (Ausschnitt):")
    for k, v in list(totals.items())[:10]:
        print(f"  {k:22s}: {v}")

    print("\nDetails (erste 5 Zeilen):")
    print(details.head())

    # 5) Summenkonsistenz (harte Prüfung)
    assert "Netto_€" in details.columns, "Spalte 'Netto_€' fehlt in Details!"
    diff = abs(details["Netto_€"].sum() - totals.get("Netto_€", 0))
    assert diff < 1e-6, f"Totals stimmen nicht mit Details überein! Abweichung={diff}"

    # --- Zusatz: zweite Beispielsimulation (MPM mit Monats-Schätzer), optional
    if "Wind Onshore_marketvalue" in df.columns and "Wind Onshore_marketvalue_est" in df.columns:
        try:
            details_mpm, totals_mpm = run_one(
                "MPM",
                use_da_id=False,
                # explizit mit Monats-Schätzer entscheiden:
                market_value_col="Wind Onshore_marketvalue",
                market_value_est_col="Wind Onshore_marketvalue_est",
                mpm_aw=60.0,
            )
            print("\n[OK] Beispielsimulation (MPM, DA_only, mit Monats-Schätzer) erfolgreich.")
            for k, v in list(totals_mpm.items())[:8]:
                print(f"  {k:22s}: {v}")
        except Exception as e:
            print("\n⚠️  MPM-Test (mit Monats-Schätzer) konnte nicht ausgeführt werden:", repr(e))
    else:
        print("\nℹ️  MPM-Test übersprungen (Schätzer/Real-MV-Spalten fehlen).")

    print("\n[OK] Smoke Test abgeschlossen – alles sieht gut aus.")

if __name__ == "__main__":
    main()
