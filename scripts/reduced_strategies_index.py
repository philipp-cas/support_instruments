# scripts/reduced_strategies_index.py
# =============================================================================
# Modul: Sensitivitätsanalyse & Market Conformity Index (0..1)
# -----------------------------------------------------------------------------
# Zweck
# -----
# Dieses Skript vergleicht Viertelstunden-Entscheidungen ("decision_qh") zweier
# Strategien mittels eines Ähnlichkeitsindex (Market Conformity Index, 0..1).
#  - 1.0  => Entscheidungen identisch
#  - 0.0  => Entscheidungen komplett gegensätzlich
#
# Zusätzlich wird eine Sensitivitätsanalyse über Parameter-Deltas (Δ) für
# verschiedene Förderregime (FIT, CFD, MPM, MPM_EX) durchgeführt und grafisch
# ausgewertet (Linienplot, Heatmap, Boxplots). Außerdem werden finanzielle
# Sensitivitäten (Netto_€, Förderung_€) gesammelt und geplottet.
#
# Voraussetzungen
# ---------------
# - `src/reduced_strategies.py` stellt `reduced_strategies(...)` bereit.
# - `src/data_store.py` stellt `get_data()` bereit und liefert den gecachten
#   DataFrame aus `base/data/data_final.csv`.
# - Optional zentrale Konstanten/Parameter in `src/config.py`.
#
# Minimalbeispiel
# ---------------
# from src.data_store import get_data
# from src.reduced_strategies import reduced_strategies
# data = get_data()  # lädt base/data/data_final.csv
#
# # Referenzstrategie (z. B. "NO") berechnen:
# details_no, totals_no = reduced_strategies(
#     df=data, regime="NO",
#     start_date="2022-01-01 00:00:00", end_date="2024-12-31 23:59:59",
#     forecast_id_col="asset_Wind Onshore_id", id_price_col="id1_price"
# )
#
# # Vergleichsstrategie (z. B. FIT mit p_fit = 70):
# details_fit, totals_fit = reduced_strategies(
#     df=data, regime="FIT", p_fit=70.0,
#     start_date="2022-01-01 00:00:00", end_date="2024-12-31 23:59:59"
# )
#
# # Market Conformity Index:
# idx   = market_conformity_index(details_no, details_fit)
# idx_w = market_conformity_index(details_no, details_fit, weight_col="Act_raw_MW")
# =============================================================================

from __future__ import annotations

# ==== Imports =================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Core & Daten-Cache
from src.reduced_strategies import reduced_strategies
from src.data_store import get_data

# Zentrale Konstanten (du kannst sie auch lokal überschreiben, falls gewünscht)
from src.config import (
    START_DATE, END_DATE,
    FORECAST_ID_COL, ID_PRICE_COL,
    MV_REAL_COL as MV_COL,  # Alias für Konsistenz zum Skript
)

# ==== Spalten-Konventionen (könnten auch lokal gesetzt werden) ================
# Wenn du NICHT die config nutzen willst, kannst du hier die Konstanten auch
# direkt setzen. Standardmäßig kommen sie aus src/config.py (s.o.).
# FORECAST_ID_COL = "asset_Wind Onshore_id"
# ID_PRICE_COL    = "id1_price"
# MV_COL          = "Wind Onshore_marketvalue"

# ==== Market Conformity Index (0..1) ==========================================
def market_conformity_index(
    ref: pd.DataFrame,
    other: pd.DataFrame,
    weight_col: str | None = None  # z.B. "Act_raw_MW" – wenn None => ungewichtet
) -> float:
    """
    Ähnlichkeitsindex ∈ [0,1] für Viertelstunden-Entscheidungen.
    1 = identisch, 0 = komplett gegensätzlich.

    Verglichen wird ref['decision_qh'] gegen other['decision_qh'] auf dem
    gemeinsamen QH-Index. Optional gewichtet mit ref[weight_col].
    """
    # Beide DataFrames müssen zeitlich indiziert sein (DatetimeIndex)
    if not isinstance(ref.index, pd.DatetimeIndex) or not isinstance(other.index, pd.DatetimeIndex):
        raise ValueError("ref und other brauchen jeweils einen DatetimeIndex.")

    # Pflichtspalte prüfen (beide Seiten müssen 'decision_qh' haben)
    for name, df in {"ref": ref, "other": other}.items():
        if "decision_qh" not in df.columns:
            raise ValueError(f"'{name}' fehlt die Spalte 'decision_qh'.")

    # Schnittmenge des Zeitindex: nur gemeinsame Viertelstunden werden verglichen
    idx = ref.index.intersection(other.index)
    if len(idx) == 0:
        raise ValueError("ref und other haben keinen gemeinsamen Zeitindex.")

    # Entscheidungen als bool-Arrays
    a = ref.loc[idx, "decision_qh"].astype(bool).to_numpy()
    b = other.loc[idx, "decision_qh"].astype(bool).to_numpy()

    # Fehlrate pro QH (XOR = unterschiedlich)
    mismatch = np.logical_xor(a, b).astype(float)

    # Ungewichteter Index: 1 - durchschnittliche Fehlrate
    if weight_col is None:
        return float(1.0 - mismatch.mean())

    # Gewichteter Index: Gewichte aus ref[weight_col]
    if weight_col not in ref.columns:
        raise ValueError(f"Gewichtsspalte '{weight_col}' fehlt in ref.")

    w = (
        ref.loc[idx, weight_col]
           .astype(float)
           .replace([np.inf, -np.inf], np.nan)
           .fillna(0.0)
           .to_numpy()
    )

    w_sum = w.sum()
    # Hinweis: Verhalten wie im Original – bei unbrauchbaren Gewichten wird ein
    # ValueError-Objekt ZURÜCKGEGEBEN (nicht geworfen).
    if w_sum <= 0 or not np.isfinite(w_sum):
        return ValueError(f"Gewichtsspalte '{weight_col}' in ref ist unbrauchbar (Summe={w_sum}).")

    return float(1.0 - float(np.dot(mismatch, w)) / float(w_sum))


# ==== Plot-Funktionen: Market Conformity ======================================
def plot_sensitivity(sens_table: pd.DataFrame, title: str = "Market Conformity Sensitivity"):
    """Liniendiagramm: Conformity-Index über Δ pro Regime."""
    plt.figure(figsize=(10, 5))
    for reg, grp in sens_table.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp["MarketSimilarityIndex"], marker="o", label=reg)
    plt.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline des jeweiligen Parameters")
    plt.ylabel("Market Conformity Index (0…1)")
    plt.title(title + f" — Zeitraum: {pd.to_datetime(START_DATE).date()} bis {pd.to_datetime(END_DATE).date()}")
    plt.grid(alpha=0.35)
    plt.ylim(0.5, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmap(sens_table: pd.DataFrame):
    """Heatmap der Conformity-Werte je Regime und Δ."""
    pivot = sens_table.pivot(index="Delta", columns="Regime", values="MarketSimilarityIndex")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Conformity Index'})
    plt.title("Heatmap: Market Conformity bei verschiedenen Δ")
    plt.xlabel("Regime"); plt.ylabel("Δ (€/MWh)")
    plt.tight_layout(); plt.show()


def plot_boxplots(sens_table: pd.DataFrame):
    """Boxplots der Conformity-Verteilung pro Regime (mit Jitterpunkte)."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Regime", y="MarketSimilarityIndex", data=sens_table)
    sns.stripplot(x="Regime", y="MarketSimilarityIndex", data=sens_table,
                  color="black", alpha=0.5, jitter=True)
    plt.title("Verteilung der Market Conformity Indices pro Regime")
    plt.ylabel("Market Conformity Index"); plt.xlabel("Regime")
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# ==== Plot-Funktion: finanzielle Sensitivitäten ===============================
def plot_financial_sensitivity(df, value_col="Netto_€", title=None, ylim=None):
    """Liniendiagramm: finanzielle Kennzahl über Δ pro Regime."""
    plt.figure(figsize=(10,5))
    for reg, grp in df.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp[value_col], marker="o", label=reg)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline des jeweiligen Parameters")
    plt.ylabel(value_col.replace("_", " "))
    if title:
        plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==== Finanz-Helfer ===========================================================
def _sens_row(regime, delta, base, test, totals):
    """
    Hilfsfunktion: extrahiert Kennzahlen aus totals und baut eine Ergebniszeile.
    Förderung_€ = Summe der Förderkomponenten; negative Werte sind möglich.
    """
    # Förderung = SUMME der ausgewiesenen Förderkomponenten, auch negativ erlaubt
    support_eur = 0.0
    for k in ("FIT_€", "MPM_€", "CfD_€"):
        if k in totals:
            support_eur += float(totals[k])

    return {
        "Regime": regime,
        "Delta": delta,
        "BaseValue": base,
        "TestValue": test,
        "Netto_€": float(totals.get("Netto_€", 0.0)),
        "Revenue_DA_€": float(totals.get("Revenue_DA_€", 0.0)),
        "Revenue_ID_€": float(totals.get("Revenue_ID_€", 0.0)),
        "reBAP_€": float(totals.get("reBAP_€", 0.0)),
        "FIT_€": float(totals.get("FIT_€", 0.0)),
        "MPM_€": float(totals.get("MPM_€", 0.0)),
        "CfD_€": float(totals.get("CfD_€", 0.0)),
        "Förderung_€": support_eur,  # kann < 0 sein
        # optional sinnvoll für €/MWh-Plots:
        "Energie_Act_MWh": float(totals.get("Energie_Act_MWh", 0.0)),
        "Energie_Sched_MWh": float(totals.get("Energie_Sched_MWh", 0.0)),
    }


# ==== Main-Analyse ============================================================
def main():
    # --- Daten laden (CSV-Cache) ---
    data = get_data()  # lädt base/data/data_final.csv

    # --- Analyse-Zeitraum & Deltas ---
    START = START_DATE
    END   = END_DATE
    DELTAS = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70]

    # --- Referenz (NO) EINMAL rechnen – mit ID (beide Spalten setzen!) ---
    details_no, totals_no = reduced_strategies(
        df=data, regime="NO",
        start_date=START, end_date=END,
        forecast_id_col=FORECAST_ID_COL,   # <- ID-Entscheidung aktivieren
        id_price_col=ID_PRICE_COL
    )

    # Zusätzliche (optionale) Gewichtsspalte: Roh-Einspeisung * absoluter ID1-Preis
    #  - abs() vermeidet Vorzeichenkompensation bei negativen Preisen
    details_no["w_qh"] = (
        details_no["Act_raw_MW"].astype(float) *
        details_no["p_ID_€/MWh"].astype(float).abs()
    )

    # --- Sensitivität: Market Conformity ---
    results: list[dict] = []

    # FIT (klassisch): p_fit variieren
    BASE_FIT = 70.0
    for delta in DELTAS:
        test_fit = BASE_FIT + delta
        details_fit, _ = reduced_strategies(
            df=data, regime="FIT", p_fit=test_fit,
            start_date=START, end_date=END
            # kein ID-Trade im FIT-Zweig, da keine Marktteilnahme
        )
        idx = market_conformity_index(details_no, details_fit, weight_col="Act_raw_MW")
        results.append({
            "Regime": "FIT",
            "Delta": delta,
            "BaseValue": BASE_FIT,
            "TestValue": test_fit,
            "MarketSimilarityIndex": idx
        })

    # CfD: Strike variieren
    BASE_STRIKE = 70.0
    for delta in DELTAS:
        test_strike = BASE_STRIKE + delta
        details_cfd, _ = reduced_strategies(
            df=data, regime="CFD", cfd_strike=test_strike,
            # Darstellung (reporting/transparent) ändert die Entscheidung nicht – optional:
            # da_present_as_diff=True,
            start_date=START, end_date=END,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        idx = market_conformity_index(details_no, details_cfd, weight_col="Act_raw_MW")
        results.append({
            "Regime": "CFD",
            "Delta": delta,
            "BaseValue": BASE_STRIKE,
            "TestValue": test_strike,
            "MarketSimilarityIndex": idx
        })

    # MPM: anzulegender Wert (mpm_aw) variieren
    BASE_MPM_AW = 70.0
    for delta in DELTAS:
        test_aw = BASE_MPM_AW + delta
        details_mpm, _ = reduced_strategies(
            df=data, regime="MPM",
            mpm_aw=test_aw,
            market_value_col=MV_COL,          # reale Monats-Marktwerte
            market_value_est_col=None,        # Vormonat wird intern als Est genutzt
            start_date=START, end_date=END,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        idx = market_conformity_index(details_no, details_mpm, weight_col="Act_raw_MW")
        results.append({
            "Regime": "MPM",
            "Delta": delta,
            "BaseValue": BASE_MPM_AW,
            "TestValue": test_aw,
            "MarketSimilarityIndex": idx
        })

    # MPM_EX: anzulegender Wert (mpm_aw) variieren
    BASE_MPM_AW = 70.0
    for delta in DELTAS:
        test_aw = BASE_MPM_AW + delta
        details_mpm_ex, _ = reduced_strategies(
            df=data, regime="MPM_EX",
            mpm_aw=test_aw,
            market_value_col=MV_COL,          # reale Monats-Marktwerte
            market_value_est_col=None,        # Vormonat wird intern als Est genutzt
            start_date=START, end_date=END,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        idx = market_conformity_index(details_no, details_mpm_ex, weight_col="Act_raw_MW")
        results.append({
            "Regime": "MPM_EX",
            "Delta": delta,
            "BaseValue": BASE_MPM_AW,
            "TestValue": test_aw,
            "MarketSimilarityIndex": idx
        })

    # Ergebnis-Tabelle (Conformity)
    sens_df = pd.DataFrame(results).sort_values(["Regime", "Delta"]).reset_index(drop=True)
    print(sens_df.groupby("Regime")["MarketSimilarityIndex"].describe())

    # Plots erzeugen
    plot_sensitivity(sens_df, title="Sensitivität: FIT/FIT_PREMIUM (p_fit), CfD (K), MPM (A)")
    plot_heatmap(sens_df)
    plot_boxplots(sens_df)

    # ---------- Finanzielle Sensitivitäten sammeln ----------
    fin_results: list[dict] = []

    # FIT (klassisch)
    BASE_FIT = 70.0
    for delta in DELTAS:
        test_fit = BASE_FIT + delta
        _, totals = reduced_strategies(
            df=data, regime="FIT", p_fit=test_fit,
            start_date=START, end_date=END
        )
        fin_results.append(_sens_row("FIT", delta, BASE_FIT, test_fit, totals))

    # CfD (Strike sweep)
    BASE_STRIKE = 70.0
    for delta in DELTAS:
        test_strike = BASE_STRIKE + delta
        _, totals = reduced_strategies(
            df=data, regime="CFD", cfd_strike=test_strike,
            start_date=START, end_date=END,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        fin_results.append(_sens_row("CFD", delta, BASE_STRIKE, test_strike, totals))

    # MPM (A sweep)
    BASE_MPM_AW = 70.0
    for delta in DELTAS:
        test_aw = BASE_MPM_AW + delta
        _, totals = reduced_strategies(
            df=data, regime="MPM", mpm_aw=test_aw,
            market_value_col=MV_COL, market_value_est_col=None,
            start_date=START, end_date=END,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        fin_results.append(_sens_row("MPM", delta, BASE_MPM_AW, test_aw, totals))

    # MPM_EX (A sweep)
    BASE_MPM_AW = 70.0
    for delta in DELTAS:
        test_aw = BASE_MPM_AW + delta
        _, totals = reduced_strategies(
            df=data, regime="MPM_EX", mpm_aw=test_aw,
            market_value_col=MV_COL, market_value_est_col=None,
            start_date=START, end_date=END,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        fin_results.append(_sens_row("MPM_EX", delta, BASE_MPM_AW, test_aw, totals))

    fin_df = pd.DataFrame(fin_results).sort_values(["Regime","Delta"]).reset_index(drop=True)

    # Plots: finanzielle Sensitivität
    plot_financial_sensitivity(
        fin_df, value_col="Netto_€",
        title=f"Nettoerlöse — {pd.to_datetime(START).date()} bis {pd.to_datetime(END).date()}"
    )
    plot_financial_sensitivity(
        fin_df, value_col="Förderung_€",
        title=f"Erhaltene Förderung (Summe aller Komponenten) — {pd.to_datetime(START).date()} bis {pd.to_datetime(END).date()}"
    )


if __name__ == "__main__":
    main()
