<<<<<<< HEAD
# =============================================================================
# Modul: metrics – Indizes & Sensitivitäten auf Basis der Simulationsdetails
# -----------------------------------------------------------------------------
# Zweck
# -----
# Dieses Modul baut auf den Ergebnissen von `reduced_strategies(...)` auf und
# liefert:
#   1) Market Conformity Index (0..1) zwischen zwei Strategien
#   2) Sensitivitätsanalysen über Parameter-Δ (FIT: p_fit; CFD: K; MPM/MPM_EX: A)
#      - a) Conformity-Sensitivität: Wie ähnlich sind die QH-Entscheidungen?
#      - b) Finanzielle Sensitivität: Wie verändern sich Netto/Komponenten?
#   3) Plot-Helfer für beide Sensitivitäten
#
# Einordnung im Projekt
# ---------------------
# - `reduced_strategies.py` enthält die **Kernlogik** (eine Strategie → details+totals).
# - `reduced_strategies_call.py` ist syntaktischer Zucker, um viele Läufe
#   bequem zu orchestrieren (z. B. alle Regime, DA_only/DA_ID).
# - **Dieses Modul** rechnet gezielt “Serien” von Läufen zum Vergleichen:
#   Referenzlauf(e) vs. Parameter-Sweeps; baut daraus Tabellen & Plots.
#
# Daten & Abhängigkeiten
# ----------------------
# - Eingabe ist ein DataFrame `df` mit DatetimeIndex (kommt typischerweise aus
#   `src.data_store.get_data()` → CSV-Cache).
# - Für MPM/MPM_EX erwartet `reduced_strategies(...)` realisierte Monatsmarktwerte
#   (config: `MV_REAL_COL`) und nutzt als Estimator **Vormonat** (oder eine explizite
#   Estimator-Spalte, wenn angegeben).
# - Conformity-Referenz ist standardmäßig **NO mit ID** (Intraday aktiv), d. h.
#   wir übergeben `forecast_id_col=config.FORECAST_ID_COL` und
#   `id_price_col=config.ID_PRICE_COL`.
#
# Kompatibilität zur Core-Logik
# -----------------------------
# - Parameter heißen identisch wie in `reduced_strategies(...)` (`p_fit`, `cfd_strike`,
#   `mpm_aw`, `market_value_col`, `market_value_est_col`, `forecast_id_col`, `id_price_col`).
# - ID ist nur aktiv, wenn **beide** Spalten gesetzt sind (wie im Core).
# - MPM/MPM_EX: Auszahlung mit realem Monats-MV; Entscheidung mit Estimator (hier: Vormonat),
#   exakt wie im Core beschrieben.
#
# Outputs
# -------
# - `conformity_sensitivity(...)` → DataFrame je Regime/Δ mit `MarketSimilarityIndex`
# - `financial_sensitivity(...)` → DataFrame je Regime/Δ mit Netto & Komponenten
# - Plotfunktionen (Lines/Heatmap für Conformity, Lines für Finanzen)
#
# Minimalbeispiel
# ---------------
# from src.data_store import get_data
# from src.metrics import conformity_sensitivity, financial_sensitivity, \
#     plot_conformity_lines, plot_conformity_heatmap, plot_fin_sensitivity
# import src.config as cfg
#
# cfg.START_DATE, cfg.END_DATE = "2024-01-01 00:00:00", "2024-12-31 23:59:59"
# df = get_data()
#
# sens = conformity_sensitivity(df, start=cfg.START_DATE, end=cfg.END_DATE,
#                               deltas=range(-50, 51, 10), use_abs_id_price_weight=True)
# plot_conformity_lines(sens, "Conformity 2024")
# plot_conformity_heatmap(sens, "Conformity 2024 – Heatmap")
#
# fin = financial_sensitivity(df, start=cfg.START_DATE, end=cfg.END_DATE,
#                             deltas=range(-50, 51, 10))
# plot_fin_sensitivity(fin, value_col="Netto_€", title="Netto – Sensitivität 2024")
# plot_fin_sensitivity(fin, value_col="Förderung_€", title="Förderung – Sensitivität 2024")
#
# Typische Stolpersteine
# ----------------------
# - Fehlende ID-Spalten → Conformity-Referenz NO MUSS mit ID laufen (sonst wenig Aussage).
# - Matplotlib/Seaborn müssen installiert sein (requirements).
# - Zeiträume: START/END konsistent halten, sonst vergleicht man unterschiedliche Fenster.
# - Gewichtung beim Conformity: Standard ist ungewicktet; optional
#   `Act_raw_MW * |p_ID_€/MWh|` (robuster bei negativen Preisen).
# =============================================================================


# src/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Iterable, Optional, Tuple

from .reduced_strategies import reduced_strategies
from .config import (
    FORECAST_ID_COL,
    ID_PRICE_COL,
    MV_REAL_COL as MV_COL,   # realisierte Monats-Marktwerte
    START_DATE,
    END_DATE,
    P_FIT,
    CFD_K,
    MPM_AW,
)

# ---------------------------------------------------------------------
# 0) Market Conformity Index (kompatibel zu deiner Logik)
# ---------------------------------------------------------------------
def market_conformity_index(
    ref: pd.DataFrame,
    other: pd.DataFrame,
    weight_col: str | None = None
) -> float:
    if not isinstance(ref.index, pd.DatetimeIndex) or not isinstance(other.index, pd.DatetimeIndex):
        raise ValueError("ref/other brauchen DatetimeIndex.")
    if "decision_qh" not in ref.columns or "decision_qh" not in other.columns:
        raise ValueError("Beide DFs brauchen 'decision_qh'.")

    idx = ref.index.intersection(other.index)
    if len(idx) == 0:
        raise ValueError("ref und other haben keinen gemeinsamen Zeitbereich.")

    a = ref.loc[idx, "decision_qh"].astype(bool).to_numpy()
    b = other.loc[idx, "decision_qh"].astype(bool).to_numpy()
    mismatch = np.logical_xor(a, b).astype(float)

    if weight_col is None:
        return float(1.0 - mismatch.mean())

    if weight_col not in ref.columns:
        # Fallback: ungewicktet, statt Exception
        return float(1.0 - mismatch.mean())

    w = (ref.loc[idx, weight_col]
           .astype(float)
           .replace([np.inf, -np.inf], np.nan)
           .fillna(0.0)
           .to_numpy())
    w_sum = w.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        return float(1.0 - mismatch.mean())
    return float(1.0 - float(np.dot(mismatch, w)) / float(w_sum))


# ---------------------------------------------------------------------
# 1) Sensitivität – Market Conformity (Δ wird wirklich angewandt)
#    Referenz = NO (mit ID)
# ---------------------------------------------------------------------
def conformity_sensitivity(
    df: pd.DataFrame,
    start: str = START_DATE,
    end: str = END_DATE,
    deltas: Iterable[int] = (-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70),
    weight: str | None = None,
) -> pd.DataFrame:
    """
    Berechnet den Market Conformity Index (0..1) für FIT/CFD/MPM/MPM_EX über Δ.
    Referenz = NO (mit ID). Optional gewichtet mit einer Spalte des Referenz-DFs,
    z. B. weight="Act_raw_MW".

    Parameters
    ----------
    df : DataFrame
    start, end : str
    deltas : iterable of int/float
    weight : Optional[str]
        Spaltenname im Referenz-Details-DF (NO). Falls None oder Spalte fehlt → ungewichtet.
    """
    # Referenzstrategie (NO) mit ID
    det_ref, _ = reduced_strategies(
        df=df, regime="NO",
        start_date=start, end_date=end,
        forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
    )

    weight_col = weight if (weight and weight in det_ref.columns) else None

    rows = []

    # FIT: p_fit sweep
    base_fit = float(P_FIT)
    for d in deltas:
        p = base_fit + d
        det_fit, _ = reduced_strategies(
            df=df, regime="FIT", p_fit=p,
            start_date=start, end_date=end
        )
        idx = market_conformity_index(det_ref, det_fit, weight_col=weight_col)
        rows.append({"Regime":"FIT", "Delta": d, "BaseValue": base_fit, "TestValue": p, "MarketSimilarityIndex": idx})

    # CFD: K sweep
    base_k = float(CFD_K)
    for d in deltas:
        k = base_k + d
        det_cfd, _ = reduced_strategies(
            df=df, regime="CFD", cfd_strike=k,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        idx = market_conformity_index(det_ref, det_cfd, weight_col=weight_col)
        rows.append({"Regime":"CFD", "Delta": d, "BaseValue": base_k, "TestValue": k, "MarketSimilarityIndex": idx})

    # MPM: A sweep
    base_aw = float(MPM_AW)
    for d in deltas:
        a = base_aw + d
        det_mpm, _ = reduced_strategies(
            df=df, regime="MPM", mpm_aw=a,
            market_value_col=MV_COL, market_value_est_col=None,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        idx = market_conformity_index(det_ref, det_mpm, weight_col=weight_col)
        rows.append({"Regime":"MPM", "Delta": d, "BaseValue": base_aw, "TestValue": a, "MarketSimilarityIndex": idx})

    # MPM_EX: A sweep
    for d in deltas:
        a = base_aw + d
        det_mpmx, _ = reduced_strategies(
            df=df, regime="MPM_EX", mpm_aw=a,
            market_value_col=MV_COL, market_value_est_col=None,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        idx = market_conformity_index(det_ref, det_mpmx, weight_col=weight_col)
        rows.append({"Regime":"MPM_EX", "Delta": d, "BaseValue": base_aw, "TestValue": a, "MarketSimilarityIndex": idx})

    return (pd.DataFrame(rows)
              .sort_values(["Regime","Delta"])
              .reset_index(drop=True))


# ---------------------------------------------------------------------
# 2) Sensitivität – finanzielle Kennzahlen (Netto, Förderung, …)
# ---------------------------------------------------------------------
def _row_from_totals(regime: str, delta: float, base: float, test: float, totals: dict) -> dict:
    support = float(totals.get("FIT_€", 0.0)) + float(totals.get("MPM_€", 0.0)) + float(totals.get("CfD_€", 0.0))
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
        "Förderung_€": support,
        "Energie_Act_MWh": float(totals.get("Energie_Act_MWh", 0.0)),
        "Energie_Sched_MWh": float(totals.get("Energie_Sched_MWh", 0.0)),
    }

def financial_sensitivity(
    df: pd.DataFrame,
    start: str = START_DATE,
    end: str = END_DATE,
    deltas: Iterable[int] = (-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70),
) -> pd.DataFrame:
    rows = []

    # FIT
    base_fit = float(P_FIT)
    for d in deltas:
        p = base_fit + d
        _, totals = reduced_strategies(
            df=df, regime="FIT", p_fit=p,
            start_date=start, end_date=end
        )
        rows.append(_row_from_totals("FIT", d, base_fit, p, totals))

    # CFD
    base_k = float(CFD_K)
    for d in deltas:
        k = base_k + d
        _, totals = reduced_strategies(
            df=df, regime="CFD", cfd_strike=k,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        rows.append(_row_from_totals("CFD", d, base_k, k, totals))

    # MPM
    base_aw = float(MPM_AW)
    for d in deltas:
        a = base_aw + d
        _, totals = reduced_strategies(
            df=df, regime="MPM", mpm_aw=a,
            market_value_col=MV_COL, market_value_est_col=None,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        rows.append(_row_from_totals("MPM", d, base_aw, a, totals))

    # MPM_EX
    for d in deltas:
        a = base_aw + d
        _, totals = reduced_strategies(
            df=df, regime="MPM_EX", mpm_aw=a,
            market_value_col=MV_COL, market_value_est_col=None,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        rows.append(_row_from_totals("MPM_EX", d, base_aw, a, totals))

    return (pd.DataFrame(rows)
              .sort_values(["Regime","Delta"])
              .reset_index(drop=True))


# ---------------------------------------------------------------------
# 3) Plot-Helfer (Conformity + Finanzen)
# ---------------------------------------------------------------------
def plot_conformity_lines(sens: pd.DataFrame, title: str = "Market Conformity Sensitivity"):
    plt.figure(figsize=(10, 5))
    for reg, grp in sens.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp["MarketSimilarityIndex"], marker="o", label=reg)
    plt.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline")
    plt.ylabel("Market Conformity Index (0…1)")
    plt.title(title)
    plt.grid(alpha=0.35)
    plt.ylim(0.5, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_conformity_heatmap(sens: pd.DataFrame, title: str = "Market Conformity Heatmap"):
    pivot = sens.pivot(index="Delta", columns="Regime", values="MarketSimilarityIndex")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Conformity Index'})
    plt.title(title)
    plt.xlabel("Regime"); plt.ylabel("Δ (€/MWh)")
    plt.tight_layout(); plt.show()

def plot_fin_sensitivity(fin_df: pd.DataFrame, value_col="Netto_€", title=None, ylim=None):
    plt.figure(figsize=(10,5))
    for reg, grp in fin_df.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp[value_col], marker="o", label=reg)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline")
    plt.ylabel(value_col.replace("_"," "))
    if title: plt.title(title)
    if ylim: plt.ylim(*ylim)
    plt.grid(alpha=0.35); plt.legend(); plt.tight_layout(); plt.show()
=======
# =============================================================================
# Modul: metrics – Indizes & Sensitivitäten auf Basis der Simulationsdetails
# -----------------------------------------------------------------------------
# Zweck
# -----
# Dieses Modul baut auf den Ergebnissen von `reduced_strategies(...)` auf und
# liefert:
#   1) Market Conformity Index (0..1) zwischen zwei Strategien
#   2) Sensitivitätsanalysen über Parameter-Δ (FIT: p_fit; CFD: K; MPM/MPM_EX: A)
#      - a) Conformity-Sensitivität: Wie ähnlich sind die QH-Entscheidungen?
#      - b) Finanzielle Sensitivität: Wie verändern sich Netto/Komponenten?
#   3) Plot-Helfer für beide Sensitivitäten
#
# Einordnung im Projekt
# ---------------------
# - `reduced_strategies.py` enthält die **Kernlogik** (eine Strategie → details+totals).
# - `reduced_strategies_call.py` ist syntaktischer Zucker, um viele Läufe
#   bequem zu orchestrieren (z. B. alle Regime, DA_only/DA_ID).
# - **Dieses Modul** rechnet gezielt “Serien” von Läufen zum Vergleichen:
#   Referenzlauf(e) vs. Parameter-Sweeps; baut daraus Tabellen & Plots.
#
# Daten & Abhängigkeiten
# ----------------------
# - Eingabe ist ein DataFrame `df` mit DatetimeIndex (kommt typischerweise aus
#   `src.data_store.get_data()` → CSV-Cache).
# - Für MPM/MPM_EX erwartet `reduced_strategies(...)` realisierte Monatsmarktwerte
#   (config: `MV_REAL_COL`) und nutzt als Estimator **Vormonat** (oder eine explizite
#   Estimator-Spalte, wenn angegeben).
# - Conformity-Referenz ist standardmäßig **NO mit ID** (Intraday aktiv), d. h.
#   wir übergeben `forecast_id_col=config.FORECAST_ID_COL` und
#   `id_price_col=config.ID_PRICE_COL`.
#
# Kompatibilität zur Core-Logik
# -----------------------------
# - Parameter heißen identisch wie in `reduced_strategies(...)` (`p_fit`, `cfd_strike`,
#   `mpm_aw`, `market_value_col`, `market_value_est_col`, `forecast_id_col`, `id_price_col`).
# - ID ist nur aktiv, wenn **beide** Spalten gesetzt sind (wie im Core).
# - MPM/MPM_EX: Auszahlung mit realem Monats-MV; Entscheidung mit Estimator (hier: Vormonat),
#   exakt wie im Core beschrieben.
#
# Outputs
# -------
# - `conformity_sensitivity(...)` → DataFrame je Regime/Δ mit `MarketSimilarityIndex`
# - `financial_sensitivity(...)` → DataFrame je Regime/Δ mit Netto & Komponenten
# - Plotfunktionen (Lines/Heatmap für Conformity, Lines für Finanzen)
#
# Minimalbeispiel
# ---------------
# from src.data_store import get_data
# from src.metrics import conformity_sensitivity, financial_sensitivity, \
#     plot_conformity_lines, plot_conformity_heatmap, plot_fin_sensitivity
# import src.config as cfg
#
# cfg.START_DATE, cfg.END_DATE = "2024-01-01 00:00:00", "2024-12-31 23:59:59"
# df = get_data()
#
# sens = conformity_sensitivity(df, start=cfg.START_DATE, end=cfg.END_DATE,
#                               deltas=range(-50, 51, 10), use_abs_id_price_weight=True)
# plot_conformity_lines(sens, "Conformity 2024")
# plot_conformity_heatmap(sens, "Conformity 2024 – Heatmap")
#
# fin = financial_sensitivity(df, start=cfg.START_DATE, end=cfg.END_DATE,
#                             deltas=range(-50, 51, 10))
# plot_fin_sensitivity(fin, value_col="Netto_€", title="Netto – Sensitivität 2024")
# plot_fin_sensitivity(fin, value_col="Förderung_€", title="Förderung – Sensitivität 2024")
#
# Typische Stolpersteine
# ----------------------
# - Fehlende ID-Spalten → Conformity-Referenz NO MUSS mit ID laufen (sonst wenig Aussage).
# - Matplotlib/Seaborn müssen installiert sein (requirements).
# - Zeiträume: START/END konsistent halten, sonst vergleicht man unterschiedliche Fenster.
# - Gewichtung beim Conformity: Standard ist ungewicktet; optional
#   `Act_raw_MW * |p_ID_€/MWh|` (robuster bei negativen Preisen).
# =============================================================================


# src/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Iterable, Optional, Tuple

from .reduced_strategies import reduced_strategies
from .config import (
    FORECAST_ID_COL,
    ID_PRICE_COL,
    MV_REAL_COL as MV_COL,   # realisierte Monats-Marktwerte
    START_DATE,
    END_DATE,
    P_FIT,
    CFD_K,
    MPM_AW,
)

# ---------------------------------------------------------------------
# 0) Market Conformity Index (kompatibel zu deiner Logik)
# ---------------------------------------------------------------------
def market_conformity_index(
    ref: pd.DataFrame,
    other: pd.DataFrame,
    weight_col: str | None = None
) -> float:
    if not isinstance(ref.index, pd.DatetimeIndex) or not isinstance(other.index, pd.DatetimeIndex):
        raise ValueError("ref/other brauchen DatetimeIndex.")
    if "decision_qh" not in ref.columns or "decision_qh" not in other.columns:
        raise ValueError("Beide DFs brauchen 'decision_qh'.")

    idx = ref.index.intersection(other.index)
    if len(idx) == 0:
        raise ValueError("ref und other haben keinen gemeinsamen Zeitbereich.")

    a = ref.loc[idx, "decision_qh"].astype(bool).to_numpy()
    b = other.loc[idx, "decision_qh"].astype(bool).to_numpy()
    mismatch = np.logical_xor(a, b).astype(float)

    if weight_col is None:
        return float(1.0 - mismatch.mean())

    if weight_col not in ref.columns:
        # Fallback: ungewicktet, statt Exception
        return float(1.0 - mismatch.mean())

    w = (ref.loc[idx, weight_col]
           .astype(float)
           .replace([np.inf, -np.inf], np.nan)
           .fillna(0.0)
           .to_numpy())
    w_sum = w.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        return float(1.0 - mismatch.mean())
    return float(1.0 - float(np.dot(mismatch, w)) / float(w_sum))


# ---------------------------------------------------------------------
# 1) Sensitivität – Market Conformity (Δ wird wirklich angewandt)
#    Referenz = NO (mit ID)
# ---------------------------------------------------------------------
def conformity_sensitivity(
    df: pd.DataFrame,
    start: str = START_DATE,
    end: str = END_DATE,
    deltas: Iterable[int] = (-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70),
    weight: str | None = None,
) -> pd.DataFrame:
    """
    Berechnet den Market Conformity Index (0..1) für FIT/CFD/MPM/MPM_EX über Δ.
    Referenz = NO (mit ID). Optional gewichtet mit einer Spalte des Referenz-DFs,
    z. B. weight="Act_raw_MW".

    Parameters
    ----------
    df : DataFrame
    start, end : str
    deltas : iterable of int/float
    weight : Optional[str]
        Spaltenname im Referenz-Details-DF (NO). Falls None oder Spalte fehlt → ungewichtet.
    """
    # Referenzstrategie (NO) mit ID
    det_ref, _ = reduced_strategies(
        df=df, regime="NO",
        start_date=start, end_date=end,
        forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
    )

    weight_col = weight if (weight and weight in det_ref.columns) else None

    rows = []

    # FIT: p_fit sweep
    base_fit = float(P_FIT)
    for d in deltas:
        p = base_fit + d
        det_fit, _ = reduced_strategies(
            df=df, regime="FIT", p_fit=p,
            start_date=start, end_date=end
        )
        idx = market_conformity_index(det_ref, det_fit, weight_col=weight_col)
        rows.append({"Regime":"FIT", "Delta": d, "BaseValue": base_fit, "TestValue": p, "MarketSimilarityIndex": idx})

    # CFD: K sweep
    base_k = float(CFD_K)
    for d in deltas:
        k = base_k + d
        det_cfd, _ = reduced_strategies(
            df=df, regime="CFD", cfd_strike=k,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        idx = market_conformity_index(det_ref, det_cfd, weight_col=weight_col)
        rows.append({"Regime":"CFD", "Delta": d, "BaseValue": base_k, "TestValue": k, "MarketSimilarityIndex": idx})

    # MPM: A sweep
    base_aw = float(MPM_AW)
    for d in deltas:
        a = base_aw + d
        det_mpm, _ = reduced_strategies(
            df=df, regime="MPM", mpm_aw=a,
            market_value_col=MV_COL, market_value_est_col=None,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        idx = market_conformity_index(det_ref, det_mpm, weight_col=weight_col)
        rows.append({"Regime":"MPM", "Delta": d, "BaseValue": base_aw, "TestValue": a, "MarketSimilarityIndex": idx})

    # MPM_EX: A sweep
    for d in deltas:
        a = base_aw + d
        det_mpmx, _ = reduced_strategies(
            df=df, regime="MPM_EX", mpm_aw=a,
            market_value_col=MV_COL, market_value_est_col=None,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        idx = market_conformity_index(det_ref, det_mpmx, weight_col=weight_col)
        rows.append({"Regime":"MPM_EX", "Delta": d, "BaseValue": base_aw, "TestValue": a, "MarketSimilarityIndex": idx})

    return (pd.DataFrame(rows)
              .sort_values(["Regime","Delta"])
              .reset_index(drop=True))


# ---------------------------------------------------------------------
# 2) Sensitivität – finanzielle Kennzahlen (Netto, Förderung, …)
# ---------------------------------------------------------------------
def _row_from_totals(regime: str, delta: float, base: float, test: float, totals: dict) -> dict:
    support = float(totals.get("FIT_€", 0.0)) + float(totals.get("MPM_€", 0.0)) + float(totals.get("CfD_€", 0.0))
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
        "Förderung_€": support,
        "Energie_Act_MWh": float(totals.get("Energie_Act_MWh", 0.0)),
        "Energie_Sched_MWh": float(totals.get("Energie_Sched_MWh", 0.0)),
    }

def financial_sensitivity(
    df: pd.DataFrame,
    start: str = START_DATE,
    end: str = END_DATE,
    deltas: Iterable[int] = (-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70),
) -> pd.DataFrame:
    rows = []

    # FIT
    base_fit = float(P_FIT)
    for d in deltas:
        p = base_fit + d
        _, totals = reduced_strategies(
            df=df, regime="FIT", p_fit=p,
            start_date=start, end_date=end
        )
        rows.append(_row_from_totals("FIT", d, base_fit, p, totals))

    # CFD
    base_k = float(CFD_K)
    for d in deltas:
        k = base_k + d
        _, totals = reduced_strategies(
            df=df, regime="CFD", cfd_strike=k,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        rows.append(_row_from_totals("CFD", d, base_k, k, totals))

    # MPM
    base_aw = float(MPM_AW)
    for d in deltas:
        a = base_aw + d
        _, totals = reduced_strategies(
            df=df, regime="MPM", mpm_aw=a,
            market_value_col=MV_COL, market_value_est_col=None,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        rows.append(_row_from_totals("MPM", d, base_aw, a, totals))

    # MPM_EX
    for d in deltas:
        a = base_aw + d
        _, totals = reduced_strategies(
            df=df, regime="MPM_EX", mpm_aw=a,
            market_value_col=MV_COL, market_value_est_col=None,
            start_date=start, end_date=end,
            forecast_id_col=FORECAST_ID_COL, id_price_col=ID_PRICE_COL
        )
        rows.append(_row_from_totals("MPM_EX", d, base_aw, a, totals))

    return (pd.DataFrame(rows)
              .sort_values(["Regime","Delta"])
              .reset_index(drop=True))


# ---------------------------------------------------------------------
# 3) Plot-Helfer (Conformity + Finanzen)
# ---------------------------------------------------------------------
def plot_conformity_lines(sens: pd.DataFrame, title: str = "Market Conformity Sensitivity"):
    plt.figure(figsize=(10, 5))
    for reg, grp in sens.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp["MarketSimilarityIndex"], marker="o", label=reg)
    plt.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline")
    plt.ylabel("Market Conformity Index (0…1)")
    plt.title(title)
    plt.grid(alpha=0.35)
    plt.ylim(0.5, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_conformity_heatmap(sens: pd.DataFrame, title: str = "Market Conformity Heatmap"):
    pivot = sens.pivot(index="Delta", columns="Regime", values="MarketSimilarityIndex")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Conformity Index'})
    plt.title(title)
    plt.xlabel("Regime"); plt.ylabel("Δ (€/MWh)")
    plt.tight_layout(); plt.show()

def plot_fin_sensitivity(fin_df: pd.DataFrame, value_col="Netto_€", title=None, ylim=None):
    plt.figure(figsize=(10,5))
    for reg, grp in fin_df.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp[value_col], marker="o", label=reg)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline")
    plt.ylabel(value_col.replace("_"," "))
    if title: plt.title(title)
    if ylim: plt.ylim(*ylim)
    plt.grid(alpha=0.35); plt.legend(); plt.tight_layout(); plt.show()
>>>>>>> 6bc1d56 (general update)
