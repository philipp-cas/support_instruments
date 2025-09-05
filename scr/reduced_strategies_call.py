# src/reduced_strategies_call.py
# =============================================================================
# Modul: reduced_strategies_call – Orchestrierung von Simulationsläufen
# -----------------------------------------------------------------------------
# Zweck
# -----
# Dieses Modul stellt bequeme Wrapper um `reduced_strategies(...)` bereit:
# - run_one(regime, use_da_id, data=None): genau EIN Lauf pro Regime/Strategie.
# - run_all(regimes, data=None): alle Regime, jeweils DA_only & DA_ID.
# - run_all_da_only(regimes, data=None): nur DA_only.
# - run_all_da_id(regimes, data=None): nur DA+ID (Intraday aktiv).
# - make_details_by_regime(...): zerlegt MultiIndex-Details in Dict je Regime.
# - save_results_csv(...) / load_results_csv(...): Ergebnisse als CSV (kein Parquet).
#
# Integration in dein Projekt
# ---------------------------
# - Datenzugriff: via CSV-Cache aus base/data/data_final.csv (siehe src/data_store.py).
#   Du kannst `data` explizit reinreichen – oder None lassen (dann wird es intern
#   mit get_data() geladen).
# - Zentrale Konstanten kommen aus src/config.py.
#
# Minimalbeispiele
# ----------------
# from src.reduced_strategies_call import run_one, run_all_da_only, run_all_da_id
# from src.config import START_DATE, END_DATE
#
# # Einzel-Lauf: CFD, nur Day-Ahead (kein ID)
# details_cfd_da_only, totals_cfd_da_only = run_one("CFD", use_da_id=False)
#
# # Alle Regime – DA_only
# REGS = ["NO","QUANT","FIT","FIT_PREMIUM","MPM","CFD"]
# details_all_DA_only, totals_DA_only = run_all_da_only(REGS)
#
# # Alle Regime – DA_ID (Intraday aktiv)
# details_all_DA_ID, totals_DA_ID = run_all_da_id(REGS)
#
# # Ergebnisse als CSV sichern (in ./results)
# from pathlib import Path
# save_results_csv(
#     outdir=Path("results"),
#     details_all_DA_only=details_all_DA_only,
#     details_all_DA_ID=details_all_DA_ID,
#     totals_DA_only=totals_DA_only,
#     totals_DA_ID=totals_DA_ID
# )
# =============================================================================

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Iterable, Tuple, Dict

from .reduced_strategies import reduced_strategies
from .data_store import get_data
from .config import (
    # Zeitraum
    START_DATE, END_DATE,
    # Spalten
    FORECAST_DA_COL, ACTUAL_COL, DA_PRICE_COL, REBAP_COL,
    FORECAST_ID_COL, ID_PRICE_COL, MV_REAL_COL,
    # Parameter
    P_FIT, CFD_K, CFD_VIEW, MPM_AW,
)

# -----------------------------------------------------------------------------
# interne Helfer
# -----------------------------------------------------------------------------
def _id_kwargs(use_da_id: bool) -> dict:
    """
    Gibt die kwargs zurück, um Intraday zu aktivieren/deaktivieren.
    - True  => beide ID-Spalten setzen
    - False => gar nichts setzen (DA-only)
    """
    return {"forecast_id_col": FORECAST_ID_COL, "id_price_col": ID_PRICE_COL} if use_da_id else {}


def _regime_extras(regime: str) -> dict:
    """
    Regime-spezifische Zusatzargumente (ohne Logikänderung).
    """
    r = regime.upper()
    if r in {"FIT", "FIT_PREMIUM"}:
        return {"p_fit": P_FIT}
    if r == "MPM":
        # Erwartet einen Estimator: "__MV_EST__" (falls vorhanden). Wenn du keinen Estimator
        # als Spalte lieferst, gib später market_value_est_col=None an, dann nutzt
        # reduced_strategies den Vormonat automatisch.
        return {"mpm_aw": MPM_AW, "market_value_col": MV_REAL_COL, "market_value_est_col": "__MV_EST__"}
    if r == "CFD":
        return {
            "cfd_strike": CFD_K,
            "da_present_as_diff": (CFD_VIEW == "transparent"),
            "cfd_collapse_to_strike": False,
        }
    return {}


# -----------------------------------------------------------------------------
# 1) EIN Lauf: genau ein Regime + (DA_only | DA_ID)
# -----------------------------------------------------------------------------
def run_one(regime: str, use_da_id: bool, data: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, dict]:
    """
    Führt genau einen reduced_strategies()-Lauf aus.

    Parameters
    ----------
    regime : {"NO","QUANT","FIT","FIT_PREMIUM","MPM","CFD"}
    use_da_id : bool
        False => DA_only (kein Intraday); True => DA+ID (Intraday aktiv).
    data : pd.DataFrame | None
        Optional: bereits geladener DataFrame. Wenn None, wird er via get_data() geladen.

    Returns
    -------
    details : pd.DataFrame   (QH-Zeitreihe mit Zahlungen/Flags)
    totals  : dict           (aggregierte Kennzahlen)
    """
    if data is None:
        data = get_data()  # CSV-Cache aus base/data/data_final.csv

    return reduced_strategies(
        df=data,
        regime=regime.upper(),
        start_date=START_DATE,
        end_date=END_DATE,
        forecast_da_col=FORECAST_DA_COL,
        actual_col=ACTUAL_COL,
        da_price_col=DA_PRICE_COL,
        rebap_col=REBAP_COL,
        **_id_kwargs(use_da_id),
        **_regime_extras(regime),
    )


# -----------------------------------------------------------------------------
# 2) ALLE Regime: DA_only & DA_ID
# -----------------------------------------------------------------------------
def run_all(regimes: Iterable[str], data: pd.DataFrame | None = None
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Führt für alle übergebenen Regime jeweils (a) DA_only und (b) DA+ID aus.

    Returns
    -------
    details_all_DA_only : pd.DataFrame   (MultiIndex: level "regime")
    details_all_DA_ID   : pd.DataFrame   (MultiIndex: level "regime")
    totals_DA_only      : pd.DataFrame   (Index: "regime")
    totals_DA_ID        : pd.DataFrame   (Index: "regime")
    """
    if data is None:
        data = get_data()

    # --- DA_only ---
    details_DA_only: Dict[str, pd.DataFrame] = {}
    totals_rows_only = []
    for r in regimes:
        det, tot = run_one(r, use_da_id=False, data=data)
        details_DA_only[r] = det
        totals_rows_only.append({"regime": r, **tot})
    details_all_DA_only = pd.concat(details_DA_only, names=["regime"])
    totals_DA_only = pd.DataFrame(totals_rows_only).set_index("regime").sort_index()

    # --- DA_ID ---
    details_DA_ID: Dict[str, pd.DataFrame] = {}
    totals_rows_id = []
    for r in regimes:
        det, tot = run_one(r, use_da_id=True, data=data)
        details_DA_ID[r] = det
        totals_rows_id.append({"regime": r, **tot})
    details_all_DA_ID = pd.concat(details_DA_ID, names=["regime"])
    totals_DA_ID = pd.DataFrame(totals_rows_id).set_index("regime").sort_index()

    return details_all_DA_only, details_all_DA_ID, totals_DA_only, totals_DA_ID


# -----------------------------------------------------------------------------
# 3) Convenience: nur DA_only
# -----------------------------------------------------------------------------
def run_all_da_only(regimes: Iterable[str], data: pd.DataFrame | None = None
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Führt alle Regime ausschließlich als DA_only aus und bündelt Details/Totals."""
    if data is None:
        data = get_data()
    details: Dict[str, pd.DataFrame] = {}
    rows = []
    for r in regimes:
        det, tot = run_one(r, use_da_id=False, data=data)
        details[r] = det
        rows.append({"regime": r, **tot})
    details_all = pd.concat(details, names=["regime"])
    totals = pd.DataFrame(rows).set_index("regime").sort_index()
    return details_all, totals


# -----------------------------------------------------------------------------
# 4) Convenience: nur DA_ID
# -----------------------------------------------------------------------------
def run_all_da_id(regimes: Iterable[str], data: pd.DataFrame | None = None
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Führt alle Regime als DA+ID (Intraday aktiv) aus und bündelt Details/Totals."""
    if data is None:
        data = get_data()
    details: Dict[str, pd.DataFrame] = {}
    rows = []
    for r in regimes:
        det, tot = run_one(r, use_da_id=True, data=data)
        details[r] = det
        rows.append({"regime": r, **tot})
    details_all = pd.concat(details, names=["regime"])
    totals = pd.DataFrame(rows).set_index("regime").sort_index()
    return details_all, totals


# -----------------------------------------------------------------------------
# 5) Plot-Adapter (praktisch für Regime-weise Plots)
# -----------------------------------------------------------------------------
def make_details_by_regime(details_all_strategy: pd.DataFrame, start=None, end=None) -> Dict[str, pd.DataFrame]:
    """
    Zerlegt einen MultiIndex-Details-DataFrame (level 'regime') in ein Dict je Regime.
    Optionaler Zeitfilter [start, end].
    """
    regs = details_all_strategy.index.get_level_values("regime").unique()
    out = {reg: details_all_strategy.xs(reg, level="regime") for reg in regs}
    if start or end:
        s = pd.to_datetime(start) if start else None
        e = pd.to_datetime(end) if end else None
        for reg in out:
            df = out[reg]
            out[reg] = df.loc[
                (df.index >= (s or df.index.min())) &
                (df.index <= (e or df.index.max()))
            ]
    return out


# -----------------------------------------------------------------------------
# 6) Ergebnisse speichern/laden (CSV only)
# -----------------------------------------------------------------------------
def _ensure_index_names(details: pd.DataFrame) -> pd.DataFrame:
    """
    Sorgt dafür, dass der MultiIndex für CSV rund ist:
    - level 0: 'regime'
    - level 1: 'DateTime' (DatetimeIndex)
    """
    if details.index.nlevels == 2:
        names = list(details.index.names)
        if not names or names[0] != "regime":
            names[0] = "regime"
        if len(names) < 2 or not names[1]:
            names = ["regime", "DateTime"]
        details = details.copy()
        details.index.set_names(names, inplace=True)
    return details


def save_results_csv(
    outdir: Path | str,
    details_all_DA_only: pd.DataFrame,
    details_all_DA_ID: pd.DataFrame,
    totals_DA_only: pd.DataFrame,
    totals_DA_ID: pd.DataFrame,
):
    """
    Speichert alle Ergebnisse als CSV (ohne Parquet).
    - Details haben MultiIndex (regime, DateTime) → Index wird mitspeichert.
    - Totals sind einfache Tabellen (Index=regime).
    """
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)

    d1 = _ensure_index_names(details_all_DA_only)
    d2 = _ensure_index_names(details_all_DA_ID)

    # Details als CSV (Index mit speichern)
    d1.to_csv(p / "details_DA_only.csv", index=True)
    d2.to_csv(p / "details_DA_ID.csv", index=True)

    # Totals als CSV
    totals_DA_only.to_csv(p / "totals_DA_only.csv")
    totals_DA_ID.to_csv(p / "totals_DA_ID.csv")


def load_results_csv(outdir: Path | str
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lädt die gespeicherten CSVs wieder ein.
    Achtung: Details haben MultiIndex (regime, DateTime); DateTime wird geparst.
    """
    p = Path(outdir)

    # Details: MultiIndex rekonstruieren
    d1 = pd.read_csv(p / "details_DA_only.csv", parse_dates=["DateTime"])
    d2 = pd.read_csv(p / "details_DA_ID.csv",   parse_dates=["DateTime"])

    d1 = d1.set_index(["regime", "DateTime"]).sort_index()
    d2 = d2.set_index(["regime", "DateTime"]).sort_index()

    # Totals
    t1 = pd.read_csv(p / "totals_DA_only.csv", index_col=0)
    t2 = pd.read_csv(p / "totals_DA_ID.csv",   index_col=0)

    return d1, d2, t1, t2
