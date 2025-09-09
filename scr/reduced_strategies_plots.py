# src/reduced_strategies_plots.py
"""
Plot-Utilities für Ergebnisse aus reduced_strategies.

Kompatible Eingaben:
- dict[str, pd.DataFrame]  (z.B. {"NO": df_no, "CFD": df_cfd, ...})
- MultiIndex-DataFrame mit Index-Level "regime" (z.B. details_all_DA_ID)
- Einzelner DataFrame (eine Strategie)

Übliche Spalten in den Details-DataFrames:
- Zahlungen: "Revenue_DA_€", "Revenue_ID_€", "reBAP_€", "FIT_€", "MPM_€", "CfD_€", "Netto_€"
- Energie/Planung: "Sched_MW", "Act_MW"
- Flags: "buyback", "late_production_qh", "late_production_hour", "decision_qh"

Hinweise:
- Bei Aggregation zu €/Monat, €/Jahr werden Zahlungen summiert.
- Energie in MWh = MW * Δ mit Δ=0.25 (Viertelstunden).
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from typing import Dict

__all__ = [
    "plot_strategies_breakdown_ID",
    "plot_regime_totals_comparison",
    "plot_production_energy_bars_ID",
    "plot_production_energy_timeseries_ID",
    "plot_production_energy_comparison_ID",
    "plot_strategies_annual_comparison_ID",
    "plot_strategy_timeseries",
    "plot_strategy_and_prices_timeseries",
    "plot_decision_share_ID",
    "plot_rebap_diagnostics_ID",
    "plot_support_by_regime",
    "summarize_details",
]

# ===== Farben & Reihenfolge =====
COMP_COLORS = {
    "Revenue_DA_€": "#1f77b4",  # blau
    "Revenue_ID_€": "#17becf",  # cyan
    "reBAP_€":      "#d62728",  # rot
    "CfD_€":        "#2ca02c",  # grün
    "FIT_€":        "#ff7f0e",  # orange
    "MPM_€":        "#9467bd",  # lila
}
COMP_ORDER = ["Revenue_DA_€","Revenue_ID_€","reBAP_€","CfD_€","FIT_€","MPM_€"]

SUPPORT_COLS = ["FIT_€", "MPM_€", "CfD_€"]
SUPPORT_COLORS = {
    "FIT_€": COMP_COLORS.get("FIT_€", "#ff7f0e"),
    "MPM_€": COMP_COLORS.get("MPM_€", "#9467bd"),
    "CfD_€": COMP_COLORS.get("CfD_€", "#2ca02c"),
}

Δ = 0.25  # Stunden pro Viertelstunde (MWh = MW * Δ)


# ===== Helpers ================================================================

def _as_regime_dict(obj, label_for_single="ALL") -> Dict[str, pd.DataFrame]:
    """
    Konvertiert Eingabe in dict[str, DataFrame]:
    - dict -> unverändert
    - MultiIndex-DataFrame mit Level 'regime' -> split per xs
    - einfacher DataFrame -> {label_for_single: df}
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.index, pd.MultiIndex) and ("regime" in obj.index.names):
            regs = obj.index.get_level_values("regime").unique()
            return {str(reg): obj.xs(reg, level="regime") for reg in regs}
        return {label_for_single: obj}
    raise TypeError("Erwarte dict[str, DataFrame] oder DataFrame (ggf. mit MultiIndex-Level 'regime').")

def _ensure_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in COMP_ORDER:
        if c not in out.columns:
            out[c] = 0.0
    if "Netto_€" not in out.columns:
        out["Netto_€"] = out[COMP_ORDER].sum(axis=1)
    return out

def _idx_width_loc_fmt(idx: pd.DatetimeIndex, f: str):
    fu = (f or "").upper()
    if fu in {"MS","M"}:
        idx = idx.to_period("M").to_timestamp(how="start")
        width = 20
        loc = mdates.MonthLocator()
        fmt = mdates.DateFormatter("%Y-%m")
    elif fu in {"YE","YS"}:
        idx = idx.to_period("Y").to_timestamp(how="start")
        width = 200
        loc = mdates.YearLocator()
        fmt = mdates.DateFormatter("%Y")
    elif fu in {"QE","QS"}:
        idx = idx.to_period("Q").to_timestamp(how="start")
        width = 60
        loc = mdates.MonthLocator(interval=3)
        fmt = mdates.DateFormatter("%Y-%m")
    elif fu == "W":
        width = 5
        loc = mdates.WeekdayLocator(byweekday=mdates.MO)
        fmt = mdates.DateFormatter("%Y-%m-%d")
    else:
        width = 10
        loc = mdates.AutoDateLocator()
        fmt = mdates.DateFormatter("%Y-%m-%d")
    return idx, width, loc, fmt


# ===== 1) Variable Auflösung: Erlös-Breakdown je Regime =======================

def plot_strategies_breakdown_ID(details_or_multi,
                                 freq: str = "MS",
                                 title: str = "Erlösstruktur je Regime",
                                 sharey: bool = True):
    """
    Gestapelte Erlöse (DA/ID/reBAP/CfD/FIT/MPM) pro Regime, Netto-Linie in Schwarz.
    Akzeptiert dict oder MultiIndex-DataFrame.
    """
    details_by_regime = _as_regime_dict(details_or_multi)

    regs = list(details_by_regime.keys())
    n = len(regs)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.6*nrows), squeeze=False, sharey=sharey)
    axes = axes.flatten()

    for i, reg in enumerate(regs):
        df = _ensure_components(details_by_regime[reg])
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Details für '{reg}' brauchen DatetimeIndex.")
        agg = df.resample(freq)[COMP_ORDER + ["Netto_€"]].sum()

        agg.index, bar_width, locator, formatter = _idx_width_loc_fmt(agg.index, freq)
        x = agg.index; ax = axes[i]

        pos = agg[COMP_ORDER].clip(lower=0); neg = agg[COMP_ORDER].clip(upper=0)
        bottom_pos = np.zeros(len(agg)); bottom_neg = np.zeros(len(agg))

        for c in COMP_ORDER:
            vals = pos[c].to_numpy()
            if np.any(vals != 0):
                ax.bar(x, vals, width=bar_width, bottom=bottom_pos,
                       color=COMP_COLORS[c], edgecolor="none")
                bottom_pos += vals
        for c in COMP_ORDER:
            vals = neg[c].to_numpy()
            if np.any(vals != 0):
                ax.bar(x, vals, width=bar_width, bottom=bottom_neg,
                       color=COMP_COLORS[c], edgecolor="none", alpha=0.95)
                bottom_neg += vals

        ax.plot(x, agg["Netto_€"], color="black", lw=2.2)
        ax.set_title(f"{reg} — Komponenten & Netto"); ax.set_ylabel("€")
        ax.grid(axis="y", alpha=0.35)
        ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        if i == 0:
            handles = [mpatches.Patch(color=COMP_COLORS[c], label=c) for c in COMP_ORDER]
            handles.append(plt.Line2D([], [], color="black", lw=2.2, label="Netto"))
            fig.legend(handles, [h.get_label() for h in handles],
                       ncol=7, loc="upper center", frameon=True)

    for j in range(i+1, len(axes)): fig.delaxes(axes[j])
    fig.suptitle(title, fontsize=15, y=0.98); fig.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()


# ===== 2) Regime-Totals (gestapelt, z.B. Jahr) ================================

def plot_regime_totals_comparison(details_or_multi,
                                  freq: str = "YE",
                                  title: str = "Vergleich: gestapelte Erlöse je Regime"):
    """
    Eine Gruppe je Regime, gestapelte Komponenten (DA, ID, reBAP, CfD, FIT, MPM).
    freq="YE" → Jahressummen.
    """
    details_by_regime = _as_regime_dict(details_or_multi)

    stacks = []
    regimes = []
    for reg, df in details_by_regime.items():
        df2 = _ensure_components(df)
        agg = df2.resample(freq)[COMP_ORDER + ["Netto_€"]].sum()
        row = agg.iloc[-1] if len(agg) > 0 else pd.Series(0.0, index=COMP_ORDER + ["Netto_€"])
        stacks.append(row); regimes.append(reg)

    agg_totals = pd.DataFrame(stacks, index=regimes)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(regimes)); width = 0.6

    bottom = np.zeros(len(regimes))
    for c in COMP_ORDER:
        vals = agg_totals[c].to_numpy()
        ax.bar(x, vals, width=width, bottom=bottom, color=COMP_COLORS[c], edgecolor="none", label=c)
        bottom += vals

    ax.plot(x, agg_totals["Netto_€"].to_numpy(), "ko", ms=7, label="Netto")

    ax.set_xticks(x, regimes); ax.set_ylabel("€"); ax.set_title(title)
    ax.grid(axis="y", alpha=0.35)
    handles = [mpatches.Patch(color=COMP_COLORS[c], label=c) for c in COMP_ORDER]
    handles.append(plt.Line2D([], [], color="black", marker="o", lw=0, label="Netto"))
    ax.legend(handles=handles, ncol=4, loc="upper center")
    plt.tight_layout(); plt.show()


# ===== 3) Produktionsenergie (Balken) ========================================

def plot_production_energy_bars_ID(details_or_multi,
                                   freq: str = "MS",
                                   title: str = "Erzeugte/Geplante Energie (Balken)",
                                   show: str = "Act"):
    """
    show: "Act", "Sched" oder "Both"
    """
    details_by_regime = _as_regime_dict(details_or_multi)

    frames = []
    for reg, df in details_by_regime.items():
        if show.upper() in {"ACT","BOTH"}:
            energy_act = (df["Act_MW"] * Δ).resample(freq).sum().rename((reg, "Act"))
            frames.append(energy_act)
        if show.upper() in {"SCHED","BOTH"}:
            energy_sched = (df["Sched_MW"] * Δ).resample(freq).sum().rename((reg, "Sched"))
            frames.append(energy_sched)

    mat = pd.concat(frames, axis=1)
    if isinstance(mat.columns, pd.MultiIndex):
        mat.columns = [' - '.join(col) for col in mat.columns]

    ax = mat.plot(kind="bar", figsize=(max(11, 1.2*len(mat)), 5))
    ax.set_title(f"{title} — Aggregation: {freq}")
    ax.set_ylabel("Energie (MWh)")
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout(); plt.show()


# ===== 4) Produktionsenergie (Zeitreihe) =====================================

def plot_production_energy_timeseries_ID(details_or_multi,
                                         freq: str = "W",
                                         title: str = "Erzeugte Energie je Periode (Zeitreihe)",
                                         which: str = "Act"):
    """
    which: "Act" oder "Sched"
    """
    details_by_regime = _as_regime_dict(details_or_multi)

    plt.figure(figsize=(13, 5))
    for reg, df in details_by_regime.items():
        col = "Act_MW" if which.lower() == "act" else "Sched_MW"
        energy = (df[col] * Δ).resample(freq).sum()
        plt.plot(energy.index, energy.values, label=f"{reg} ({which})")
    plt.title(f"{title} — Aggregation: {freq}")
    plt.ylabel("Energie (MWh)")
    plt.grid(True, alpha=0.35)
    plt.legend(ncol=3)
    plt.tight_layout(); plt.show()


# ===== 5) Energievergleich (Jahreswerte) =====================================

def plot_production_energy_comparison_ID(details_or_multi,
                                         freq: str = "YE",
                                         title: str = "Jährliche Energie: Act vs. Sched je Regime"):
    details_by_regime = _as_regime_dict(details_or_multi)

    regs = list(details_by_regime.keys())
    sched_vals, act_vals = [], []
    for reg in regs:
        df = details_by_regime[reg]
        sched_vals.append((df["Sched_MW"] * Δ).resample(freq).sum().iloc[-1])
        act_vals.append((df["Act_MW"]   * Δ).resample(freq).sum().iloc[-1])

    x = np.arange(len(regs)); w = 0.35
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(x - w/2, sched_vals, width=w, label="Fahrplan (MWh)")
    ax.bar(x + w/2, act_vals,   width=w, label="Ist (MWh)")
    ax.set_xticks(x, regs); ax.set_ylabel("MWh"); ax.set_title(title)
    ax.grid(axis="y", alpha=0.35); ax.legend()
    plt.tight_layout(); plt.show()


# ===== 6) Jahres-Vergleich: gestapelte Erlöse ================================

def plot_strategies_annual_comparison_ID(details_or_multi,
                                         title: str = "Annual revenue comparison (DA+ID+reBAP+Support)"):
    details_by_regime = _as_regime_dict(details_or_multi)

    regs = list(details_by_regime.keys())
    annual = pd.DataFrame(index=regs, columns=COMP_ORDER, dtype=float).fillna(0.0)
    for reg, df in details_by_regime.items():
        x = _ensure_components(df)
        annual.loc[reg, COMP_ORDER] = x[COMP_ORDER].sum().values

    x = np.arange(len(regs)); width = 0.65
    fig, ax = plt.subplots(figsize=(1.6 + 1.4*len(regs), 6))
    pos_bottom = np.zeros(len(regs)); neg_bottom = np.zeros(len(regs))
    for c in COMP_ORDER:
        vals = annual[c].to_numpy()
        pos = np.where(vals > 0, vals, 0.0); neg = np.where(vals < 0, vals, 0.0)
        if np.any(pos != 0):
            ax.bar(x, pos, width=width, bottom=pos_bottom, color=COMP_COLORS[c], edgecolor="none", label=c)
            pos_bottom += pos
        if np.any(neg != 0):
            ax.bar(x, neg, width=width, bottom=neg_bottom, color=COMP_COLORS[c], edgecolor="none", alpha=0.95)
            neg_bottom += neg

    net = annual.sum(axis=1).to_numpy()
    ax.scatter(x, net, color="black", s=30, zorder=3, label="Netto")
    for xi, yi in zip(x, net): ax.plot([xi, xi], [0, yi], color="red", lw=1, alpha=0.6)

    ax.set_xticks(x); ax.set_xticklabels(regs)
    ax.set_ylabel("€ (annual)"); ax.set_title(title); ax.grid(axis="y", alpha=0.35)
    handles = [mpatches.Patch(color=COMP_COLORS[c], label=c) for c in COMP_ORDER]
    handles.append(plt.Line2D([], [], color="black", marker="o", linestyle="none", label="Netto"))
    ax.legend(handles=handles, loc="upper left", ncol=3, frameon=True)
    plt.tight_layout(); plt.show()


# ===== 7) Timeseries-Diagnose (Sched/Act/ID/Buybacks/Late) ====================

def plot_strategy_timeseries(details: pd.DataFrame,
                             title: str = "Fahrplan vs. Ist (Buybacks/ID)",
                             start=None, end=None, sample: str | None = None):
    """
    Plottet eine Strategie (ein DataFrame). Erwartet:
    - Sched_MW, Act_MW
    - ID_traded_MW (optional)
    - buyback (bool) – optional
    - late_production_qh (bool) – optional
    """
    x = details.loc[
        (details.index >= (pd.to_datetime(start) if start else details.index.min())) &
        (details.index <= (pd.to_datetime(end)   if end   else details.index.max()))
    ].copy()

    if sample:
        x = x.resample(sample).mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(x.index, x["Sched_MW"], label="Fahrplan (Sched_MW)")
    ax.plot(x.index, x["Act_MW"],   label="Ist (Act_MW)", alpha=0.9)

    if "buyback" in x.columns:
        bb = x["buyback"].astype(bool)
        if bb.any():
            ax.fill_between(x.index, 0, 1, where=bb, transform=ax.get_xaxis_transform(),
                            alpha=0.33, color="#d62728", label="Buyback QH")

    if "late_production_qh" in x.columns:
        lp = x["late_production_qh"].astype(bool)
        if lp.any():
            ax.fill_between(x.index, 0, 1, where=lp, transform=ax.get_xaxis_transform(),
                            alpha=0.33, color="#1f77b4", label="Late Production QH")

    if "ID_traded_MW" in x.columns:
        ax.plot(x.index, x["ID_traded_MW"], label="ID gehandelt (MW)", alpha=0.6)

    ax.set_title(title); ax.set_ylabel("MW"); ax.grid(True, alpha=0.35)
    ax.legend(ncol=4); plt.tight_layout(); plt.show()


# ==== 8) Timeseries-Diagnose inkl. Preisen

import pandas as pd
import matplotlib.pyplot as plt

def plot_strategy_and_prices_timeseries(
    details: pd.DataFrame,
    title: str = "Fahrplan / Ist + Preise (DA/ID)",
    start=None,
    end=None,
    sample: str | None = None,
    show_id_traded: bool = True,
    show_flags: bool = True,
):
    """
    Zeitreihen-Plot für eine Strategie (ein DataFrame), kombiniert mit Preisen.
    Linke y-Achse: Leistungen (MW) -> Sched_MW, Act_MW, optional ID_traded_MW
    Rechte y-Achse: Preise (€/MWh) -> p_DA_€/MWh, p_ID_€/MWh
    """

    # ---------------- Farben zentral definieren ----------------
    c_sched    = "blue"         # blau (Fahrplan)
    c_act      = "forestgreen"  # grün (Ist)
    c_idmw     = "darkorange"   # orange (ID gehandelt)

    c_buyback  = "aquamarine"     # Buyback Fläche
    c_late     = "lightsteelblue" # Late Production Fläche

    c_price_da = "red"           # DA-Preis
    c_price_id = "tomato"     # ID-Preis
    c_zero     = "black"         # 0€-Linie
    # -----------------------------------------------------------

    # --- Zeitfenster schneiden ---
    x = details.loc[
        (details.index >= (pd.to_datetime(start) if start else details.index.min())) &
        (details.index <= (pd.to_datetime(end)   if end   else details.index.max()))
    ].copy()

    # --- (optional) Resampling/Glättung ---
    if sample:
        # numerische Spalten mitteln
        num_cols = x.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0:
            x_num = x[num_cols].resample(sample).mean(numeric_only=True)
        else:
            x_num = pd.DataFrame(index=x.index)

        # bool-Flags als "irgendein QH wahr?" aggregieren
        x_bool = {}
        for bcol in ("buyback", "late_production_qh"):
            if bcol in x.columns:
                x_bool[bcol] = x[bcol].astype(bool).resample(sample).max()

        # zusammenbauen
        x = x_num
        for bcol, s in x_bool.items():
            x[bcol] = s

    # --- Figure & Axes ---
    fig, ax_mw = plt.subplots(figsize=(13, 4.8))
    ax_eur = ax_mw.twinx()

    # --- Linke Achse (MW) ---
    if "Sched_MW" in x.columns:
        ax_mw.plot(x.index, x["Sched_MW"], label="Fahrplan (Sched_MW)", lw=1.8, color=c_sched)
    if "Act_MW" in x.columns:
        ax_mw.plot(x.index, x["Act_MW"],   label="Ist (Act_MW)",       lw=1.8, color=c_act, alpha=0.95)
    if show_id_traded and "ID_traded_MW" in x.columns:
        ax_mw.plot(x.index, x["ID_traded_MW"], label="ID gehandelt (MW)", lw=1.4, color=c_idmw, alpha=0.85)

    ax_mw.set_ylabel("MW")
    ax_mw.grid(True, which="major", axis="both", alpha=0.35)

    # --- Rechte Achse (€/MWh) ---
    price_handles = []
    price_labels  = []

    if "p_DA_€/MWh" in x.columns:
        h_da, = ax_eur.plot(x.index, x["p_DA_€/MWh"], lw=1.6, color=c_price_da, label="DA-Preis (€/MWh)")
        price_handles.append(h_da); price_labels.append("DA-Preis (€/MWh)")

    if "p_ID_€/MWh" in x.columns:
        h_id, = ax_eur.plot(x.index, x["p_ID_€/MWh"], lw=1.6, color=c_price_id, label="ID-Preis (€/MWh)")
        price_handles.append(h_id); price_labels.append("ID-Preis (€/MWh)")

    ax_eur.set_ylabel("Preis (€/MWh)")
    # 0€-Linie
    ax_eur.axhline(0.0, color=c_zero, lw=1.0, alpha=0.8)

    # --- Flächenmarkierungen ---
    if show_flags:
        if "buyback" in x.columns:
            bb = x["buyback"].astype(bool)
            if bb.any():
                ax_mw.fill_between(
                    x.index, 0, 1, where=bb,
                    transform=ax_mw.get_xaxis_transform(),
                    alpha=0.28, color=c_buyback, label="Buyback QH"
                )
        if "late_production_qh" in x.columns:
            lp = x["late_production_qh"].astype(bool)
            if lp.any():
                ax_mw.fill_between(
                    x.index, 0, 1, where=lp,
                    transform=ax_mw.get_xaxis_transform(),
                    alpha=0.28, color=c_late, label="Late Production QH"
                )

    # --- Titel & Legende ---
    ax_mw.set_title(title)

    handles_mw, labels_mw = ax_mw.get_legend_handles_labels()
    handles_eur, labels_eur = ax_eur.get_legend_handles_labels()
    if handles_mw or handles_eur:
        ax_mw.legend(handles_mw + handles_eur, labels_mw + labels_eur,
                     ncol=4, loc="upper center", frameon=True)

    plt.tight_layout()
    plt.show()


# ===== 9) Produktionsentscheidung (Anteil ON) =================================

def plot_decision_share_ID(details_or_multi,
                           freq: str = "MS",
                           title: str = "Anteil Viertelstunden mit Produktionsentscheidung = ON"):
    details_by_regime = _as_regime_dict(details_or_multi)

    fig, ax = plt.subplots(figsize=(12, 4))
    for reg, df in details_by_regime.items():
        col = "decision_qh" if "decision_qh" in df.columns else None
        if col is None:
            continue
        share = df[col].astype(bool).resample(freq).mean().mul(100)
        ax.plot(share.index, share.values, label=reg)
    ax.set_title(title); ax.set_ylabel("% ON"); ax.grid(True, alpha=0.35)
    ax.legend(ncol=4); plt.tight_layout(); plt.show()


# ===== 10) reBAP-Diagnostik ====================================================

def plot_rebap_diagnostics_ID(details: pd.DataFrame, freq: str = "MS",
                              title_hist: str = "Verteilung reBAP € pro QH",
                              title_month: str = "Monatliche reBAP-Summe (Kosten/Gutschrift)"):
    x = details.copy()
    fig, ax = plt.subplots(figsize=(10,4))
    x["reBAP_€"].hist(bins=60, ax=ax)
    ax.set_title(title_hist); ax.set_xlabel("€ pro QH"); ax.set_ylabel("Häufigkeit")
    plt.tight_layout(); plt.show()

    agg = x["reBAP_€"].resample(freq).sum()
    idx, width, loc, fmt = _idx_width_loc_fmt(agg.index, freq)
    pos = agg.clip(lower=0); neg = agg.clip(upper=0)
    fig, ax = plt.subplots(figsize=(12,4))
    ax.bar(idx, pos, width=width, color=COMP_COLORS["reBAP_€"], alpha=0.9, label="reBAP (Gutschrift)")
    ax.bar(idx, neg, width=width, color=COMP_COLORS["reBAP_€"], alpha=0.6, label="reBAP (Kosten)")
    ax.set_title(title_month); ax.grid(axis="y", alpha=0.35)
    ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(fmt)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(); plt.tight_layout(); plt.show()


# ===== 11) Buyback-Anteil =====================================================

def plot_buyback_share(details_or_multi,
                       freq: str = "MS",
                       title: str = "Buyback-Häufigkeit je Regime"):
    details_by_regime = _as_regime_dict(details_or_multi)

    fig, ax = plt.subplots(figsize=(12,4))
    for reg, df in details_by_regime.items():
        if "buyback" not in df.columns:
            continue
        share = df["buyback"].astype(bool).resample(freq).mean().mul(100)
        ax.plot(share.index, share.values, label=f"{reg} (% QH Buyback)")
    ax.set_title(title); ax.set_ylabel("% QH"); ax.grid(True, alpha=0.35)
    ax.legend(ncol=4); plt.tight_layout(); plt.show()


# ===== 12) Förderhöhe je Regime ===============================================

def plot_support_by_regime(details_or_multi,
                           freq: str = "MS",
                           title: str = "Förderhöhe je Regime",
                           sharey: bool = True,
                           stacked: bool = True):
    """
    Zeigt FIT_€, MPM_€, CfD_€ je Regime mit gewünschter zeitlicher Auflösung.
    """
    details_by_regime = _as_regime_dict(details_or_multi)

    regs = list(details_by_regime.keys())
    n = len(regs); ncols = 2 if n > 1 else 1; nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.6*nrows), squeeze=False, sharey=sharey)
    axes = axes.flatten()

    for i, reg in enumerate(regs):
        df = details_by_regime[reg].copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Details für '{reg}' brauchen DatetimeIndex.")

        for c in SUPPORT_COLS:
            if c not in df.columns: df[c] = 0.0

        agg = df[SUPPORT_COLS].resample(freq).sum()
        idx, bar_width, locator, formatter = _idx_width_loc_fmt(agg.index, freq)
        agg.index = idx

        ax = axes[i]
        if stacked:
            pos = agg.clip(lower=0); neg = agg.clip(upper=0)
            bottom_pos = np.zeros(len(agg)); bottom_neg = np.zeros(len(agg))
            for c in SUPPORT_COLS:
                vals_p = pos[c].to_numpy()
                if np.any(vals_p != 0):
                    ax.bar(idx, vals_p, width=bar_width, bottom=bottom_pos,
                           color=SUPPORT_COLORS[c], edgecolor="none", label=c)
                    bottom_pos += vals_p
            for c in SUPPORT_COLS:
                vals_n = neg[c].to_numpy()
                if np.any(vals_n != 0):
                    ax.bar(idx, vals_n, width=bar_width, bottom=bottom_neg,
                           color=SUPPORT_COLORS[c], edgecolor="none", alpha=0.95, label=c)
                    bottom_neg += vals_n
        else:
            for c in SUPPORT_COLS:
                series = agg[c]
                if (series != 0).any():
                    ax.plot(series.index, series.values, label=c, lw=2, color=SUPPORT_COLORS[c])

        total = agg.sum(axis=1)
        ax.plot(total.index, total.values, color="black", lw=2.2, label="Summe Förderung")

        ax.set_title(f"{reg} — Förderung ({freq})")
        ax.set_ylabel("€")
        ax.grid(axis="y", alpha=0.35)
        ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        if i == 0:
            handles = [mpatches.Patch(color=SUPPORT_COLORS[c], label=c) for c in SUPPORT_COLS]
            handles.append(plt.Line2D([], [], color="black", lw=2.2, label="Summe Förderung"))
            fig.legend(handles, [h.get_label() for h in handles], ncol=4, loc="upper center", frameon=True)

    for j in range(i+1, len(axes)): fig.delaxes(axes[j])
    fig.suptitle(title, fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()


# ===== 13) Übersichtstabelle (Summaries) =====================================

def summarize_details(details, freq: str = "MS") -> pd.DataFrame:
    """
    Aggregiert je Periode u. a.:
      - Energy_Act_MWh, Energy_Sched_MWh
      - Revenue_DA_€, Revenue_ID_€, reBAP_€
      - Support_FIT_€, Support_MPM_€, Support_CfD_€, Support_Total_€
      - Netto_€
      - Buybacks_QH, Buyback_%QH
      - LateProd_QH, LateProd_%QH, LateProd_hours (falls vorhanden)
      - Decision_ON_%QH

    Eingaben:
      - dict[str, DataFrame]: (regime -> df)
      - MultiIndex-DataFrame mit Level "regime"
      - Einzel-DataFrame

    Rückgabe:
      - dict / MultiIndex → Index (regime, period)
      - Einzel-DataFrame → Index (period)
    """
    Δ = 0.25

    def _safe(df: pd.DataFrame, col: str) -> pd.Series:
        return df[col] if col in df.columns else pd.Series(0.0, index=df.index)

    def _one(df: pd.DataFrame) -> pd.DataFrame:
        act = df["Act_MW"]
        sched = df["Sched_MW"]

        da  = _safe(df, "Revenue_DA_€")
        idr = _safe(df, "Revenue_ID_€")
        rbp = _safe(df, "reBAP_€")
        fit = _safe(df, "FIT_€")
        mpm = _safe(df, "MPM_€")
        cfd = _safe(df, "CfD_€")

        net = df["Netto_€"] if "Netto_€" in df.columns else (da + idr + rbp + fit + mpm + cfd)

        buy   = df["buyback"].astype(int) if "buyback" in df.columns else pd.Series(0, index=df.index)
        lp_qh = df["late_production_qh"].astype(int) if "late_production_qh" in df.columns else pd.Series(0, index=df.index)
        lp_h  = df["late_production_hour"].astype(int) if "late_production_hour" in df.columns else pd.Series(0, index=df.index)
        dec_qh= df["decision_qh"].astype(int) if "decision_qh" in df.columns else pd.Series(0, index=df.index)

        out = pd.DataFrame({
            "Energy_Act_MWh":   (act*Δ).resample(freq).sum(),
            "Energy_Sched_MWh": (sched*Δ).resample(freq).sum(),

            "Revenue_DA_€": da.resample(freq).sum(),
            "Revenue_ID_€": idr.resample(freq).sum(),
            "reBAP_€":      rbp.resample(freq).sum(),

            "Support_FIT_€": fit.resample(freq).sum(),
            "Support_MPM_€": mpm.resample(freq).sum(),
            "Support_CfD_€": cfd.resample(freq).sum(),
        })
        out["Support_Total_€"] = out[["Support_FIT_€","Support_MPM_€","Support_CfD_€"]].sum(axis=1)
        out["Netto_€"] = net.resample(freq).sum()

        out["Buybacks_QH"]   = buy.resample(freq).sum()
        out["Buyback_%QH"]   = buy.resample(freq).mean().mul(100)

        out["LateProd_QH"]   = lp_qh.resample(freq).sum()
        out["LateProd_%QH"]  = lp_qh.resample(freq).mean().mul(100)

        # 'late_production_hour' ist stündlich – deshalb zuerst stündlich bündeln, dann erneut auf freq
        if "late_production_hour" in df.columns:
            lp_hours = lp_h.groupby(df.index.floor("h")).max()
            out["LateProd_hours"] = lp_hours.resample(freq).sum()
        else:
            out["LateProd_hours"] = 0

        out["Decision_ON_%QH"] = dec_qh.resample(freq).mean().mul(100)
        return out

    # Eingabeformen unterstützen
    if isinstance(details, dict):
        parts = []
        for reg, d in details.items():
            x = _one(d)
            x.insert(0, "regime", reg)
            parts.append(x)
        res = pd.concat(parts)
        return res.set_index("regime", append=True).swaplevel(0,1).sort_index()

    if isinstance(details, pd.DataFrame) and "regime" in (details.index.names or []):
        out = []
        for reg, d in details.groupby(level="regime"):
            x = _one(d.droplevel("regime"))
            x.insert(0, "regime", reg)
            out.append(x)
        res = pd.concat(out)
        return res.set_index("regime", append=True).swaplevel(0,1).sort_index()

    # Einzel-DataFrame
    return _one(details)
