<<<<<<< HEAD
# scripts/run_2024_analysis.py
# =============================================================================
# Jahresanalyse 2024: DA+ID-Läufe, Monats-Plots, Conformity- & Finanz-Sensitivitäten
# -----------------------------------------------------------------------------
# Voraussetzungen:
#   pip install -r requirements.txt
#   python scripts/build_cache.py   # falls data/data_final.csv noch fehlt
#
# Start (Terminal):
#   python scripts/run_2024_analysis.py
#
# Start (VS Code Interactive Window):
#   - Datei öffnen, gesamten Inhalt ausführen
#   - oder: Rechtsklick → "Run Current File in Interactive Window"
# =============================================================================

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_store import get_data
from src.reduced_strategies_call import run_all_da_id
from src.reduced_strategies_plots import (
    plot_strategies_breakdown_ID,
    plot_support_by_regime,
    plot_strategy_timeseries,
    plot_strategy_and_prices_timeseries,
    summarize_details,
)
from src.metrics import (
    conformity_sensitivity,
    financial_sensitivity,
)
from src.config import RESULTS_DIR

# ------------------------------ Konfiguration --------------------------------
START_2024 = "2024-01-01 00:00:00"
END_2024   = "2024-12-31 23:59:59"
REGS = ["NO", "FIT", "FIT_PREMIUM", "MPM", "MPM_EX", "CFD"]

# Speichern der PNGs?
SAVE_PNG = True  # auf False setzen, wenn nur Anzeige gewünscht

# Deltas für Sensitivitäten (€/MWh)
DELTAS = (-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70)

# ------------------------------- Main-Logik -----------------------------------
def main():
    outdir = Path(RESULTS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Daten laden
    df = get_data(force_refresh=False)
    print(f"[OK] Cache geladen: {df.shape}, {df.index.min()} → {df.index.max()}")

    # 2) Alle Regime als DA+ID rechnen
    details_all, _totals_unused = run_all_da_id(REGS, data=df)

    # 3) Auf 2024 filtern (MultiIndex: level 'regime', 'DateTime')
    dt = details_all.index.get_level_values("DateTime")
    details_all = (
        details_all.loc[
            (dt >= pd.to_datetime(START_2024)) & (dt <= pd.to_datetime(END_2024))
        ]
        .sort_index()
    )

    # 4) Totals 2024 aus Details aggregieren
    totals_2024 = (
        details_all
        .groupby(level="regime")[["Revenue_DA_€","Revenue_ID_€","reBAP_€","FIT_€","MPM_€","CfD_€","Netto_€"]]
        .sum()
    )
    energy = details_all.groupby(level="regime")[["Sched_MW","Act_MW"]].sum().mul(0.25)
    totals_2024["Energie_Sched_MWh"] = energy["Sched_MW"]
    totals_2024["Energie_Act_MWh"]   = energy["Act_MW"]

    print("\n[Totals 2024] (Ausschnitt)")
    print(totals_2024.round(2).head(10))

    # 5) Plots: Erlösstruktur & Förderung (monatlich)
    plot_strategies_breakdown_ID(
        details_all, freq="MS",
        title="Erlösstruktur je Regime – 2024 (DA+ID)"
    )
    if SAVE_PNG:
        plt.savefig(outdir / "plot_breakdown_2024.png", dpi=160, bbox_inches="tight")
    plt.show()

    plot_support_by_regime(
        details_all, freq="MS",
        title="Förderhöhe je Regime – 2024 (DA+ID)", stacked=True
    )
    if SAVE_PNG:
        plt.savefig(outdir / "plot_support_2024.png", dpi=160, bbox_inches="tight")
    plt.show()

    # Optional: Zeitreihen-Diagnose für ein Regime
    if "CFD" in details_all.index.get_level_values("regime"):
        plot_strategy_timeseries(
            details_all.xs("CFD", level="regime"),
            title="CFD – Fahrplan vs. Ist – 2024",
            start=START_2024, end=END_2024, sample=None
        )
        if SAVE_PNG:
            plt.savefig(outdir / "plot_timeseries_CFD_2024.png", dpi=160, bbox_inches="tight")
        plt.show()

    if "NO" in details_all.index.get_level_values("regime"):
        plot_strategy_and_prices_timeseries(
            details_all.xs("NO", level="regime"),
            title="NO – Fahrplan, Ist & Preise – Jan 2024",
            start="2024-01-01", 
            end="2024-01-07", 
            sample=None
        )

    # 6) Zusammenfassungstabelle (monatlich)
    summary_2024 = summarize_details(details_all, freq="MS")
    print("\n[Summary 2024] (letzte Zeilen)")
    print(summary_2024.tail(6).round(2))

    # 7) Conformity-Sensitivität (Index 0..1) berechnen
    sens_idx = conformity_sensitivity(
        df=df,
        start=START_2024, end=END_2024,
        deltas=DELTAS,
        weight="Act_raw_MW",   # << das ist jetzt die Gewichtung
    )



    # ---- Plot: Conformity (Linie je Regime) ----
    plt.figure(figsize=(10, 5))
    for reg, grp in sens_idx.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp["MarketSimilarityIndex"], marker="o", label=reg)
    plt.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline")
    plt.ylabel("Market Conformity Index (0…1)")
    plt.title("Conformity-Sensitivität – 2024")
    plt.grid(alpha=0.35)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    if SAVE_PNG:
        plt.savefig(outdir / "plot_conformity_sensitivity_2024.png", dpi=160)
    plt.show()

    # ---- Plot: Conformity Heatmap (Δ × Regime) ----
    pivot = sens_idx.pivot(index="Delta", columns="Regime", values="MarketSimilarityIndex")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Conformity'})
    plt.title("Conformity-Heatmap – 2024")
    plt.xlabel("Regime"); plt.ylabel("Δ (€/MWh)")
    plt.tight_layout()
    if SAVE_PNG:
        plt.savefig(outdir / "plot_conformity_heatmap_2024.png", dpi=160)
    plt.show()

    # 8) Finanzielle Sensitivität (Netto & Förderung) berechnen
    sens_fin = financial_sensitivity(
        df=df,
        start=START_2024, end=END_2024,
        deltas=DELTAS,
    )

    # ---- Plot: Netto-Erlöse über Δ ----
    plt.figure(figsize=(10, 5))
    for reg, grp in sens_fin.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp["Netto_€"], marker="o", label=reg)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline")
    plt.ylabel("Netto € (Summe 2024)")
    plt.title("Finanzielle Sensitivität – Nettoerlöse – 2024")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_PNG:
        plt.savefig(outdir / "plot_financial_sensitivity_netto_2024.png", dpi=160)
    plt.show()

    # ---- Plot: Förderung (Summe FIT+MPM+CfD) über Δ ----
    plt.figure(figsize=(10, 5))
    for reg, grp in sens_fin.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp["Förderung_€"], marker="o", label=reg)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline")
    plt.ylabel("Förderung € (Summe 2024)")
    plt.title("Finanzielle Sensitivität – Förderung – 2024")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_PNG:
        plt.savefig(outdir / "plot_financial_sensitivity_support_2024.png", dpi=160)
    plt.show()

    # # 9) Einen Regime-DF herausziehen
    # d_no = details_all.xs("NO", level="regime")

    # # Direkt mit Strings als Skalar aufrufen
    # plot_strategy_timeseries(
    #     d_no,
    #     title="NO – Fahrplan vs. Ist – 2024",
    #     start="2024-01-01",
    #     end="2024-12-31",
    #     sample=None
    # )

    # 9) CSVs speichern
    (outdir / "details_DA_ID_2024.csv").write_text(
        details_all.reset_index().to_csv(index=False), encoding="utf-8"
    )
    totals_2024.to_csv(outdir / "totals_DA_ID_2024.csv")
    sens_idx.to_csv(outdir / "sensitivity_conformity_2024.csv", index=False)
    sens_fin.to_csv(outdir / "sensitivity_financial_2024.csv", index=False)
    print(f"\n[OK] CSVs und Plots gespeichert unter: {outdir}")

if __name__ == "__main__":
    main()
=======
# scripts/run_2024_analysis.py
# =============================================================================
# Jahresanalyse 2024: DA+ID-Läufe, Monats-Plots, Conformity- & Finanz-Sensitivitäten
# -----------------------------------------------------------------------------
# Voraussetzungen:
#   pip install -r requirements.txt
#   python scripts/build_cache.py   # falls data/data_final.csv noch fehlt
#
# Start (Terminal):
#   python scripts/run_2024_analysis.py
#
# Start (VS Code Interactive Window):
#   - Datei öffnen, gesamten Inhalt ausführen
#   - oder: Rechtsklick → "Run Current File in Interactive Window"
# =============================================================================

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_store import get_data
from src.reduced_strategies_call import run_all_da_id
from src.reduced_strategies_plots import (
    plot_strategies_breakdown_ID,
    plot_support_by_regime,
    plot_strategy_timeseries,
    plot_strategy_and_prices_timeseries,
    summarize_details,
)
from src.metrics import (
    conformity_sensitivity,
    financial_sensitivity,
)
from src.config import RESULTS_DIR

# ------------------------------ Konfiguration --------------------------------
START_2024 = "2024-01-01 00:00:00"
END_2024   = "2024-12-31 23:59:59"
REGS = ["NO", "FIT", "FIT_PREMIUM", "MPM", "MPM_EX", "CFD"]

# Speichern der PNGs?
SAVE_PNG = True  # auf False setzen, wenn nur Anzeige gewünscht

# Deltas für Sensitivitäten (€/MWh)
DELTAS = (-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70)

# ------------------------------- Main-Logik -----------------------------------
def main():
    outdir = Path(RESULTS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Daten laden
    df = get_data(force_refresh=False)
    print(f"[OK] Cache geladen: {df.shape}, {df.index.min()} → {df.index.max()}")

    # 2) Alle Regime als DA+ID rechnen
    details_all, _totals_unused = run_all_da_id(REGS, data=df)

    # 3) Auf 2024 filtern (MultiIndex: level 'regime', 'DateTime')
    dt = details_all.index.get_level_values("DateTime")
    details_all = (
        details_all.loc[
            (dt >= pd.to_datetime(START_2024)) & (dt <= pd.to_datetime(END_2024))
        ]
        .sort_index()
    )

    # 4) Totals 2024 aus Details aggregieren
    totals_2024 = (
        details_all
        .groupby(level="regime")[["Revenue_DA_€","Revenue_ID_€","reBAP_€","FIT_€","MPM_€","CfD_€","Netto_€"]]
        .sum()
    )
    energy = details_all.groupby(level="regime")[["Sched_MW","Act_MW"]].sum().mul(0.25)
    totals_2024["Energie_Sched_MWh"] = energy["Sched_MW"]
    totals_2024["Energie_Act_MWh"]   = energy["Act_MW"]

    print("\n[Totals 2024] (Ausschnitt)")
    print(totals_2024.round(2).head(10))

    # 5) Plots: Erlösstruktur & Förderung (monatlich)
    plot_strategies_breakdown_ID(
        details_all, freq="MS",
        title="Erlösstruktur je Regime – 2024 (DA+ID)"
    )
    if SAVE_PNG:
        plt.savefig(outdir / "plot_breakdown_2024.png", dpi=160, bbox_inches="tight")
    plt.show()

    plot_support_by_regime(
        details_all, freq="MS",
        title="Förderhöhe je Regime – 2024 (DA+ID)", stacked=True
    )
    if SAVE_PNG:
        plt.savefig(outdir / "plot_support_2024.png", dpi=160, bbox_inches="tight")
    plt.show()

    # Optional: Zeitreihen-Diagnose für ein Regime
    if "CFD" in details_all.index.get_level_values("regime"):
        plot_strategy_timeseries(
            details_all.xs("CFD", level="regime"),
            title="CFD – Fahrplan vs. Ist – 2024",
            start=START_2024, end=END_2024, sample=None
        )
        if SAVE_PNG:
            plt.savefig(outdir / "plot_timeseries_CFD_2024.png", dpi=160, bbox_inches="tight")
        plt.show()

    if "NO" in details_all.index.get_level_values("regime"):
        plot_strategy_and_prices_timeseries(
            details_all.xs("NO", level="regime"),
            title="NO – Fahrplan, Ist & Preise – Jan 2024",
            start="2024-01-01", 
            end="2024-01-07", 
            sample=None
        )

    # 6) Zusammenfassungstabelle (monatlich)
    summary_2024 = summarize_details(details_all, freq="MS")
    print("\n[Summary 2024] (letzte Zeilen)")
    print(summary_2024.tail(6).round(2))

    # 7) Conformity-Sensitivität (Index 0..1) berechnen
    sens_idx = conformity_sensitivity(
        df=df,
        start=START_2024, end=END_2024,
        deltas=DELTAS,
        weight="Act_raw_MW",   # << das ist jetzt die Gewichtung
    )



    # ---- Plot: Conformity (Linie je Regime) ----
    plt.figure(figsize=(10, 5))
    for reg, grp in sens_idx.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp["MarketSimilarityIndex"], marker="o", label=reg)
    plt.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline")
    plt.ylabel("Market Conformity Index (0…1)")
    plt.title("Conformity-Sensitivität – 2024")
    plt.grid(alpha=0.35)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    if SAVE_PNG:
        plt.savefig(outdir / "plot_conformity_sensitivity_2024.png", dpi=160)
    plt.show()

    # ---- Plot: Conformity Heatmap (Δ × Regime) ----
    pivot = sens_idx.pivot(index="Delta", columns="Regime", values="MarketSimilarityIndex")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Conformity'})
    plt.title("Conformity-Heatmap – 2024")
    plt.xlabel("Regime"); plt.ylabel("Δ (€/MWh)")
    plt.tight_layout()
    if SAVE_PNG:
        plt.savefig(outdir / "plot_conformity_heatmap_2024.png", dpi=160)
    plt.show()

    # 8) Finanzielle Sensitivität (Netto & Förderung) berechnen
    sens_fin = financial_sensitivity(
        df=df,
        start=START_2024, end=END_2024,
        deltas=DELTAS,
    )

    # ---- Plot: Netto-Erlöse über Δ ----
    plt.figure(figsize=(10, 5))
    for reg, grp in sens_fin.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp["Netto_€"], marker="o", label=reg)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline")
    plt.ylabel("Netto € (Summe 2024)")
    plt.title("Finanzielle Sensitivität – Nettoerlöse – 2024")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_PNG:
        plt.savefig(outdir / "plot_financial_sensitivity_netto_2024.png", dpi=160)
    plt.show()

    # ---- Plot: Förderung (Summe FIT+MPM+CfD) über Δ ----
    plt.figure(figsize=(10, 5))
    for reg, grp in sens_fin.groupby("Regime"):
        grp = grp.sort_values("Delta")
        plt.plot(grp["Delta"], grp["Förderung_€"], marker="o", label=reg)
    plt.axhline(0.0, color="gray", lw=1, ls="--", alpha=0.6)
    plt.xlabel("Δ (€/MWh) relativ zur Baseline")
    plt.ylabel("Förderung € (Summe 2024)")
    plt.title("Finanzielle Sensitivität – Förderung – 2024")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_PNG:
        plt.savefig(outdir / "plot_financial_sensitivity_support_2024.png", dpi=160)
    plt.show()

    # # 9) Einen Regime-DF herausziehen
    # d_no = details_all.xs("NO", level="regime")

    # # Direkt mit Strings als Skalar aufrufen
    # plot_strategy_timeseries(
    #     d_no,
    #     title="NO – Fahrplan vs. Ist – 2024",
    #     start="2024-01-01",
    #     end="2024-12-31",
    #     sample=None
    # )

    # 9) CSVs speichern
    (outdir / "details_DA_ID_2024.csv").write_text(
        details_all.reset_index().to_csv(index=False), encoding="utf-8"
    )
    totals_2024.to_csv(outdir / "totals_DA_ID_2024.csv")
    sens_idx.to_csv(outdir / "sensitivity_conformity_2024.csv", index=False)
    sens_fin.to_csv(outdir / "sensitivity_financial_2024.csv", index=False)
    print(f"\n[OK] CSVs und Plots gespeichert unter: {outdir}")

if __name__ == "__main__":
    main()
>>>>>>> 6bc1d56 (general update)
