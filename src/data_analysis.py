# =============================================================================
# src/data_analysis.py
# =============================================================================
# Aufgabe:
#  - Nutzung des vorbereiteten Datensatzes (data_final.csv)
#  - Basisanalysen (Korrelationen, NaN-Checks, Histogramme)
#  - Vergleich Asset vs. Referenz
#  - Vergleich: tatsächlicher Monats- & Jahres-Marktwert vs. Schätzer (as of 11:00)
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 0) Helper: CSV robust laden (Zeitindex parsen)
# -----------------------------------------------------------------------------
def load_data_complete(csv_path: Path) -> pd.DataFrame:
    """
    Erwartet data_complete.csv mit einer Zeitspalte. Versucht in der Reihenfolge:
    1) 'DateTime' als Zeitspalte
    2) Erste Spalte als Index, parse_dates=True
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=["DateTime"])
        df = df.set_index("DateTime")
        return df.sort_index()
    except Exception:
        # Fallback: erste Spalte ist der Index
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return df.sort_index()

# -----------------------------------------------------------------------------
# 1) Datensatz laden
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
data = load_data_complete(DATA_DIR / "data_complete.csv")

print("✅ Datensatz erfolgreich geladen")
print(f"Zeitraum: {data.index.min()} – {data.index.max()}")
print(f"Spalten: {len(data.columns)}")

# -----------------------------------------------------------------------------
# 2) Korrelationsanalyse (jährlich)
# -----------------------------------------------------------------------------
columns_to_plot = [
    "asset_Wind Onshore_da",
    "asset_Wind Onshore_id",
    "asset_Wind Onshore_act",
    "asset_ref_Wind Onshore_da",
    "asset_ref_Wind Onshore_id",
    "asset_ref_Wind Onshore_act",
]

years = sorted({ts.year for ts in data.index})
correlation_matrices = {}

for y in years:
    df_year = data[data.index.year == y]
    existing = [c for c in columns_to_plot if c in df_year.columns]
    if not existing:
        continue
    corr = df_year[existing].corr()
    correlation_matrices[y] = corr
    print(f"\n=== Korrelationsmatrix für das Jahr {y} ===")
    print(corr.round(3))

# -----------------------------------------------------------------------------
# 3) Datenqualitätsanalyse (NaN-Bericht)
# -----------------------------------------------------------------------------
print("\n=== Übersicht fehlender Werte (NaN) ===")
for col in data.columns:
    m = data[col].isna()
    n_nan = int(m.sum())
    share = m.mean() * 100
    first_na = last_na = "-"
    if n_nan > 0:
        na_idx = data.index[m]
        first_na = na_idx[0]
        last_na  = na_idx[-1]
    print(f"{col:40s} NaN: {n_nan:7d} ({share:6.2f}%)  first: {first_na}  last: {last_na}")

# -----------------------------------------------------------------------------
# 4) Verteilungsanalyse (Asset vs. Referenz)
# -----------------------------------------------------------------------------
start_date, end_date = "2023-01-01", "2024-12-31"
mask = (data.index >= start_date) & (data.index <= end_date)
df_sub = data.loc[mask]

column_pairs = [
    ("asset_Wind Onshore_act", "asset_ref_Wind Onshore_act"),
    ("asset_Wind Onshore_da",  "asset_ref_Wind Onshore_da"),
    ("asset_Wind Onshore_id",  "asset_ref_Wind Onshore_id"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (col_asset, col_ref) in zip(axes, column_pairs):
    if col_asset not in df_sub.columns or col_ref not in df_sub.columns:
        ax.text(0.5, 0.5, "Spalte fehlt", ha="center", va="center")
        ax.set_title(f"{col_asset} vs. {col_ref}")
        continue
    ax.hist(df_sub[col_asset].dropna(), bins=60, alpha=0.6, label="Asset", density=True)
    ax.hist(df_sub[col_ref].dropna(), bins=60, alpha=0.6, label="Referenz", density=True)
    ax.set_title(col_asset.replace("asset_", "").replace("_Wind Onshore_", ""))
    ax.set_xlabel("Wert [MW]")
    ax.legend()

fig.suptitle(f"Verteilungen {start_date}–{end_date}", fontsize=14)
fig.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 5) Vergleichs-Funktionen Marktwert (Monat / Jahr)
# -----------------------------------------------------------------------------
def compare_estimator(
    df: pd.DataFrame,
    *,
    col_true: str,
    col_est: str,
    year: int | None = None,
    daily: bool = True,
    title: str = "Marktwert: Realität vs. Schätzer",
):
    """
    Generische Vergleichsfunktion für zwei Zeitreihen (true vs. estimate).
    - daily=True: auf Tagesmittel resamplen (empfohlen).
    """
    if col_true not in df.columns or col_est not in df.columns:
        missing = [c for c in [col_true, col_est] if c not in df.columns]
        raise KeyError(f"Spalten nicht gefunden: {missing}")

    s_true = df[col_true].copy()
    s_est  = df[col_est].copy()

    if year is not None:
        mask = (df.index.year == year)
        s_true = s_true.loc[mask]
        s_est  = s_est.loc[mask]

    if s_true.empty or s_est.empty:
        raise ValueError("Zeitraum enthält keine Daten.")

    if daily:
        s_true = s_true.resample("D").mean()
        s_est  = s_est.resample("D").mean()

    s = pd.concat({"true": s_true, "est": s_est}, axis=1).dropna()
    if s.empty:
        raise ValueError("Keine überlappenden Daten zwischen Realität und Schätzer.")

    err  = s["est"] - s["true"]
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    mape = float(np.mean(np.abs(err / s["true"])) * 100)

    print(f"\n=== {title} ===")
    if year is not None:
        print(f"Jahr: {year}")
    print(f"MAE : {mae:.2f} €/MWh")
    print(f"RMSE: {rmse:.2f} €/MWh")
    print(f"Bias: {bias:.2f} €/MWh")
    print(f"MAPE: {mape:.2f} %")

    # Plot 1: Zeitreihe
    plt.figure(figsize=(12,5))
    plt.plot(s.index, s["true"], label="tatsächlich", color="black", lw=2)
    plt.plot(s.index, s["est"],  label="Schätzer (as of 11:00)", lw=2, alpha=0.9)
    ttl_year = f" – {year}" if year is not None else ""
    plt.title(f"{title}{ttl_year}\n"
              f"MAE={mae:.2f}, RMSE={rmse:.2f}, Bias={bias:.2f}, MAPE={mape:.2f}%")
    plt.ylabel("€/MWh"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    # Plot 2: Fehlerverlauf
    plt.figure(figsize=(12,3.5))
    plt.plot(err.index, err.values)
    plt.axhline(0, color="black", lw=1)
    plt.title("Fehler (Schätzer – Realität)")
    plt.ylabel("€/MWh"); plt.grid(True); plt.tight_layout(); plt.show()

    # Plot 3: Ø absoluter Fehler nach Kalendertag (nur sinnvoll für Monats-Schätzer)
    s2 = s.copy()
    if len(s2) > 0 and (s2.index.to_series().diff().dt.days.abs().median() <= 2):
        s2["day"] = s2.index.day
        err_abs_by_day = s2.groupby("day").apply(lambda d: np.mean(np.abs(d["est"] - d["true"])))
        plt.figure(figsize=(8,4))
        plt.plot(err_abs_by_day.index, err_abs_by_day.values, marker="o")
        plt.title("Ø absoluter Fehler nach Kalendertag")
        plt.xlabel("Tag im Monat"); plt.ylabel("|Fehler| [€/MWh]")
        plt.grid(True); plt.tight_layout(); plt.show()

    return {"MAE": mae, "RMSE": rmse, "Bias": bias, "MAPE_%": mape}

# -----------------------------------------------------------------------------
# 6) MONAT: Realität vs. Schätzer
#    Hinweis: Spaltennamen anpassen, falls du andere verwendest.
# -----------------------------------------------------------------------------
# 'Wind Onshore_marketvalue'     = ex-post realisierter Monatsmarktwert (15-min gefüllt)
# 'Wind Onshore_marketvalue_est' = Monats-Schätzer (as of 11:00)
monthly_true_col = "Wind Onshore_marketvalue"
monthly_est_col  = "Wind Onshore_marketvalue_est"

if monthly_true_col in data.columns and monthly_est_col in data.columns:
    compare_estimator(
        data,
        col_true=monthly_true_col,
        col_est=monthly_est_col,
        year=None,             # oder z.B. 2024
        daily=True,
        title="Monatlicher Marktwert (Wind Onshore): Realität vs. Schätzer",
    )
else:
    print("\n⚠️ Monats-Vergleich übersprungen (Spalten nicht gefunden).")

# -----------------------------------------------------------------------------
# 7) JAHR: Realität vs. Schätzer
# -----------------------------------------------------------------------------
# 'Wind Onshore_marketvalue_year_realized'   = ex-post Jahresmarktwert (falls vorhanden)
#   Falls nicht als Spalte vorhanden, wird er hier on-the-fly aus 15-min Daten berechnet.
# 'Wind Onshore_marketvalue_year_est'        = Jahres-Schätzer (as of 11:00)
yearly_est_col  = "Wind Onshore_marketvalue_year_est"

if yearly_est_col in data.columns:
    # Ex-post Jahresmarktwert bauen, falls nicht vorhanden:
    yearly_true_col = "Wind Onshore_marketvalue_year_realized"
    if yearly_true_col not in data.columns:
        # ex-post aus 15-min Daten: pro Jahr windgewichtet (mit Hochrechnung)
        if {"da_price", "DE_Wind Onshore_hochrechnung"}.issubset(data.columns):
            df_h = data.resample("h").mean(numeric_only=True)
            yy = []
            for y in sorted({ts.year for ts in df_h.index}):
                sel = df_h[df_h.index.year == y]
                p, w = sel["da_price"], sel["DE_Wind Onshore_hochrechnung"]
                den = float(w.sum())
                mv  = np.nan if den <= 0 else float((p * w).sum() / den)
                # schreibe als konstante Jahresreihe (ffill über das Jahr)
                idx_y = data[data.index.year == y].index
                ser = pd.Series(mv, index=idx_y)
                yy.append(ser)
            if yy:
                data[yearly_true_col] = pd.concat(yy).reindex(data.index).ffill()
        else:
            print("\n⚠️ Kann Jahres-Realität nicht berechnen (da_price oder Hochrechnung fehlt).")

    if yearly_true_col in data.columns:
        compare_estimator(
            data,
            col_true=yearly_true_col,
            col_est=yearly_est_col,
            year=None,           # oder z.B. 2024
            daily=True,
            title="Jährlicher Marktwert (Wind Onshore): Realität vs. Schätzer",
        )
    else:
        print("\n⚠️ Jahres-Vergleich übersprungen (Jahres-Realität nicht verfügbar).")
else:
    print("\n⚠️ Jahres-Vergleich übersprungen (Schätzer-Spalte fehlt).")

# -----------------------------------------------------------------------------
# 8) Zusatzplot: Differenz Extrapolation vs. Actuals (TransnetBW)
# -----------------------------------------------------------------------------
if {"TransnetBW_Wind Onshore_hochrechnung", "TransnetBW_Wind Onshore_act"}.issubset(data.columns):
    plt.figure(figsize=(12, 6))
    diff = data["TransnetBW_Wind Onshore_hochrechnung"] - data["TransnetBW_Wind Onshore_act"]
    plt.plot(data.index, diff, label="Difference")
    plt.title("Difference between Extrapolation and Actuals – TransnetBW Wind Onshore")
    plt.xlabel("DateTime"); plt.ylabel("Difference [MW]")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()
else:
    print("\nℹ️ TransnetBW Differenz-Plot ausgelassen (Spalten fehlen).")
