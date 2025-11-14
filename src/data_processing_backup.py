# -*- coding: utf-8 -*-
"""
End-to-End Data Preparation Script
----------------------------------
Zweck:
------
Dieses Skript lädt und kombiniert alle relevanten Strommarktdaten
(ENTSO-E, reBAP, ID1, Futures, etc.) zu einem konsistenten 15-Minuten-Datensatz.

Darauf aufbauend werden:
1. Ein simuliertes deutsches Wind-Asset (2 % TransnetBW) erzeugt,
2. ein belgisches Referenz-Asset gebildet (skaliert auf vergleichbare Größe),
3. der geschätzte Marktwert (as of 11:00) pro Monat berechnet.

Das finale Dataset enthält alle Spalten, die für spätere Analysen
(z. B. Financial CfD-Simulationen) notwendig sind.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.data_import import build_data_from_sources


# =============================================================================
# 0) Allgemeine Einstellungen und Pfade
# =============================================================================

DATA_DIR = Path(r"C:\Users\Philipp.Castro\OneDrive - Universitaet Duisburg-Essen\Dokumente\Git\support_instruments\data")

ASSET_SHARE = 0.02  # Wir modellieren ein Asset, das 2 % der installierten TransnetBW-Kapazität repräsentiert
YEARS = (2021, 2022, 2023, 2024, 2025)


# =============================================================================
# 1) Rohdaten laden (15-Minuten-Auflösung)
# =============================================================================

# Diese Funktion ruft die ENTSO-E-Datenbank ab, liest reBAP, ID1, Futures, BE-Referenzdaten und MStR-Daten,
# harmonisiert sie alle auf ein gemeinsames Zeitraster und gibt einen großen DataFrame zurück.
data = build_data_from_sources(
    rebap_csv        = DATA_DIR / "reBAP_utc.csv",
    id1_xlsx         = DATA_DIR / "id1_price_utc.xlsx",
    be_csv           = DATA_DIR / "Belgium_elia.csv",
    mstr_csv         = DATA_DIR / "marktstammdatenregister_windleistung_transnetbw.csv",
    hochrechnung_csv = DATA_DIR / "Netztransparenz_WindOnshore_Hochrechnungen.csv",
    futures_dir      = DATA_DIR / "futures",
    years            = YEARS,
)

# Alias hinzufügen: für spätere Modelle nennen wir die Futures-Spalte konsistent „future_current_month“
if "monthly_future_base" in data.columns:
    data["future_current_month"] = data["monthly_future_base"]


# =============================================================================
# 2) Definition des simulierten Wind-Assets und Referenz-Assets (Belgien)
# =============================================================================

# -----------------------------------------------------------------------------
# a) Deutsches Wind-Asset: 2 % der installierten Kapazität in TransnetBW
# -----------------------------------------------------------------------------
# Dieses Asset ist eine rein proportionale Skalierung der TransnetBW-Wind-Zeitreihen.
# Damit kann das Verhalten eines kleinen Portfolios abgebildet werden,
# das sich wie das gesamte TSO-Gebiet verhält.

for suffix in ["da", "id", "act", "capacity"]:
    src = f"TransnetBW_Wind Onshore_{suffix}"
    if src in data.columns:
        data[f"asset_Wind Onshore_{suffix}"] = data[src] * ASSET_SHARE


# -----------------------------------------------------------------------------
# b) Belgisches Referenz-Asset (BE)
# -----------------------------------------------------------------------------
# In Belgien gibt es zusätzlich Informationen über aktivierte mFRR-Maßnahmen
# (manuelle Redispatches). Wenn Windanlagen abgeregelt werden, steht in
# "BE_Wind Onshore_mFRR" eine NEGATIVE Zahl. Um die „wahre“ ungedrosselte
# Erzeugung zu approximieren, subtrahieren wir diesen Wert:
# (–10 MW mFRR bedeutet +10 MW tatsächliche Einspeisung, die verhindert wurde)
# => BE_Wind Onshore_act = act - mFRR

if {"BE_Wind Onshore_act", "BE_Wind Onshore_mFRR"}.issubset(data.columns):
    data["BE_Wind Onshore_mFRR"] = data["BE_Wind Onshore_mFRR"].fillna(0.0)
    data["BE_Wind Onshore_act"]  = data["BE_Wind Onshore_act"] - data["BE_Wind Onshore_mFRR"]


# -----------------------------------------------------------------------------
# c) Belgische Referenz auf gleiche Assetgröße skalieren
# -----------------------------------------------------------------------------
# Da das belgische Windportfolio viel größer ist als das simulierte deutsche Asset,
# skalieren wir es proportional zur Kapazität. So kann das Referenz-Asset direkt
# für CfD-Vergleiche verwendet werden.
#
# Skalenfaktor:
#     scale_t = (TransnetBW_Capacity_t * ASSET_SHARE) / BE_Capacity_t

if {"TransnetBW_Wind Onshore_capacity", "BE_Wind Onshore_capacity"}.issubset(data.columns):
    denom = data["BE_Wind Onshore_capacity"].replace(0, np.nan)
    scale = (data["TransnetBW_Wind Onshore_capacity"] * ASSET_SHARE / denom)
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for suffix in ["da", "da_11am", "id", "act", "capacity"]:
        src = f"BE_Wind Onshore_{suffix}"
        if src in data.columns:
            data[f"asset_ref_Wind Onshore_{suffix}"] = data[src] * scale


# =============================================================================
# 3) Monatlicher Marktwert-Schätzer „as of 11:00“
# =============================================================================
#
# Idee:
# -----
# Der Marktwert wird an jedem Tag um 11:00 neu geschätzt – so, als ob wir zu diesem Zeitpunkt
# nur die vergangenen Stunden des Monats (REAL) und die Futures-Erwartung für den Rest des Monats
# (REST) kennen würden.
#
# Vorgehen:
# 1. Wir resampeln auf Stundenraster, um Rauschen zu reduzieren.
# 2. Für jeden Tag 11:00:
#    - Realer Teil = Wind-gewichteter Durchschnittspreis bis 11:00
#    - Rest = Future-Preis * Verhältnis von Wind-Marktwert zu Basepreis (aus Rolling-Window)
#    - Windprofil des Restes ≈ typisches Tagesprofil der letzten x Tage
# 3. Daraus ergibt sich ein „synthetischer“ Marktwert bis Monatsende.
#

# --- Stundenraster ---
df_h = data.resample("h").mean(numeric_only=True)

# Liste aller Tage + Zeitpunkte 11:00
days = pd.Index(sorted(df_h.index.normalize().unique()))
asof_11 = [d + pd.Timedelta(hours=11) for d in days]

rows = []

for asof in asof_11:
    if asof not in df_h.index:
        continue

    # Monatsgrenzen (inkl. Monatsende + 1 Tag → exklusiv)
    m_start = pd.Timestamp(asof.year, asof.month, 1)
    m_end   = (m_start + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(days=1)
    x_days  = (m_end - m_start).days
    if x_days <= 0:
        continue

    # -------------------------------------------------------------------------
    # REAL-Teil: bis 11:00
    # -------------------------------------------------------------------------
    past = df_h[(df_h.index >= m_start) & (df_h.index < asof)]
    if past.empty:
        num_real = 0.0
        den_real = 0.0
    else:
        num_real = float((past["da_price"] * past["DE_Wind Onshore_hochrechnung"]).sum())
        den_real = float(past["DE_Wind Onshore_hochrechnung"].sum())

    # -------------------------------------------------------------------------
    # REST-Teil: ab 11:00 bis Monatsende-1h
    # -------------------------------------------------------------------------
    rest_idx = pd.date_range(asof, m_end - pd.Timedelta(hours=1), freq="h")
    if len(rest_idx) == 0:
        # Monatsletzter Tag: kein Rest → Marktwert = Realteil
        mw_hat = np.nan if den_real == 0 else num_real / den_real
        rows.append((asof.normalize(), asof.month, mw_hat))
        continue

    # Rolling-Fenster: letzte x Tage für Basisrelationen
    win = df_h[(df_h.index >= asof - pd.Timedelta(days=x_days)) & (df_h.index < asof)]
    if win.empty:
        win = df_h[df_h.index < asof]

    # Korrekturfaktor: Verhältnis Wind-gewichteter Preis zu Basepreis
    num_win  = (win["da_price"] * win["DE_Wind Onshore_hochrechnung"]).sum()
    den_win  = win["DE_Wind Onshore_hochrechnung"].sum()
    mw_win   = float(num_win / den_win) if den_win > 0 else np.nan
    base_win = float(win["da_price"].mean()) if not win.empty else np.nan
    factor   = 1.0 if (not np.isfinite(mw_win) or not np.isfinite(base_win) or base_win == 0) else float(mw_win / base_win)

    # Future-basierter Basepreis für den Rest (Fallback: Durchschnitt der letzten x Tage)
    if "future_current_month" in df_h.columns:
        fut_series = df_h.loc[:asof, "future_current_month"].ffill()
        base_rest = float(fut_series.iloc[-1]) if len(fut_series) > 0 and np.isfinite(fut_series.iloc[-1]) else np.nan
    else:
        base_rest = np.nan

    if not np.isfinite(base_rest):
        # Fallback, falls Future leer oder NaN
        base_rest = float(win["da_price"].mean()) if not win.empty else float(df_h.loc[:asof, "da_price"].tail(24).mean())

    # Anpassung Base → Wind durch Faktor
    price_rest_adj = base_rest * factor

    # -------------------------------------------------------------------------
    # Wind-Profil des Restes: typisches stündliches Tagesprofil der letzten x Tage
    # -------------------------------------------------------------------------
    hourly_wind = win.groupby(win.index.hour)["DE_Wind Onshore_hochrechnung"].mean()
    rest_wind = pd.Series(index=rest_idx, dtype=float)
    if hourly_wind.empty:
        rest_wind[:] = 0.0
    else:
        rest_wind[:] = [hourly_wind.get(h, 0.0) for h in rest_idx.hour]

    # Restliche Windmenge gewichten mit korrigiertem Preis
    num_rest = float(price_rest_adj * rest_wind.sum())
    den_rest = float(rest_wind.sum())

    # -------------------------------------------------------------------------
    # Gesamtschätzung des Marktwerts
    # -------------------------------------------------------------------------
    num_all = num_real + num_rest
    den_all = den_real + den_rest
    mw_hat  = np.nan if den_all == 0 else (num_all / den_all)

    rows.append((asof.normalize(), asof.month, mw_hat))


# Ergebnis: ein täglicher Schätzwert (as of 11:00)
df_est = pd.DataFrame(rows, columns=["date", "month", "Wind Onshore_marketvalue_est"]).drop_duplicates("date")

# Auf das 15-Minuten-Raster des Haupt-Datasets zurückführen
data = data.merge(df_est.set_index("date"), left_index=True, right_index=True, how="left")
data["Wind Onshore_marketvalue_est"] = data["Wind Onshore_marketvalue_est"].ffill()

# ============================================================================================
# 4) JÄHRLICHER MARKTWERT-SCHÄTZER (mit monatl. Vorjahres-Faktoren)
# ============================================================================================

# 1) Stundenraster bilden
df_h = data.resample("h").mean(numeric_only=True)

# 2) Hilfsfunktionen ----------------------------------------------------------
def _wind_base_factor(hourly_df: pd.DataFrame) -> float:
    """Wind/Base-Faktor = (wind-gewichteter Preis) / (Basepreis-Ø)"""
    if hourly_df.empty:
        return np.nan
    p = hourly_df["da_price"]
    w = hourly_df["DE_Wind Onshore_hochrechnung"]
    den = float(w.sum())
    mw_win = float((p * w).sum() / den) if den > 0 else np.nan
    base = float(p.mean()) if len(p) else np.nan
    if not np.isfinite(mw_win) or not np.isfinite(base) or base == 0:
        return np.nan
    return float(mw_win / base)

def _monthly_factors_prev_year(df_h: pd.DataFrame, year: int) -> dict[int, float]:
    """Monats-Faktoren (1..12) aus dem Vorjahr (Wind/Base) auf Stundenbasis."""
    prev = year - 1
    out = {}
    for m in range(1, 13):
        sel = df_h[(df_h.index.year == prev) & (df_h.index.month == m)]
        out[m] = _wind_base_factor(sel)
    return out

def _monthly_wind_profiles_prev_year(df_h: pd.DataFrame, year: int) -> dict[int, pd.Series]:
    """
    Für jeden Monat (1..12) ein 24h-Profil der DE_Wind Onshore_hochrechnung (Vorjahr),
    als pd.Series(index=0..23) – Ø je Stunde des Tages.
    """
    prev = year - 1
    prof = {}
    for m in range(1, 13):
        sel = df_h[(df_h.index.year == prev) & (df_h.index.month == m)]
        if not sel.empty:
            prof[m] = sel.groupby(sel.index.hour)["DE_Wind Onshore_hochrechnung"].mean()
        else:
            prof[m] = pd.Series(dtype=float)  # leer → später Fallback
    return prof

def _global_factor_rolling(df_h: pd.DataFrame, asof: pd.Timestamp, days_window: int) -> float:
    """Globaler Fallback-Faktor aus Rolling-Fenster (wie bisher)."""
    win = df_h[(df_h.index >= asof - pd.Timedelta(days=days_window)) & (df_h.index < asof)]
    if win.empty:
        win = df_h[df_h.index < asof]
    return _wind_base_factor(win)

# 3) as-of 11:00 Zeitpunkte (wie bisher)
days = sorted(df_h.index.normalize().unique())
asof_11 = [d + pd.Timedelta(hours=11) for d in days]

rows_year = []
for asof in asof_11:
    if asof not in df_h.index:
        continue

    # --------- Jahresgrenzen / Rest des Jahres ----------
    y_start = pd.Timestamp(asof.year, 1, 1)
    y_end   = pd.Timestamp(asof.year, 12, 31, 23, 0)  # inkl. letzte Stunde des Jahres
    # realisierter Teil: Jahresbeginn bis < asof
    past = df_h[(df_h.index >= y_start) & (df_h.index < asof)]

    # Zähler/DZ realer Teil (windgewichtet)
    if not past.empty:
        num_real = float((past["da_price"] * past["DE_Wind Onshore_hochrechnung"]).sum())
        den_real = float(past["DE_Wind Onshore_hochrechnung"].sum())
    else:
        num_real, den_real = 0.0, 0.0

    # --------- Monats-Faktoren & Windprofile aus dem Vorjahr ----------
    factors_by_month = _monthly_factors_prev_year(df_h, asof.year)  # {1..12: factor or nan}
    profiles_by_month = _monthly_wind_profiles_prev_year(df_h, asof.year)  # {1..12: Series(hour→MW)}

    # Globaler Fallback-Faktor (Rolling-Fenster Länge = Tage im Jahr)
    days_in_year = 366 if pd.Timestamp(asof.year, 12, 31).dayofyear == 366 else 365
    global_factor = _global_factor_rolling(df_h, asof, days_in_year)
    if not np.isfinite(global_factor):
        global_factor = 1.0  # letzter Notanker

    # Fallback-Profil (global, Vorjahr gesamt), falls Monatsprofil leer:
    prev_year = asof.year - 1
    prev_all = df_h[(df_h.index.year == prev_year)]
    if not prev_all.empty:
        fallback_profile = prev_all.groupby(prev_all.index.hour)["DE_Wind Onshore_hochrechnung"].mean()
    else:
        # ultimatives Fallback: Rolling-Fenster-Profil (letzte 30 Tage)
        win = df_h[(df_h.index >= asof - pd.Timedelta(days=30)) & (df_h.index < asof)]
        fallback_profile = win.groupby(win.index.hour)["DE_Wind Onshore_hochrechnung"].mean() if not win.empty else pd.Series(0.0, index=range(24))

    # --------- Restindex: asof .. Jahresende (stundenweise) ----------
    rest_idx = pd.date_range(asof, y_end, freq="h")
    if len(rest_idx) == 0:
        # Kein Rest → nur realisierter Teil
        mw_year_hat = np.nan if den_real == 0 else num_real / den_real
        rows_year.append((asof.normalize(), asof.year, mw_year_hat))
        continue

    # Basepreise für den Rest: bevorzugt monthly_future_base; fallback: Ø da_price der letzten 30 Tage
    # Bevorzugt den Alias verwenden
    base_col = "future_current_month" if "future_current_month" in df_h.columns else "monthly_future_base"
    if base_col in df_h.columns:
        base_rest_hourly = df_h.reindex(rest_idx)[base_col].ffill()
    else:
        win30 = df_h[(df_h.index >= asof - pd.Timedelta(days=30)) & (df_h.index < asof)]
        base_val = float(win30["da_price"].mean()) if not win30.empty else float(df_h.loc[:asof, "da_price"].tail(24).mean())
        base_rest_hourly = pd.Series(base_val, index=rest_idx)


    # 4) Monatsweise anwenden: faktor(month) * base_rest_hourly + Windprofil(month)
    #    und alles windgewichtet aggregieren.
    num_rest = 0.0
    den_rest = 0.0

    # Wir gehen tag/monat über rest_idx, um den richtigen Faktor/Profil je Monat zu nutzen
    for month in sorted(rest_idx.to_period("M").unique()):
        # alle Stunden dieses Monats im Rest
        month_idx = rest_idx[(rest_idx >= month.start_time) & (rest_idx <= month.end_time)]
        if len(month_idx) == 0:
            continue

        m = month.month

        # Faktor: Vorjahres-Faktor je Monat mit Fallback auf globalen Faktor
        f_m = factors_by_month.get(m, np.nan)
        if not np.isfinite(f_m):
            f_m = global_factor

        # Profil: Vorjahres-Monatsprofil (24h), ansonsten globales Fallback-Profil
        prof_m = profiles_by_month.get(m)
        if prof_m is None or prof_m.empty:
            prof_m = fallback_profile
        # auf 24 Stunden sicherstellen (falls Lücken)
        prof_m = prof_m.reindex(range(24)).fillna(0.0)

        # Windgewichte für die Stunden dieses Monats (einfacher Tageszyklus, über die Tage wiederholt)
        hours = pd.Index(month_idx.hour)
        wind_weights = pd.Series(hours.map(prof_m).to_numpy(dtype=float), index=month_idx)

        # Falls komplett 0 (extrem unwahrscheinlich), auf Epsilon setzen, damit keine Division durch 0
        if wind_weights.sum() == 0:
            wind_weights = pd.Series(1.0, index=month_idx)  # gleichgewichtet als Notfall
        else:
            wind_weights = pd.Series(wind_weights.values, index=month_idx)

        # Monatsweise korrigierter Preis
        price_adj = base_rest_hourly.loc[month_idx] * f_m

        num_rest += float((price_adj * wind_weights).sum())
        den_rest += float(wind_weights.sum())

    # 5) Gesamtjahres-Schätzer (realisiert + Rest)
    num_all = num_real + num_rest
    den_all = den_real + den_rest
    mw_year_hat = np.nan if den_all == 0 else num_all / den_all

    rows_year.append((asof.normalize(), asof.year, mw_year_hat))

# Ergebnis auf 15-Minuten zurückführen
df_year_est = (
    pd.DataFrame(rows_year, columns=["date", "year", "Wind Onshore_marketvalue_year_est"])
    .drop_duplicates("date")
    .set_index("date")
    .sort_index()
)

data = data.merge(df_year_est[["Wind Onshore_marketvalue_year_est"]], left_index=True, right_index=True, how="left")
data["Wind Onshore_marketvalue_year_est"] = data["Wind Onshore_marketvalue_year_est"].ffill()
# =============================================================================================


# =============================================================================
# 5) Abschluss: Speichern und Kurzübersicht
# =============================================================================

# Sortierung nach Zeit
data = data.sort_index()

# Speicherpfade
out_csv     = DATA_DIR / "data_complete.csv"

# Export im effizienten Parquet-Format und als lesbares CSV
data.to_csv(out_csv, index=True)

# Diagnoseausgabe
print("\n✅ Datenaufbau abgeschlossen.")
print("Gesamtform:", data.shape)
print("Zeitraum:", data.index.min(), "→", data.index.max())
print("Beispielspalten vorhanden:",
      all(c in data.columns for c in [
          "da_price", "DE_Wind Onshore_hochrechnung", "Wind Onshore_marketvalue_est",
          "future_current_month", "asset_Wind Onshore_act", "Wind Onshore_marketvalue_year_est",
          "asset_ref_Wind Onshore_act", "Wind Onshore_marketvalue_est"
      ]))
print("Gespeichert als:")

print(" -", out_csv)
