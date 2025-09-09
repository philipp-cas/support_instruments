# src/reduced_strategies.py
# =============================================================================
# Core-Modul: reduced_strategies
# -----------------------------------------------------------------------------
# Zweck
# -----
# Simuliert das Dispatch- & Trading-Verhalten einer Anlage unter verschiedenen
# Förderregimes (NO, QUANT, FIT, FIT_PREMIUM, MPM, MPM_EX, CFD) auf QH-Basis.
#
# Design-Entscheidungen
# ---------------------
# - **Kein Datenimport hier.** Übergib ein DataFrame `df` mit DatetimeIndex.
# - **Δ = 0.25 h (Viertelstunde)** ist fest; DA-Entscheidung ist stündlich
#   (DA_MW = Mittel der 4 QHs der Stunde; p_DA stündlich, auf QH gespiegelt).
# - **Intraday (ID)** ist optional und nur aktiv, wenn **beide** Spalten
#   (forecast_id_col & id_price_col) gesetzt sind.
# - **MPM/MPM_EX** benötigen reale Monats-Marktwerte (`market_value_col`).
#   Für die Entscheidung wird ein Estimator genutzt:
#     * wenn `market_value_est_col` gesetzt → diese Spalte
#     * sonst: Vormonat von `market_value_col`.
# - **reBAP**-Vorzeichenlogik:
#     * Kosten (negativ), wenn r·d < 0
#     * Gutschrift (positiv), wenn r·d > 0
#     * 0 bei r==0 oder d==0
#
# Rückgaben
# ---------
# details : pd.DataFrame (QH-Zeitreihe) mit
#   - Preisen/Energien (p_DA_€/MWh, p_ID_€/MWh, rebap_€/MWh, DA_MW, ...)
#   - Fahrplan/Abweichung (DA_traded_MW, ID_traded_MW, Sched_MW, Act_MW, d_MW)
#   - Flags (decision_qh, buyback, late_production_qh/hour, ...)
#   - Zahlungen (Revenue_DA_€, Revenue_ID_€, reBAP_€, FIT_€, MPM_€, CfD_€, Netto_€)
#   - (MPM/EX) Prämien (prem_est_€/MWh, prem_real_€/MWh)
#
# totals  : dict mit Summen/Key-KPIs (z. B. Netto_€, Energie_*_MWh, Buyback_QHs)
#
# Minimalbeispiel (Daten kommen z. B. aus src.data_store.get_data())
# -----------------------------------------------------------------
# from src.data_store import get_data
# from src.reduced_strategies import reduced_strategies
#
# data = get_data()  # lädt base/data/data_final.csv (DatetimeIndex garantiert)
#
# details_no, totals_no = reduced_strategies(
#     df=data, regime="NO",
#     start_date="2022-01-01 00:00:00", end_date="2024-12-31 23:59:59",
#     forecast_id_col="asset_Wind Onshore_id", id_price_col="id1_price"
# )
#
# =============================================================================

from __future__ import annotations
import numpy as np
import pandas as pd


def reduced_strategies(
    df: pd.DataFrame,
    *,
    # --- Spalten (Defaults anpassbar) -----------------------------------------
    forecast_da_col: str = "asset_Wind Onshore_da",   # QH-DA-Forecast (MW), stündlich gemittelt
    actual_col:      str = "asset_Wind Onshore_act",  # QH-Ist (MW)
    da_price_col:    str = "da_price",                # €/MWh (stündlich; je QH konstant)
    rebap_col:       str = "rebap",                   # €/MWh (QH)

    # --- Optional: Intraday (beide oder keine setzen) -------------------------
    forecast_id_col: str | None = None,               # QH-ID-Forecast (MW); None => DA-only
    id_price_col:    str | None = None,               # €/MWh (QH); None => DA-only

    # --- Regime & Parameter ---------------------------------------------------
    regime: str = "NO",                               # "NO" | "QUANT" | "FIT" | "FIT_PREMIUM" | "MPM" | "MPM_EX" | "CFD"
    p_fit: float = 60.0,                              # €/MWh (FIT & FIT_PREMIUM)
    cfd_strike: float = 60.0,                         # €/MWh (CFD-Strike K)
    da_present_as_diff: bool = False,                 # CFD: DA als Diff vs. Act darstellen
    cfd_collapse_to_strike: bool = False,             # CFD: DA_EUR = 0; CfD = K * Act

    # --- MPM (Marktprämie) ----------------------------------------------------
    mpm_aw: float = 60.0,                             # anzulegender Wert A (€/MWh)
    market_value_col: str | None = "Wind Onshore_marketvalue",  # realer Monats-MV (für Auszahlung & Vormonats-Schätzer)
    market_value_est_col: str | None = None,          # optionaler MV-Schätzer; wenn None => Vormonat von market_value_col

    # --- Zeitraumfilter -------------------------------------------------------
    start_date=None,
    end_date=None,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulation des Dispatch-/Trading-Verhaltens für eine Anlage unter diversen Förderregimes.

    Parameter
    ---------
    df : pd.DataFrame
        Zeitreihe mit DatetimeIndex. Muss u. a. Spalten für Forecasts, Preise, Ist etc. enthalten.
    forecast_da_col, actual_col, da_price_col, rebap_col : str
        Spaltennamen für DA-Forecast, Ist, DA-Preis (€/MWh, stündlich), reBAP (€/MWh, QH).
    forecast_id_col, id_price_col : Optional[str]
        Wenn **beide** gesetzt → Intraday aktiv (ID-Forecast, ID-Preis). Sonst: DA-only.
    regime : str
        Eines von {"NO","QUANT","FIT","FIT_PREMIUM","MPM","MPM_EX","CFD"} (case-insensitive).
    p_fit : float
        fester €/MWh-Wert für FIT & FIT_PREMIUM.
    cfd_strike : float
        CfD-Strike K (€/MWh).
    da_present_as_diff, cfd_collapse_to_strike : bool
        Alternative Darstellungen für CfD-Zahlungen (berührt **nicht** die Entscheidungen).
    mpm_aw : float
        anzulegender Wert A (€/MWh) für MPM/MPM_EX.
    market_value_col : str | None
        Reale Monats-Marktwerte (€/MWh) – **Pflicht** für MPM/MPM_EX.
    market_value_est_col : str | None
        Estimator für die Entscheidung; wenn None → Vormonat von market_value_col.
    start_date, end_date
        optionaler Zeitraumfilter.

    Rückgabe
    --------
    details : pd.DataFrame
        QH-Zeitreihe mit Preisen, Energiemengen, Flags und Zahlungen.
    totals : dict
        Aggregierte Summen/KPIs (Netto_€, Energie_*_MWh, Buyback_QHs, ...).
    """
    # ---------- Eingangs-Prüfungen --------------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df needs a DatetimeIndex.")

    reg = regime.upper()
    Δ = 0.25   # Viertelstunde in Stunden
    eps = 1e-9

    # Intraday-Konsistenz: entweder beide Parameter gesetzt oder beide None
    if (forecast_id_col is None) ^ (id_price_col is None):
        raise ValueError(
            "Intraday erfordert sowohl 'forecast_id_col' (ID-Forecast) als auch 'id_price_col' (ID-Preis). "
            "Bitte beide setzen – oder beide None lassen."
        )
    use_id = (forecast_id_col is not None) and (id_price_col is not None)

    # MPM: harte Anforderung an market_value_col
    if reg in {"MPM", "MPM_EX"}:
        if (market_value_col is None) or (market_value_col not in df.columns):
            raise ValueError(
                "MPM/MPM_EX erfordern 'market_value_col' mit realisierten Monats-Marktwerten (€/MWh)."
            )

    # ---------- FIT (klassisch): Kurzpfad -------------------------------------
    # Keine Marktteilnahme: Fahrplan = Act, Zahlungen nur FIT_€
    if reg == "FIT":
        # Zeitraum-Subset (nur Ist benötigt)
        use = df.loc[
            (df.index >= (pd.to_datetime(start_date) if start_date else df.index.min())) &
            (df.index <= (pd.to_datetime(end_date)   if end_date   else df.index.max())),
            [actual_col]
        ].copy()

        act = use[actual_col].astype(float).to_numpy()

        out = pd.DataFrame(index=use.index)
        out["p_DA_€/MWh"]   = 0.0
        out["p_ID_€/MWh"]   = 0.0
        out["rebap_€/MWh"]  = 0.0
        out["DA_MW"]        = 0.0
        out["DA_traded_MW"] = 0.0
        out["ID_traded_MW"] = 0.0
        out["Sched_MW"]     = act
        out["Act_MW"]       = act
        out["Act_raw_MW"]   = act
        out["d_MW"]         = 0.0

        # Entscheidungs-/Diagnoseflags
        out["produce_hour_DA"]        = False
        out["produce_qh_ID_if_no_DA"] = False
        out["decision_qh"]            = True
        out["buyback"]                = False
        out["late_production_qh"]     = False
        out["late_production_hour"]   = False

        # Zahlungen
        out["Revenue_DA_€"] = 0.0
        out["Revenue_ID_€"] = 0.0
        out["reBAP_€"]      = 0.0
        out["FIT_€"]        = float(p_fit) * act * Δ
        out["MPM_€"]        = 0.0
        out["CfD_€"]        = 0.0
        out["Netto_€"]      = out[["Revenue_DA_€","Revenue_ID_€","reBAP_€","FIT_€","MPM_€","CfD_€"]].sum(axis=1)

        totals = {
            "Revenue_DA_€": 0.0,
            "Revenue_ID_€": 0.0,
            "reBAP_€":      0.0,
            "FIT_€":        float(out["FIT_€"].sum()),
            "MPM_€":        0.0,
            "CfD_€":        0.0,
            "Netto_€":      float(out["Netto_€"].sum()),
            "Energie_Sched_MWh": float((out["Sched_MW"] * Δ).sum()),
            "Energie_Act_MWh":   float((out["Act_MW"]   * Δ).sum()),
            "Buyback_QHs":       0,
        }
        return out, totals

    # ---------- Gemeinsame Vorbereitung ---------------------------------------
    # Spalten sammeln (abhängig von ID/MPM)
    cols = [forecast_da_col, actual_col, da_price_col, rebap_col]
    if use_id:
        cols += [forecast_id_col, id_price_col]

    need_mpm = (reg in {"MPM", "MPM_EX"})
    if need_mpm:
        if (market_value_est_col is not None) and (market_value_est_col in df.columns):
            cols.append(market_value_est_col)
        cols.append(market_value_col)  # market_value_col ist garantiert vorhanden

    # Zeitraum-Subset
    use = df.loc[
        (df.index >= (pd.to_datetime(start_date) if start_date else df.index.min())) &
        (df.index <= (pd.to_datetime(end_date)   if end_date   else df.index.max())),
        cols
    ].copy()

    # Stündliche Ableitungen (DA_MW: Mittel; p_DA: stündlich, auf QH gespiegelt)
    hour_idx = use.index.floor("h")
    use["DA_MW"]        = use.groupby(hour_idx)[forecast_da_col].transform("mean").astype(float)
    use["p_DA_€/MWh"]   = use.groupby(hour_idx)[da_price_col].transform("first").astype(float)
    use["rebap_€/MWh"]  = use[rebap_col].astype(float)
    use["Act_raw_MW"]   = use[actual_col].astype(float)
    use["p_ID_€/MWh"]   = use[id_price_col].astype(float) if use_id else 0.0

    # ----- MPM: Prämien (prem_est für Entscheidung, prem_real für Auszahlung) -
    prem_est = prem_real = None
    if need_mpm:
        # Monatswerte der realisierten Marktwerte
        mv_monthly = df[market_value_col].groupby(df.index.to_period("M")).last()
        months = use.index.to_period("M")

        # Auszahlung: realer Monatswert
        mv_real = mv_monthly.reindex(months).to_numpy(dtype=float)

        # Estimator: explizite Spalte oder Vormonat
        if (market_value_est_col is not None) and (market_value_est_col in df.columns):
            mv_est = df.loc[use.index, market_value_est_col].to_numpy(dtype=float)
        else:
            mv_est = mv_monthly.reindex(months - 1).to_numpy(dtype=float)

        aw = float(mpm_aw)
        prem_est  = np.maximum(aw - mv_est,  0.0)  # Entscheidung
        prem_real = np.maximum(aw - mv_real, 0.0)  # Auszahlung

        # MPM_EX: in Stunden mit negativem DA-Preis Prämie auf 0 setzen (Est + Real)
        if reg == "MPM_EX":
            neg_mask = use["p_DA_€/MWh"].to_numpy() < 0.0
            prem_est[neg_mask]  = 0.0
            prem_real[neg_mask] = 0.0
            # (Optional: neg_mask für Diagnose behalten)

    # ---------- Entscheidungslogik --------------------------------------------
    # DA-Entscheidung (stündlich, aber auf QH gespiegelt)
    if reg in {"NO", "QUANT"}:
        da_trade_mask = use["p_DA_€/MWh"] > 0.0
    elif reg == "FIT_PREMIUM":
        da_trade_mask = (use["p_DA_€/MWh"] + float(p_fit)) > 0.0
    elif reg in {"MPM", "MPM_EX"}:
        # prem_est hat QH-Auflösung (über Months auf QH gemappt)
        da_trade_mask = (use["p_DA_€/MWh"].to_numpy() + prem_est) > 0.0
    elif reg == "CFD":
        da_trade_mask = np.ones(len(use), dtype=bool)  # immer DA
    else:
        raise ValueError('regime must be one of {"NO","QUANT","FIT","FIT_PREMIUM","MPM","MPM_EX","CFD"}.')

    # gehandelte DA-Menge (QH-spiegel)
    use["DA_traded_MW"] = np.where(da_trade_mask, use["DA_MW"].to_numpy(), 0.0)

    # ID-Handel inkl. Buyback-Regeln
    if use_id:
        if reg in {"NO", "QUANT"}:
            id_trade_std = np.where(
                use["DA_traded_MW"] > 0.0,
                use[forecast_id_col].to_numpy() - use["DA_traded_MW"].to_numpy(),
                np.where(use["p_ID_€/MWh"] > 0.0, use[forecast_id_col].to_numpy(), 0.0)
            )
            buyback_mask = (use["p_ID_€/MWh"] < 0.0) & (use["DA_traded_MW"] > 0.0)

        elif reg == "FIT_PREMIUM":
            id_trade_std = np.where(
                use["DA_traded_MW"] > 0.0,
                use[forecast_id_col].to_numpy() - use["DA_traded_MW"].to_numpy(),
                np.where((use["p_ID_€/MWh"] + float(p_fit)) > 0.0, use[forecast_id_col].to_numpy(), 0.0)
            )
            buyback_mask = (use["p_ID_€/MWh"] < -float(p_fit)) & (use["DA_traded_MW"] > 0.0)

        elif reg in {"MPM", "MPM_EX"}:
            id_trade_std = np.where(
                use["DA_traded_MW"] > 0.0,
                use[forecast_id_col].to_numpy() - use["DA_traded_MW"].to_numpy(),
                np.where((use["p_ID_€/MWh"].to_numpy() + prem_est) > 0.0,
                         use[forecast_id_col].to_numpy(), 0.0)
            )
            buyback_mask = (use["p_ID_€/MWh"].to_numpy() < -prem_est) & (use["DA_traded_MW"].to_numpy() > 0.0)

        elif reg == "CFD":
            id_trade_std = np.where(
                use["DA_traded_MW"] > 0.0,
                use[forecast_id_col].to_numpy() - use["DA_traded_MW"].to_numpy(),
                0.0
            )
            buyback_mask = (use["p_ID_€/MWh"] < (use["p_DA_€/MWh"] - float(cfd_strike))) & (use["DA_traded_MW"] > 0.0)
        else:
            raise ValueError("Unknown regime in ID handling.")

        id_trade = id_trade_std.copy()
        # Buyback = komplette DA-Menge zurückkaufen
        id_trade[buyback_mask] = - use.loc[buyback_mask, "DA_traded_MW"].to_numpy()
    else:
        id_trade = np.zeros(len(use))
        buyback_mask = np.zeros(len(use), dtype=bool)

    # ---------- Fahrplan & Abweichungen ---------------------------------------
    sched   = use["DA_traded_MW"].to_numpy() + id_trade
    act_eff = use["Act_raw_MW"].to_numpy().copy()
    # Abschalten: wenn Nettomenge == 0 → Act_eff = 0
    act_eff[np.isclose(sched, 0.0, atol=eps)] = 0.0
    d = act_eff - sched

    # Späte Produktion: QH-Flag und Hour-Flag
    late_qh = (use["DA_traded_MW"].to_numpy() <= eps) & (sched > eps)
    da_all_zero_h   = use["DA_traded_MW"].groupby(hour_idx).transform(lambda s: (s <= eps).all()).to_numpy()
    sched_any_pos_h = pd.Series(sched, index=use.index).groupby(hour_idx).transform(lambda s: (s > eps).any()).to_numpy()
    late_hour = da_all_zero_h & sched_any_pos_h

    # ---------- Zahlungen ------------------------------------------------------
    # reBAP (Kosten -, Gutschrift +)
    r = use["rebap_€/MWh"].to_numpy()
    prod = r * d
    rebap_eur = np.where(prod < 0, -np.abs(r)*np.abs(d)*Δ,
                 np.where(prod > 0,  np.abs(r)*np.abs(d)*Δ, 0.0))

    # Markterlöse
    da_eur = use["p_DA_€/MWh"].to_numpy() * use["DA_traded_MW"].to_numpy() * Δ
    id_eur = use["p_ID_€/MWh"].to_numpy() * id_trade * Δ

    # Förderzahlungen
    fit_eur = np.zeros(len(use))
    mpm_eur = np.zeros(len(use))
    cfd_eur = np.zeros(len(use))

    if reg == "FIT_PREMIUM":
        fit_eur = float(p_fit) * act_eff * Δ

    if need_mpm:
        mpm_eur = prem_real * act_eff * Δ  # Auszahlung mit realer Prämie

    if reg == "CFD":
        if da_present_as_diff:
            # DA = p_DA * (DA_traded - Act); CfD = K * Act
            da_eur  = use["p_DA_€/MWh"].to_numpy() * (use["DA_traded_MW"].to_numpy() - act_eff) * Δ
            cfd_eur = float(cfd_strike) * act_eff * Δ
        elif cfd_collapse_to_strike:
            # Alles in CfD (DA_EUR = 0); CfD = K * Act
            da_eur  = np.zeros(len(use))
            cfd_eur = float(cfd_strike) * act_eff * Δ
        else:
            # Standard: CfD = (K - p_DA) * Act
            cfd_eur = (float(cfd_strike) - use["p_DA_€/MWh"].to_numpy()) * act_eff * Δ

    # ---------- Output-Tabellen ------------------------------------------------
    out = pd.DataFrame(index=use.index)
    out["p_DA_€/MWh"]   = use["p_DA_€/MWh"]
    out["p_ID_€/MWh"]   = use["p_ID_€/MWh"]
    out["rebap_€/MWh"]  = use["rebap_€/MWh"]
    out["DA_MW"]        = use["DA_MW"]
    out["DA_traded_MW"] = use["DA_traded_MW"]
    out["ID_traded_MW"] = id_trade
    out["Sched_MW"]     = sched
    out["Act_MW"]       = act_eff
    out["Act_raw_MW"]   = use["Act_raw_MW"]
    out["d_MW"]         = d

    # Flags
    produce_hour_DA = da_trade_mask.to_numpy() if hasattr(da_trade_mask, "to_numpy") else np.asarray(da_trade_mask)
    id_on_if_no_DA  = (use["DA_traded_MW"].to_numpy() == 0.0) & (id_trade > 0.0)
    decision_qh     = (produce_hour_DA | id_on_if_no_DA) & (~buyback_mask)

    out["produce_hour_DA"]        = produce_hour_DA.astype(bool)
    out["produce_qh_ID_if_no_DA"] = id_on_if_no_DA.astype(bool)
    out["decision_qh"]            = decision_qh.astype(bool)
    out["buyback"]                = buyback_mask.astype(bool)
    out["late_production_qh"]     = late_qh.astype(bool)
    out["late_production_hour"]   = late_hour.astype(bool)

    # Zahlungen
    out["Revenue_DA_€"] = da_eur
    out["Revenue_ID_€"] = id_eur
    out["reBAP_€"]      = rebap_eur
    out["FIT_€"]        = fit_eur
    out["MPM_€"]        = mpm_eur
    out["CfD_€"]        = cfd_eur
    out["Netto_€"]      = out[["Revenue_DA_€","Revenue_ID_€","reBAP_€","FIT_€","MPM_€","CfD_€"]].sum(axis=1)

    # nur für MPM/EX
    if need_mpm:
        out["prem_est_€/MWh"]  = prem_est
        out["prem_real_€/MWh"] = prem_real

    # ---------- Aggregation (totals) ------------------------------------------
    totals = {
        "Revenue_DA_€": float(out["Revenue_DA_€"].sum()),
        "Revenue_ID_€": float(out["Revenue_ID_€"].sum()),
        "reBAP_€":      float(out["reBAP_€"].sum()),
        "FIT_€":        float(out["FIT_€"].sum()),
        "MPM_€":        float(out["MPM_€"].sum()),
        "CfD_€":        float(out["CfD_€"].sum()),
        "Netto_€":      float(out["Netto_€"].sum()),
        "Energie_Sched_MWh": float((out["Sched_MW"] * Δ).sum()),
        "Energie_Act_MWh":   float((out["Act_MW"]   * Δ).sum()),
        "Buyback_QHs":       int(out["buyback"].sum()),
    }

    return out, totals
