# src/data_import.py
# =============================================================================
# Datenaufbau aus ENTSO-E (DB) + externen Dateien (reBAP CSV, ID1 XLSX)
# ----------------------------------------------------------------------------- 
# Zweck
# -----
# Baut den finalen DataFrame für die Simulation:
#   - Index: DateTime (15min, tz-naiv)
#   - Spalten: Preise, Marktwerte, EE-DA/ID/ACT (Asset & DE), Load (DA/ACT),
#              Residual-Loads, usw. (siehe columns_final unten)
#
# Quellen
# -------
# - ENTSO-E-Datenbank (Tabellen: spec_old, vals_old) via SQLAlchemy-Engine aus src.config.get_engine()
# - reBAP: CSV (z.B. "reBAP_utc.csv") mit deutschem Dezimalkomma und ";"-Trennung
# - ID1:   Excel (z.B. "id1_price_utc.xlsx") mit Spalten:
#          ["TimeStamp UTC linksgestempelt", "id1"]
# - Belgium Onshore Wind Daten von elia: https://opendata.elia.be/explore/dataset/ods031/api/
#   more information here: file:///C:/Users/Philipp.Castro/Downloads/20221227_Congestion%20Management%20Incentive%202022%20-%20Final%20report_23_12_2022.pdf:
#   For onshore parks only 37% of the total installed capacity is measured in real time, the
#   rest being up-scaled based on the average power factor of the measured ones. A couple of days
#   later the tool receives accurate metering data for about 80% of the onshore parks. Then the upscalement is redone with this new information and
#   republished ex-post to improve the past data.
#   Diese Daten dienen als Referenzanlage für die Financial CfDs
#          Active decremental bids
#          This indicates wether wind power generation has been reduced following the activation by Elia
#          of mFRR decremental bids on at least one wind farm within the selected farms.
#
# - Marktstammdatenregister (MStR) CSV mit Wind-Onshore-Kapazitäten in TransnetBW
# Rückgabe
# --------
# - pd.DataFrame `data_final`: vollständig gemergt, sortiert, getrimmt (bis letzte Zeile,
#   in der alle Spalten vorhanden sind), Index=DateTime.
#
# Benutzung
# ---------
# from src.data_import import build_data_from_sources
# df = build_data_from_sources(
#     rebap_csv=Path("data/reBAP_utc.csv"),
#     id1_xlsx=Path("data/id1_price_utc.xlsx"),
#     be_csv=Path("data/Belgium_elia.csv"),
#     mstr_csv=Path("data/marktstammdatenregister_windleistung_transnetbw.csv"),
#     years=(2021, 2022, 2023, 2024, 2025),
# )
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import get_engine


# ----------------------------------------------------------------------------- 
# Hilfsfunktionen
# ----------------------------------------------------------------------------- 
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Macht MultiIndex-Spalten (z. B. nach pivot_table) flach und vereinheitlicht DateTime-Spaltennamen.
    """
    df = df.reset_index()
    df.columns = [
        "_".join(col) if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    df.columns = [c.replace("DateTime_", "DateTime") for c in df.columns]
    return df


def _fetch_entsoe_data(
    *,
    spec: pd.DataFrame,
    file_name: str,
    map_codes: Iterable[str],
    production_types: Iterable[str] | None = None,
    specification: Iterable[str] | None = None,
    years: Iterable[int],
) -> pd.DataFrame:
    """
    Holt Zeitreihen (vals) aus der ENTSO-E DB für eine FileName/Regionen-Kombination,
    optional gefiltert nach Produktionstypen/Specification, und merged sie mit 'spec'.
    """
    engine = get_engine()

    targets = spec[(spec["FileName"] == file_name) & (spec["MapCode"].isin(list(map_codes)))]
    if production_types is not None:
        targets = targets[targets["ProductionType"].isin(list(production_types))]
    if specification is not None:
        targets = targets[targets["Specification"].isin(list(specification))]

    if targets.empty:
        return pd.DataFrame(columns=["TimeSeriesID", "DateTime", "Value"]).assign(**{c: None for c in spec.columns})

    id_list = ", ".join(map(str, targets["TimeSeriesID"]))
    values_list: list[pd.DataFrame] = []
    for y in years:
        q = f"""
        SELECT * FROM vals_old
        WHERE TimeSeriesID IN ({id_list})
          AND YEAR(`DateTime`) = {int(y)}
        """
        values_list.append(pd.read_sql_query(q, engine))

    values = pd.concat(values_list, ignore_index=True) if values_list else pd.DataFrame()
    data = pd.merge(values, targets, on="TimeSeriesID", how="left")
    return data


# ----------------------------------------------------------------------
# NEU: Phelix Baseload Monats-Futures (CSV -> täglich -> 15min)
# ----------------------------------------------------------------------
def _load_monthly_future_base(futures_dir: Path = Path("data/futures")) -> pd.DataFrame:
    """
    Liest alle CSVs in data/futures mit Spalten 'Datum' und 'Preis',
    füllt *monatsweise* auf tägliche Auflösung (erst bfill, dann ffill),
    und gibt eine tägliche Serie 'monthly_future_base' zurück.
    """
    futures_dir = Path(futures_dir)
    files = sorted(futures_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Keine Futures-CSV-Dateien in: {futures_dir}")

    def _process_month_file(fp: Path) -> pd.DataFrame:
        df = pd.read_csv(fp, engine="python", delimiter=";")
        df.columns = [c.strip() for c in df.columns]
        if "Datum" not in df.columns or "Preis" not in df.columns:
            raise ValueError(f"Erwarte Spalten 'Datum' und 'Preis' in {fp.name}, gefunden: {df.columns.tolist()}")

        df["Datum"] = pd.to_datetime(df["Datum"], format="%Y-%m-%d", errors="raise")
        df["Preis"] = pd.to_numeric(df["Preis"], errors="raise")
        df = df.dropna(subset=["Datum"]).sort_values("Datum").drop_duplicates(subset=["Datum"])
        df = df.set_index("Datum")

        if df.empty:
            raise ValueError(f"Keine gültigen Datumswerte in {fp.name}")

        parts = []
        # Strikt monatsweise füllen, damit keine Werte in Nachbar-Monate „bluten“
        for _month, g in df.groupby(df.index.to_period("M")):
            start = pd.Timestamp(g.index.min().year, g.index.min().month, 1)
            end = start + pd.offsets.MonthEnd(0)
            full_idx = pd.date_range(start, end, freq="d")
            gg = g.reindex(full_idx)
            gg["Preis"] = gg["Preis"].bfill().ffill()  # erst rückwärts, dann vorwärts
            parts.append(gg[["Preis"]])

        out = pd.concat(parts).sort_index()
        out.index.name = "Date"
        return out

    filled = []
    for fp in files:
        try:
            filled.append(_process_month_file(fp))
        except Exception as e:
            print(f"Überspringe {fp.name} wegen Fehler: {e}")

    if not filled:
        raise RuntimeError("Keine Futures-Datei konnte verarbeitet werden.")

    daily = pd.concat(filled).sort_index()
    daily = daily[~daily.index.duplicated(keep="first")]  # bei Überlappungen ersten Wert behalten
    daily = daily.rename(columns={"Preis": "monthly_future_base"})
    return daily


# ----------------------------------------------------------------------------- 
# Hauptfunktion
# ----------------------------------------------------------------------------- 
def build_data_from_sources(
    *,
    rebap_csv: Path = Path("data/reBAP_utc.csv"),
    id1_xlsx: Path = Path("data/id1_price_utc.xlsx"),
    be_csv: Path = Path("data/Belgium_elia.csv"),
    mstr_csv: Path = Path("data/marktstammdatenregister_windleistung_transnetbw.csv"),
    hochrechnung_csv: Path = Path("data/Netztransparenz_WindOnshore_Hochrechnungen.csv"),
    futures_dir: Path = Path("data/futures"),
    years: Iterable[int] = (2021, 2022, 2023, 2024, 2025),
) -> pd.DataFrame:



    """
    Baut den finalen DataFrame (Index=DateTime) aus ENTSO-E (DB) + reBAP (CSV) + ID1 (XLSX).
    """

    engine = get_engine()

    # ------------------ SPEC laden ------------------
    spec = pd.read_sql_query("SELECT * FROM spec_old", engine)

    # ------------------ EE DA (Forecast) ------------------
    ee_da = _fetch_entsoe_data(
        spec=spec,
        file_name="DayAheadGenerationForecastForWindAndSolar_14.1.D",
        map_codes=["DE", "DE_TransnetBW"],
        years=years,
    ).pivot_table(index="DateTime", columns=["MapCode", "ProductionType"], values="Value")
    ee_da = _flatten_columns(ee_da)
    ee_da = ee_da.add_suffix("_da").rename(columns={"DateTime_da": "DateTime"})

    # ------------------ EE ID (Forecast) ------------------
    ee_id = _fetch_entsoe_data(
        spec=spec,
        file_name="CurrentGenerationForecastForWindAndSolar_14.1.D",
        map_codes=["DE", "DE_TransnetBW"],
        years=years,
    ).pivot_table(index="DateTime", columns=["MapCode", "ProductionType"], values="Value")
    ee_id = _flatten_columns(ee_id)
    ee_id = ee_id.add_suffix("_id").rename(columns={"DateTime_id": "DateTime"})

    # ------------------ EE ACT (Ist) ------------------
    ee_act = _fetch_entsoe_data(
        spec=spec,
        file_name="AggregatedGenerationPerType_16.1.B_C",
        map_codes=["DE", "DE_TransnetBW"],
        production_types=["Solar", "Wind Onshore", "Wind Offshore"],
        specification=["Output"],
        years=years,
    ).pivot_table(index="DateTime", columns=["MapCode", "ProductionType"], values="Value")
    ee_act = _flatten_columns(ee_act)
    ee_act = ee_act.add_suffix("_act").rename(columns={"DateTime_act": "DateTime"})

    # ------------------ Load DA/ACT ------------------
    load_da = _fetch_entsoe_data(
        spec=spec,
        file_name="DayAheadTotalLoadForecast_6.1.B",
        map_codes=["DE"],
        years=years,
    ).pivot_table(index="DateTime", columns="MapCode", values="Value")
    load_da = _flatten_columns(load_da).rename(columns={"DE": "DE_Load_da"})

    load_act = _fetch_entsoe_data(
        spec=spec,
        file_name="ActualTotalLoad_6.1.A",
        map_codes=["DE"],
        years=years,
    ).pivot_table(index="DateTime", columns="MapCode", values="Value")
    load_act = _flatten_columns(load_act).rename(columns={"DE": "DE_Load_act"})

    # ------------------ Day-Ahead Preise (60min → 15min) ------------------
    targets_price = spec[
        (spec["FileName"] == "DayAheadPrices_12.1.D") &
        (spec["MapCode"] == "DE_LU") &
        (spec["ResolutionCode"] == "PT60M")
    ]
    id_list = ", ".join(map(str, targets_price["TimeSeriesID"]))
    values_list = []
    for y in years:
        q = f"""
        SELECT * FROM vals_old
        WHERE TimeSeriesID IN ({id_list})
          AND YEAR(`DateTime`) = {int(y)}
        """
        values_list.append(pd.read_sql_query(q, engine))
    values_price = pd.concat(values_list, ignore_index=True)
    data_price = pd.merge(values_price, targets_price, on="TimeSeriesID")
    da_price = data_price.pivot(index="DateTime", columns="MapCode", values="Value")
    da_price.columns = ["da_price"]
    da_price = da_price.sort_index().resample("15min").ffill().reset_index()

    # ------------------ Merge ENTSO-E Quellen zu 'data' ------------------
    data = (
        ee_da.merge(ee_id, on="DateTime", how="outer")
             .merge(ee_act, on="DateTime", how="outer")
             .merge(load_da, on="DateTime", how="outer")
             .merge(load_act, on="DateTime", how="outer")
             .merge(da_price, on="DateTime", how="left")
    )
    data["DateTime"] = pd.to_datetime(data["DateTime"], utc=False)
    data = data.sort_values("DateTime").set_index("DateTime")

    # ------------------ Futures (monatlich -> täglich) einlesen ------------------
    fut_daily = _load_monthly_future_base(futures_dir=futures_dir)  # Tagesindex

    # 1) saubere, eindeutige Tagesserie
    fut_ser = fut_daily["monthly_future_base"].copy()
    fut_ser.index = pd.to_datetime(fut_ser.index).normalize()
    fut_ser = fut_ser[~fut_ser.index.duplicated(keep="last")].sort_index()

    # 2) eindeutige Ziel-Tage aus deinem 15-min Index
    unique_days = pd.DatetimeIndex(data.index).normalize().unique()

    # 3) auf die Tage ausrichten und auf alle 15-min Timestamps ausrollen
    mapped_days = fut_ser.reindex(unique_days, method="ffill")
    data["monthly_future_base"] = mapped_days.reindex(pd.DatetimeIndex(data.index).normalize()).to_numpy()

    # ------------------ Belgium CSV (Onshore wind) integrieren ------------------
    # Dieser Datensatz dient als Referenzanlage für die Financial CfDs
    be_raw = pd.read_csv(be_csv, sep=";")

    # Zeit parsen → echte UTC → tz-naiv (damit deckungsgleich mit deinem restlichen Index)
    be_raw["Datetime"] = pd.to_datetime(be_raw["Datetime"], utc=True, errors="raise")
    be_raw["Datetime"] = be_raw["Datetime"].dt.tz_localize(None)
    be_raw = be_raw.set_index("Datetime").sort_index()

    # Die Spalte "Decremental bid Indicator" muss in Float umgewandelt werden
    be_raw["Decremental bid Indicator"] = (
        be_raw["Decremental bid Indicator"]
        .astype(str)
        .str.replace("''", "", regex=False)   # doppelte '' entfernen
        .str.replace(",", ".", regex=False)   # deutsches Komma -> Punkt
        .str.strip()                          # Whitespace entfernen
        .replace("", pd.NA)                   # echte Leerstrings -> NaN
    )
    be_raw["Decremental bid Indicator"] = pd.to_numeric(
        be_raw["Decremental bid Indicator"],
        errors="raise"                       # falls doch Müll drin ist -> Exception
    ).fillna(0.0)

    # Nur Onshore
    be_on = be_raw[be_raw["Offshore/onshore"].astype(str).str.strip().str.lower() == "onshore"]

    # Nur die benötigten Spalten
    keep_cols = [
        "Measured & Upscaled",
        "Most recent forecast",
        "Day Ahead 11AM forecast",
        "Day-ahead 6PM forecast",
        "Monitored capacity",
        "Decremental bid Indicator",
    ]
    missing_cols = [c for c in keep_cols if c not in be_on.columns]
    if missing_cols:
        raise ValueError(f"BE CSV fehlt Spalten: {missing_cols}")
    be_on = be_on[keep_cols]

    # Über alle Regionen/Netzanschlüsse je Zeitstempel summieren
    be_sum = be_on.groupby(level=0).sum(numeric_only=True)

    # Nur die Jahre behalten, die im Parameter "years" angegeben sind
    be_sum = be_sum[be_sum.index.year.isin(list(years))]

    # Konsistente Spaltennamen
    columns_to_rename = {
        "Measured & Upscaled": "BE_Wind Onshore_act",
        "Most recent forecast": "BE_Wind Onshore_id",
        "Day Ahead 11AM forecast": "BE_Wind Onshore_da_11am",
        "Day-ahead 6PM forecast": "BE_Wind Onshore_da",
        "Monitored capacity": "BE_Wind Onshore_capacity",
        "Decremental bid Indicator": "BE_Wind Onshore_mFRR",
    }
    be_sum = be_sum.rename(columns=columns_to_rename)

    # Ausreißer-Block in BE_Wind Onshore_id (2023-01-05 13:00–13:45) glätten: Zeitinterpolation zwischen 12:45 und 14:00
    col = "BE_Wind Onshore_id"
    start = pd.Timestamp("2023-01-05 13:00:00")
    end   = pd.Timestamp("2023-01-05 13:45:00")
    mask = (be_sum.index >= start) & (be_sum.index <= end)
    if mask.any():
        s = be_sum[col].copy()
        s.loc[mask] = pd.NA
        s = s.interpolate(method="time", limit_direction="both")
        be_sum.loc[mask, col] = s.loc[mask]

    # Ausreißer in BE_Wind Onshore_capacity glätten: Nullwerte durch lineare Interpolation füllen
    col_cap = "BE_Wind Onshore_capacity"
    s_cap = be_sum[col_cap].copy()
    s_cap[s_cap == 0] = pd.NA  # Nullwerte als NA markieren
    s_cap = s_cap.interpolate(method="time", limit_direction="both", limit_area="inside")
    be_sum[col_cap] = s_cap

    # In Haupt-DataFrame mergen
    data = data.merge(be_sum, left_index=True, right_index=True, how="left")

    # ------------------ TransnetBW: installierte Onshore-Kapazität (MStR) ------------------
    # Direkt in dieser Funktion aufgebaut (deutsches Dezimalkomma, 15-min-Index, kumuliert).
    mstr_data = pd.read_csv(mstr_csv, sep=";")

    req = ["Bruttoleistung der Einheit", "Inbetriebnahmedatum der Einheit"]
    missing_mstr = [c for c in req if c not in mstr_data.columns]
    if missing_mstr:
        raise KeyError(f"Fehlende Spalten in mstr_csv: {missing_mstr}")

    mstr_data["Bruttoleistung_KW"] = (
        mstr_data["Bruttoleistung der Einheit"]
        .astype(str).str.replace(",", ".", regex=False).str.replace("\u00A0", "", regex=False).str.strip()
        .replace({"": pd.NA})
        .astype(float)
    )
    mstr_data["Inbetrieb_TS"] = pd.to_datetime(
        mstr_data["Inbetriebnahmedatum der Einheit"], format="%d.%m.%Y", errors="raise"
    ).dt.floor("15min")

    valid = mstr_data.dropna(subset=["Inbetrieb_TS", "Bruttoleistung_KW"])
    events = valid.groupby("Inbetrieb_TS", as_index=True)["Bruttoleistung_KW"].sum().sort_index()
    start_idx = events.index.min()
    end_idx = events.index.max()
    full_idx = pd.date_range(start=start_idx, end=end_idx, freq="15min")

    # Build cumulative capacity series on its own index first
    capacity_series = events.reindex(full_idx, fill_value=0.0).cumsum()
    # Assign to data only after all data is imported and index is finalized
    capacity_series_on_data = capacity_series.reindex(data.index, method="ffill")
    # After all merges and index operations, reindex to match data's index and transform to MW
    data["TransnetBW_Wind Onshore_capacity"] = capacity_series_on_data / 1000.0  # kW → MW

    # ------------------ Residual Loads ------------------
    # (nutzt .get(...,0), falls bestimmte DA/ACT-Spalten fehlen sollten)
    data["DE_res_da"] = (
        data.get("DE_Load_da", 0)
        - data.get("DE_Solar_da", 0)
        - data.get("DE_Wind Onshore_da", 0)
        - data.get("DE_Wind Offshore_da", 0)
    )
    data["DE_res_act"] = (
        data.get("DE_Load_act", 0)
        - data.get("DE_Solar_act", 0)
        - data.get("DE_Wind Onshore_act", 0)
        - data.get("DE_Wind Offshore_act", 0)
    )

    # ------------------ reBAP einlesen & mergen ------------------
    # CSV mit deutschem Dezimalkomma und ';' als Separator
    rebap = pd.read_csv(rebap_csv, sep=";")
    # Erwartete Spalten: "Date", "Time", "rebap"
    # Zeit bauen (UTC) → tz-naiv
    rebap["DateTime"] = pd.to_datetime(
        rebap["Date"] + " " + rebap["Time"], format="%d.%m.%Y %H:%M", utc=True
    ).dt.tz_localize(None)
    rebap = rebap[["DateTime", "rebap"]].set_index("DateTime").sort_index()
    # Deutsches Dezimalkomma zu Punkt
    if rebap["rebap"].dtype == object:
        rebap["rebap"] = rebap["rebap"].str.replace(",", ".", regex=False).astype(float)

    data = data.merge(rebap, left_index=True, right_index=True, how="left")

    # ------------------ ID1 einlesen & mergen ------------------
    id1 = pd.read_excel(
        id1_xlsx,
        usecols=["TimeStamp UTC linksgestempelt", "id1"]
    ).rename(columns={"TimeStamp UTC linksgestempelt": "DateTime", "id1": "id1_price"})
    id1["DateTime"] = pd.to_datetime(id1["DateTime"], format="%d.%m.%Y %H:%M", utc=True).dt.tz_localize(None)
    id1 = id1.set_index("DateTime").sort_index()
    # leichtes Runden, falls Stempel nicht exakt auf :00,:15,:30,:45
    id1.index = id1.index.round("15min")
    id1 = id1[id1.index.year.isin(list(years))]
    data = data.merge(id1, left_index=True, right_index=True, how="left")

    # ------------------ Marktwerte Wind Onshore (monatlich → 15min) ------------------
    mw_wind_onshore_ctkwh = {
        2021: [4.645, 4.361, 3.395, 4.353, 4.134, 6.330, 6.808, 7.253, 11.754, 10.982, 14.056, 16.077],
        2022: [12.883, 10.825, 19.766, 12.703, 13.242, 19.692, 27.824, 46.092, 28.238, 12.715, 13.718, 14.164],
        2023: [8.726, 10.620, 8.515, 8.940, 8.095, 9.236, 5.445, 6.613, 8.566, 6.864, 7.653, 4.409],
        2024: [6.502, 5.335, 5.538, 4.800, 5.608, 6.356, 4.985, 6.168, 6.266, 6.822, 8.881, 7.237],
    }
    mv_series = pd.concat([
        pd.Series(vals, index=pd.date_range(f"{year}-01-01", periods=12, freq="MS"))
        for year, vals in mw_wind_onshore_ctkwh.items()
    ]).sort_index()
    # c€/kWh → €/MWh (×10)
    data["Wind Onshore_marketvalue"] = mv_series.reindex(data.index, method="ffill").fillna(method="bfill") * 10

    # ------------------ Hochrechnungen der Wind Onshore Netzbetreiber einlesen & mergen ------------------

    # --- Hochrechnungen laden (original, stündlich) ---
    hr = pd.read_csv(hochrechnung_csv, delimiter=";")

    # Zeit -> Index (tz-naiv(original is UTC)). Format: "dd.mm.yyyy hh:mm"
    hr["DateTime"] = pd.to_datetime(hr["Zeit"], format="%d.%m.%Y %H:%M", errors="raise")
    hr = hr.set_index("DateTime").sort_index()

    # Unbenötigte Spalten entfernen
    for c in ["Zeitzone", "Zeit"]:
        if c in hr.columns:
            hr = hr.drop(columns=c)

    # Deutsches Komma -> Punkt; in float
    for c in hr.columns:
        if hr[c].dtype == object:
            hr[c] = hr[c].str.replace(",", ".", regex=False)
        hr[c] = pd.to_numeric(hr[c], errors="coerce")

    # DE-Summe über vorhandene TSO-Spalten
    tsos = [c for c in ["50Hertz", "Amprion", "TenneT TSO", "TransnetBW"] if c in hr.columns]
    hr["DE_Wind Onshore_hochrechnung"] = hr[tsos].sum(axis=1, min_count=1)

    # TransnetBW-Säule sinnvoll benennen (falls vorhanden)
    hr = hr.rename(columns={"TransnetBW": "TransnetBW_Wind Onshore_hochrechnung"})

    # Auf gewünschte Jahre einschränken
    hr = hr[hr.index.year.isin(list(years))]

    # Stündlich -> 15min (step hold)
    hr_15 = hr.resample("15min").ffill(limit=3)

    # Auf Datenindex rebasen (falls deine Daten nicht lückenlos sind)
    hr_15 = hr_15.reindex(data.index)

    # Merge in Haupt-DataFrame (linker Join)
    data = data.merge(hr_15, left_index=True, right_index=True, how="left")

    # ------------------ Nachbearbeitung ------------------
    # Anlagen in TransnetBW  umbenennen
    data.columns = [
        c.replace("DE_TransnetBW_", "TransnetBW_") if c.startswith("DE_TransnetBW_") else c
        for c in data.columns
    ]

    # ------------------ Finale Spaltenauswahl ------------------
    columns_final = [
        "da_price", "id1_price", "rebap", "Wind Onshore_marketvalue",
        "monthly_future_base",
        "TransnetBW_Wind Onshore_da", "TransnetBW_Wind Onshore_id", "TransnetBW_Wind Onshore_act",
        "TransnetBW_Solar_da", "TransnetBW_Solar_id", "TransnetBW_Solar_act", "TransnetBW_Wind Onshore_capacity", "TransnetBW_Wind Onshore_hochrechnung",
        "DE_Wind Onshore_act", "DE_Wind Onshore_da", "DE_Wind Onshore_id", "DE_Wind Onshore_hochrechnung",
        "DE_Solar_act", "DE_Solar_da", "DE_Solar_id",
        "DE_Wind Offshore_act", "DE_Wind Offshore_da", "DE_Wind Offshore_id",
        "DE_Load_da", "DE_Load_act",
        "DE_res_da", "DE_res_act",
        "BE_Wind Onshore_act", "BE_Wind Onshore_da", "BE_Wind Onshore_da_11am", "BE_Wind Onshore_id",
        "BE_Wind Onshore_capacity",
    ]

    missing = [c for c in columns_final if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in merged data: {missing}")

    data_final = data[columns_final].copy()

    # Bis zur letzten Zeile beschneiden, in der ALLE Spalten vorhanden sind
    last_valid_index = data_final.dropna().index[-1]
    data_final = data_final.loc[:last_valid_index].sort_index()

    # Sicherheit: Index als DatetimeIndex
    if not isinstance(data_final.index, pd.DatetimeIndex):
        data_final.index = pd.to_datetime(data_final.index)

    return data_final
