<<<<<<< HEAD
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
# - ENTSO-E-Datenbank (Tabellen: spec, vals) via SQLAlchemy-Engine aus src.config.get_engine()
# - reBAP: CSV (z.B. "reBAP_utc.csv") mit deutschem Dezimalkomma und ";"-Trennung
# - ID1:   Excel (z.B. "id1_price_utc.xlsx") mit Spalten:
#          ["TimeStamp UTC linksgestempelt", "id1"]
#
# Rückgabe
# --------
# - pd.DataFrame `data_final`: vollständig gemergt, sortiert, getrimmt (bis letzte Zeile,
#   in der alle Spalten vorhanden sind), Index=DateTime.
#
# Benutzung
# ---------
# from src.data_import import build_data_from_sources
# df = build_data_from_sources(
#     rebap_csv=Path("base/data/reBAP_utc.csv"),
#     id1_xlsx=Path("base/data/id1_price_utc.xlsx"),
#     years=(2021, 2022, 2023, 2024, 2025)
# )
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import get_engine


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
        SELECT * FROM vals
        WHERE TimeSeriesID IN ({id_list})
          AND YEAR(`DateTime`) = {int(y)}
        """
        values_list.append(pd.read_sql_query(q, engine))

    values = pd.concat(values_list, ignore_index=True) if values_list else pd.DataFrame()
    data = pd.merge(values, targets, on="TimeSeriesID", how="left")
    return data


# -----------------------------------------------------------------------------
# Hauptfunktion
# -----------------------------------------------------------------------------
def build_data_from_sources(
    *,
    rebap_csv: Path,
    id1_xlsx: Path,
    years: Iterable[int] = (2022, 2023, 2024, 2025),
) -> pd.DataFrame:
    """
    Baut den finalen DataFrame (Index=DateTime) aus ENTSO-E (DB) + reBAP (CSV) + ID1 (XLSX).

    Parameter
    ---------
    rebap_csv : Path
        Pfad zur reBAP CSV (mit Spalten "Date", "Time", "rebap"), deutsch formatiert ("," Dezimal, ";" Separator).
    id1_xlsx : Path
        Pfad zur ID1-Excel (mit Spalten "TimeStamp UTC linksgestempelt", "id1").
    years : Iterable[int]
        Jahre, die aus der DB gelesen werden.

    Rückgabe
    --------
    data_final : pd.DataFrame
        Vollständig gemergter DataFrame (siehe columns_final unten), Index=DateTime (tz-naiv).
    """
    engine = get_engine()

    # ------------------ SPEC laden ------------------
    spec = pd.read_sql_query("SELECT * FROM spec", engine)

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
        SELECT * FROM vals
        WHERE TimeSeriesID IN ({id_list})
          AND YEAR(`DateTime`) = {int(y)}
        """
        values_list.append(pd.read_sql_query(q, engine))
    values_price = pd.concat(values_list, ignore_index=True)
    data_price = pd.merge(values_price, targets_price, on="TimeSeriesID")
    da_price = data_price.pivot(index="DateTime", columns="MapCode", values="Value")
    da_price.columns = ["da_price"]
    da_price = da_price.sort_index().resample("15min").ffill().reset_index()

    # ------------------ Merge alles zu 'data' ------------------
    data = (
        ee_da.merge(ee_id, on="DateTime", how="outer")
             .merge(ee_act, on="DateTime", how="outer")
             .merge(load_da, on="DateTime", how="outer")
             .merge(load_act, on="DateTime", how="outer")
             .merge(da_price, on="DateTime", how="left")
    )
    data["DateTime"] = pd.to_datetime(data["DateTime"], utc=False)
    data = data.sort_values("DateTime").set_index("DateTime")

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
    id1.index = id1.index.round("1min")
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

    # ------------------ Nachbearbeitung ------------------
    # DA-Preis auf 15-min Index füllen
    data["da_price"] = data["da_price"].ffill()

    # Anlagen in TransnetBW als "asset_*" umbenennen (wie in deiner Vorlage)
    data.columns = [
        c.replace("DE_TransnetBW_", "asset_") if c.startswith("DE_TransnetBW_") else c
        for c in data.columns
    ]

    # Skalierung: asset_* Größenordnung anpassen (Proxy → realistische Anlagengröße)
    if "asset_Wind Onshore_da" in data.columns:
        data["asset_Wind Onshore_da"] = data["asset_Wind Onshore_da"] / 500
    if "asset_Wind Onshore_id" in data.columns:
        data["asset_Wind Onshore_id"] = data["asset_Wind Onshore_id"] / 500
    if "asset_Wind Onshore_act" in data.columns:
        data["asset_Wind Onshore_act"] = data["asset_Wind Onshore_act"] / 500

    # ------------------ Finale Spaltenauswahl ------------------
    columns_final = [
        "da_price", "id1_price", "rebap", "Wind Onshore_marketvalue",
        "asset_Wind Onshore_da", "asset_Wind Onshore_id", "asset_Wind Onshore_act",
        "asset_Solar_da", "asset_Solar_id", "asset_Solar_act",
        "DE_Wind Onshore_act", "DE_Wind Onshore_da", "DE_Wind Onshore_id",
        "DE_Solar_act", "DE_Solar_da", "DE_Solar_id",
        "DE_Wind Offshore_act", "DE_Wind Offshore_da", "DE_Wind Offshore_id",
        "DE_Load_da", "DE_Load_act",
        "DE_res_da", "DE_res_act",
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
=======
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
# - ENTSO-E-Datenbank (Tabellen: spec, vals) via SQLAlchemy-Engine aus src.config.get_engine()
# - reBAP: CSV (z.B. "reBAP_utc.csv") mit deutschem Dezimalkomma und ";"-Trennung
# - ID1:   Excel (z.B. "id1_price_utc.xlsx") mit Spalten:
#          ["TimeStamp UTC linksgestempelt", "id1"]
#
# Rückgabe
# --------
# - pd.DataFrame `data_final`: vollständig gemergt, sortiert, getrimmt (bis letzte Zeile,
#   in der alle Spalten vorhanden sind), Index=DateTime.
#
# Benutzung
# ---------
# from src.data_import import build_data_from_sources
# df = build_data_from_sources(
#     rebap_csv=Path("base/data/reBAP_utc.csv"),
#     id1_xlsx=Path("base/data/id1_price_utc.xlsx"),
#     years=(2021, 2022, 2023, 2024, 2025)
# )
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import get_engine


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
        SELECT * FROM vals
        WHERE TimeSeriesID IN ({id_list})
          AND YEAR(`DateTime`) = {int(y)}
        """
        values_list.append(pd.read_sql_query(q, engine))

    values = pd.concat(values_list, ignore_index=True) if values_list else pd.DataFrame()
    data = pd.merge(values, targets, on="TimeSeriesID", how="left")
    return data


# -----------------------------------------------------------------------------
# Hauptfunktion
# -----------------------------------------------------------------------------
def build_data_from_sources(
    *,
    rebap_csv: Path,
    id1_xlsx: Path,
    years: Iterable[int] = (2022, 2023, 2024, 2025),
) -> pd.DataFrame:
    """
    Baut den finalen DataFrame (Index=DateTime) aus ENTSO-E (DB) + reBAP (CSV) + ID1 (XLSX).

    Parameter
    ---------
    rebap_csv : Path
        Pfad zur reBAP CSV (mit Spalten "Date", "Time", "rebap"), deutsch formatiert ("," Dezimal, ";" Separator).
    id1_xlsx : Path
        Pfad zur ID1-Excel (mit Spalten "TimeStamp UTC linksgestempelt", "id1").
    years : Iterable[int]
        Jahre, die aus der DB gelesen werden.

    Rückgabe
    --------
    data_final : pd.DataFrame
        Vollständig gemergter DataFrame (siehe columns_final unten), Index=DateTime (tz-naiv).
    """
    engine = get_engine()

    # ------------------ SPEC laden ------------------
    spec = pd.read_sql_query("SELECT * FROM spec", engine)

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
        SELECT * FROM vals
        WHERE TimeSeriesID IN ({id_list})
          AND YEAR(`DateTime`) = {int(y)}
        """
        values_list.append(pd.read_sql_query(q, engine))
    values_price = pd.concat(values_list, ignore_index=True)
    data_price = pd.merge(values_price, targets_price, on="TimeSeriesID")
    da_price = data_price.pivot(index="DateTime", columns="MapCode", values="Value")
    da_price.columns = ["da_price"]
    da_price = da_price.sort_index().resample("15min").ffill().reset_index()

    # ------------------ Merge alles zu 'data' ------------------
    data = (
        ee_da.merge(ee_id, on="DateTime", how="outer")
             .merge(ee_act, on="DateTime", how="outer")
             .merge(load_da, on="DateTime", how="outer")
             .merge(load_act, on="DateTime", how="outer")
             .merge(da_price, on="DateTime", how="left")
    )
    data["DateTime"] = pd.to_datetime(data["DateTime"], utc=False)
    data = data.sort_values("DateTime").set_index("DateTime")

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
    id1.index = id1.index.round("1min")
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

    # ------------------ Nachbearbeitung ------------------
    # DA-Preis auf 15-min Index füllen
    data["da_price"] = data["da_price"].ffill()

    # Anlagen in TransnetBW als "asset_*" umbenennen (wie in deiner Vorlage)
    data.columns = [
        c.replace("DE_TransnetBW_", "asset_") if c.startswith("DE_TransnetBW_") else c
        for c in data.columns
    ]

    # Skalierung: asset_* Größenordnung anpassen (Proxy → realistische Anlagengröße)
    if "asset_Wind Onshore_da" in data.columns:
        data["asset_Wind Onshore_da"] = data["asset_Wind Onshore_da"] / 500
    if "asset_Wind Onshore_id" in data.columns:
        data["asset_Wind Onshore_id"] = data["asset_Wind Onshore_id"] / 500
    if "asset_Wind Onshore_act" in data.columns:
        data["asset_Wind Onshore_act"] = data["asset_Wind Onshore_act"] / 500

    # ------------------ Finale Spaltenauswahl ------------------
    columns_final = [
        "da_price", "id1_price", "rebap", "Wind Onshore_marketvalue",
        "asset_Wind Onshore_da", "asset_Wind Onshore_id", "asset_Wind Onshore_act",
        "asset_Solar_da", "asset_Solar_id", "asset_Solar_act",
        "DE_Wind Onshore_act", "DE_Wind Onshore_da", "DE_Wind Onshore_id",
        "DE_Solar_act", "DE_Solar_da", "DE_Solar_id",
        "DE_Wind Offshore_act", "DE_Wind Offshore_da", "DE_Wind Offshore_id",
        "DE_Load_da", "DE_Load_act",
        "DE_res_da", "DE_res_act",
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
>>>>>>> 6bc1d56 (general update)
