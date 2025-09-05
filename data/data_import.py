# src/data_import.py
from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from typing import Iterable, Optional
from .config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS

YEARS_DEFAULT = [2021, 2022, 2023, 2024, 2025]

def _engine():
    if not DB_USER or not DB_PASS:
        raise RuntimeError("DB_USER/DB_PASS fehlen. Lege sie in .env ab und lade sie in config.py.")
    url = f"mysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df.columns = ["_".join(c) if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [c.replace("DateTime_", "DateTime") for c in df.columns]
    return df

def fetch_spec(engine) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM spec", engine)

def fetch_entsoe_data(
    engine,
    spec: pd.DataFrame,
    file_name: str,
    map_codes: Iterable[str],
    production_types: Optional[Iterable[str]] = None,
    specification: Optional[Iterable[str]] = None,
    years: Iterable[int] = YEARS_DEFAULT,
) -> pd.DataFrame:
    targets = spec[(spec["FileName"] == file_name) & (spec["MapCode"].isin(map_codes))]
    if production_types:
        targets = targets[targets["ProductionType"].isin(production_types)]
    if specification:
        targets = targets[targets["Specification"].isin(specification)]
    if targets.empty:
        return pd.DataFrame()

    id_list = ", ".join(map(str, targets["TimeSeriesID"]))
    values_list = []
    for y in years:
        q = f"SELECT * FROM vals WHERE TimeSeriesID IN ({id_list}) AND YEAR(`DateTime`) = {y}"
        values_list.append(pd.read_sql_query(q, engine))
    values = pd.concat(values_list, ignore_index=True) if values_list else pd.DataFrame()
    if values.empty:
        return pd.DataFrame()

    data = pd.merge(values, targets, on="TimeSeriesID")
    return data

def _load_prices_da(engine, spec, years) -> pd.DataFrame:
    targets = spec[
        (spec["FileName"] == "DayAheadPrices_12.1.D")
        & (spec["MapCode"] == "DE_LU")
        & (spec["ResolutionCode"] == "PT60M")
    ]
    if targets.empty:
        return pd.DataFrame()

    id_list = ", ".join(map(str, targets["TimeSeriesID"]))
    chunks = []
    for y in years:
        q = f"SELECT * FROM vals WHERE TimeSeriesID IN ({id_list}) AND YEAR(`DateTime`) = {y}"
        chunks.append(pd.read_sql_query(q, engine))
    values = pd.concat(chunks, ignore_index=True)
    data = pd.merge(values, targets, on="TimeSeriesID")
    da_price = data.pivot(index="DateTime", columns="MapCode", values="Value")
    da_price.columns = ["da_price"]
    return da_price.sort_index().resample("15min").ffill()

def _load_rebap(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d.%m.%Y %H:%M", utc=True).dt.tz_localize(None)
    out = pd.DataFrame({"rebap": df["rebap"].astype(str).str.replace(",", ".").astype(float)}, index=ts)
    out.index.name = "DateTime"
    return out

def _load_id1(xlsx_path: Path, years) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, usecols=["TimeStamp UTC linksgestempelt", "id1"])
    df = df.rename(columns={"TimeStamp UTC linksgestempelt": "DateTime", "id1": "id1_price"})
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d.%m.%Y %H:%M", utc=True).dt.tz_localize(None)
    df = df.set_index("DateTime").sort_index()
    df.index = df.index.round("1min")
    return df[df.index.year.isin(years)]

def build_data_from_sources(
    rebap_csv: Path,
    id1_xlsx: Path,
    years = YEARS_DEFAULT,
) -> pd.DataFrame:
    """
    Zieht alle Quellen, vereinheitlicht und gibt den finalen DataFrame zurück.
    Speichern erfolgt separat (in data_store.py).
    """
    eng = _engine()
    spec = fetch_spec(eng)

    # Erzeugungsprognosen (DA/ID) & Ist
    ee_da = fetch_entsoe_data(eng, spec,
        file_name="DayAheadGenerationForecastForWindAndSolar_14.1.D",
        map_codes=["DE", "DE_TransnetBW"]
    ).pivot_table(index="DateTime", columns=["MapCode","ProductionType"], values="Value")
    ee_da = _flatten_columns(ee_da).add_suffix("_da").rename(columns={"DateTime_da":"DateTime"})

    ee_id = fetch_entsoe_data(eng, spec,
        file_name="CurrentGenerationForecastForWindAndSolar_14.1.D",
        map_codes=["DE", "DE_TransnetBW"]
    ).pivot_table(index="DateTime", columns=["MapCode","ProductionType"], values="Value")
    ee_id = _flatten_columns(ee_id).add_suffix("_id").rename(columns={"DateTime_id":"DateTime"})

    ee_act = fetch_entsoe_data(eng, spec,
        file_name="AggregatedGenerationPerType_16.1.B_C",
        map_codes=["DE", "DE_TransnetBW"],
        production_types=["Solar", "Wind Onshore", "Wind Offshore"],
        specification=["Output"]
    ).pivot_table(index="DateTime", columns=["MapCode","ProductionType"], values="Value")
    ee_act = _flatten_columns(ee_act).add_suffix("_act").rename(columns={"DateTime_act":"DateTime"})

    # Last
    load_da = fetch_entsoe_data(eng, spec,
        file_name="DayAheadTotalLoadForecast_6.1.B",
        map_codes=["DE"]
    ).pivot_table(index="DateTime", columns="MapCode", values="Value")
    load_da = _flatten_columns(load_da).rename(columns={"DE":"DE_Load_da"})

    load_act = fetch_entsoe_data(eng, spec,
        file_name="ActualTotalLoad_6.1.A",
        map_codes=["DE"]
    ).pivot_table(index="DateTime", columns="MapCode", values="Value")
    load_act = _flatten_columns(load_act).rename(columns={"DE":"DE_Load_act"})

    # Preise
    da_price = _load_prices_da(eng, spec, years).reset_index()

    # Zusammenführen
    data = (
        ee_da.merge(ee_id, on="DateTime", how="outer")
             .merge(ee_act, on="DateTime", how="outer")
             .merge(load_da, on="DateTime", how="outer")
             .merge(load_act, on="DateTime", how="outer")
             .merge(da_price, on="DateTime", how="left")
    )
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data = data.sort_values("DateTime").set_index("DateTime")

    # reBAP / ID1
    rebap = _load_rebap(Path(rebap_csv))
    id1   = _load_id1(Path(id1_xlsx), years)
    data = data.merge(rebap, left_index=True, right_index=True, how="left")
    data = data.merge(id1, left_index=True, right_index=True, how="left")

    # Marktwerte Wind Onshore (manuell, ct/kWh → €/MWh *10)
    mw_wind_onshore_ctkwh = {
        2021: [4.645, 4.361, 3.395, 4.353, 4.134, 6.330, 6.808, 7.253, 11.754, 10.982, 14.056, 16.077],
        2022: [12.883, 10.825, 19.766, 12.703, 13.242, 19.692, 27.824, 46.092, 28.238, 12.715, 13.718, 14.164],
        2023: [8.726, 10.620, 8.515, 8.940, 8.095, 9.236, 5.445, 6.613, 8.566, 6.864, 7.653, 4.409],
        2024: [6.502, 5.335, 5.538, 4.800, 5.608, 6.356, 4.985, 6.168, 6.266, 6.822, 8.881, 7.237],
    }
    mw_series = pd.concat([
        pd.Series(vals, index=pd.date_range(f"{year}-01-01", periods=12, freq="MS"))
        for year, vals in mw_wind_onshore_ctkwh.items()
    ]).sort_index()
    data["Wind Onshore_marketvalue"] = mw_series.reindex(data.index, method="ffill").fillna(method="bfill") * 10

    # Nachbearbeitung
    data["da_price"] = data["da_price"].ffill()
    data.columns = [
        c.replace("DE_TransnetBW_", "asset_") if c.startswith("DE_TransnetBW_") else c
        for c in data.columns
    ]

    # Proxy-Skalierung (wie in deinem Original)
    for col in ["asset_Wind Onshore_da", "asset_Wind Onshore_id", "asset_Wind Onshore_act"]:
        if col in data.columns:
            data[col] = data[col] / 500

    # Residual Load
    data["DE_res_da"] = (
        data.get("DE_Load_da") -
        data.get("DE_Solar_da", 0) -
        data.get("DE_Wind Onshore_da", 0) -
        data.get("DE_Wind Offshore_da", 0)
    )
    data["DE_res_act"] = (
        data.get("DE_Load_act") -
        data.get("DE_Solar_act", 0) -
        data.get("DE_Wind Onshore_act", 0) -
        data.get("DE_Wind Offshore_act", 0)
    )

    # finale Spaltenauswahl
    columns_final = [
        "da_price", "id1_price", "rebap", "Wind Onshore_marketvalue",
        "asset_Wind Onshore_da", "asset_Wind Onshore_id", "asset_Wind Onshore_act",
        "asset_Solar_da", "asset_Solar_id", "asset_Solar_act",
        "DE_Wind Onshore_act", "DE_Wind Onshore_da", "DE_Wind Onshore_id",
        "DE_Solar_act", "DE_Solar_da", "DE_Solar_id",
        "DE_Wind Offshore_act", "DE_Wind Offshore_da", "DE_Wind Offshore_id",
        "DE_Load_da", "DE_Load_act",
        "DE_res_da", "DE_res_act"
    ]
    missing = [c for c in columns_final if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data_final = data[columns_final].copy()

    # letztes vollständiges Timestamp-Cut
    last_valid_index = data_final.dropna().index[-1]
    data_final = data_final.loc[:last_valid_index]
    return data_final
