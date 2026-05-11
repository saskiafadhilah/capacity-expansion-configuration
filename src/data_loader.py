import pandas as pd
from pathlib import Path

# Project paths

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Constants

CAISO_SOLAR_NAMEPLATE_MW = 22380
# Nameplate capacity of CAISO solar generation: 22,380 MW
# Source note: CAISO Key Statistics Nov 2025


# Data loader
def load_raw_combined_file(
    input_file="aggregated_demand_and_solar_2025_normalized.csv",
):
    """
    Loads the combined raw file containing:
    - Date
    - HR
    - Solar Gen
    - Solar Gen Normalized
    - Demand
    - Demand_Normalized
    """

    path = RAW_DIR / input_file

    if not path.exists():
        raise FileNotFoundError(f"Raw input file not found: {path}")

    df = pd.read_csv(path)

    print("Raw file loaded.")
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    return df


def create_timestamp(df, date_col="Date", hour_col="HR"):
    """
    Creates hourly timestamp from Date and HR.

    Your HR column appears to be 1–24:
    HR = 1  means 00:00
    HR = 2  means 01:00
    ...
    HR = 24 means 23:00
    """

    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%Y")

    df[hour_col] = pd.to_numeric(df[hour_col], errors="coerce")

    if df[hour_col].isna().any():
        raise ValueError("Some HR values could not be converted to numbers.")

    df["timestamp"] = df[date_col] + pd.to_timedelta(df[hour_col] - 1, unit="h")

    return df


def prepare_combined_timeseries(df):
    """
    Cleans combined demand and solar data into PyPSA-ready format.

    Final output:
    - timestamp
    - demand_mw
    - solar_cf

    Important:
    - demand_mw uses actual Demand in MW.
    - solar_cf uses Solar Gen Normalized.
    """

    df = df.copy()

    df = create_timestamp(df, date_col="Date", hour_col="HR")

    df = df.rename(
        columns={
            "Demand": "demand_mw",
            "Demand_Normalized": "demand_normalized",
            "Solar Gen": "solar_generation_mw",
            "Solar Gen Normalized": "solar_cf",
        }
    )

    required_cols = [
        "timestamp",
        "demand_mw",
        "solar_generation_mw",
        "solar_cf",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")
    df["solar_generation_mw"] = pd.to_numeric(
        df["solar_generation_mw"], errors="coerce"
    )
    df["solar_cf"] = pd.to_numeric(df["solar_cf"], errors="coerce")

    df = df.dropna(subset=["timestamp", "demand_mw", "solar_cf"])

    df = df.drop_duplicates(subset=["timestamp"])

    df = df.sort_values("timestamp")

    # Safety: solar capacity factor should be between 0 and 1
    df["solar_cf"] = df["solar_cf"].clip(lower=0, upper=1)

    final_df = df[["timestamp", "demand_mw", "solar_cf"]]

    return final_df


def save_processed_outputs(df):
    """
    Saves three processed files:
    1. demand_8760.csv
    2. solar_cf_8760.csv
    3. timeseries_8760.csv
    """

    demand = df[["timestamp", "demand_mw"]]
    solar = df[["timestamp", "solar_cf"]]

    demand.to_csv(PROCESSED_DIR / "demand_8760.csv", index=False)
    solar.to_csv(PROCESSED_DIR / "solar_cf_8760.csv", index=False)
    df.to_csv(PROCESSED_DIR / "timeseries_8760.csv", index=False)

    print("\nProcessed files saved:")
    print(PROCESSED_DIR / "demand_8760.csv")
    print(PROCESSED_DIR / "solar_cf_8760.csv")
    print(PROCESSED_DIR / "timeseries_8760.csv")


def validate_timeseries(df):
    """
    Prints checks before PyPSA optimization.
    """

    print("\nValidation summary")
    print("------------------")
    print("Rows:", len(df))
    print("Start:", df["timestamp"].min())
    print("End:", df["timestamp"].max())

    print("\nDemand checks")
    print("Max demand MW:", df["demand_mw"].max())
    print("Mean demand MW:", df["demand_mw"].mean())
    print("Min demand MW:", df["demand_mw"].min())

    print("\nSolar checks")
    print("Max solar CF:", df["solar_cf"].max())
    print("Mean solar CF:", df["solar_cf"].mean())
    print("Min solar CF:", df["solar_cf"].min())

    print("\nMissing values")
    print(df.isna().sum())

    if len(df) != 8760:
        print("\nWARNING: Final dataset does not have 8760 rows.")
        print("Rows found:", len(df))
        print("Check missing timestamps, leap year, or duplicated hours.")

    if df["solar_cf"].max() > 1:
        print("\nWARNING: solar_cf is above 1. Check normalization.")

    if df["demand_mw"].max() <= 1:
        print("\nWARNING: demand_mw looks normalized. PyPSA needs actual MW demand.")


def main():
    raw = load_raw_combined_file(
        input_file="aggregated_demand_and_solar_2025_normalized.csv"
    )

    timeseries = prepare_combined_timeseries(raw)

    validate_timeseries(timeseries)

    save_processed_outputs(timeseries)


if __name__ == "__main__":
    main()