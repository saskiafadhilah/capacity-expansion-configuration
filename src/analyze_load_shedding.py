"""
analyze_load_shedding.py
========================

Identifies when load shedding occurs in gas scenarios:
    - solar_gas
    - solar_gas_co2cap
    - solar_gas_rps

It shows:
    1. Which hours have load shedding
    2. Whether it happens at night / evening / day
    3. Solar availability during those hours
    4. Demand during those hours
    5. Gas dispatch during those hours
    6. Monthly and hourly patterns

Run:
    python src/analyze_load_shedding.py

Outputs:
    results_no_prm/load_shedding_analysis/
        load_shedding_hours_{scenario}.csv
        load_shedding_by_hour_{scenario}.csv
        load_shedding_by_month_{scenario}.csv
        load_shedding_summary.csv
"""

import pandas as pd
import pypsa
import yaml
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "dataset" / "data.yaml"

# Change this if you want to analyze original PRM results instead
RESULTS_DIR_NAME = "results_no_prm"

SCENARIOS = [
    "solar_gas",
    "solar_gas_co2cap",
    "solar_gas_rps",
]


def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["paths"]["results_dir"] = RESULTS_DIR_NAME
    return config


def get_paths(config):
    results_dir = BASE_DIR / config["paths"]["results_dir"]
    networks_dir = results_dir / "networks"
    output_dir = results_dir / "load_shedding_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, networks_dir, output_dir


def classify_period(hour):
    """
    Simple time-of-day classification.
    Adjust if you want different definitions.
    """
    if 0 <= hour <= 5:
        return "night_00_05"
    elif 6 <= hour <= 9:
        return "morning_06_09"
    elif 10 <= hour <= 15:
        return "midday_10_15"
    elif 16 <= hour <= 20:
        return "evening_peak_16_20"
    else:
        return "late_evening_21_23"


def analyze_scenario(config, scenario, networks_dir, output_dir):
    network_path = networks_dir / f"{scenario}.nc"

    if not network_path.exists():
        raise FileNotFoundError(
            f"Cannot find solved network: {network_path}\n"
            f"Run optimizer first."
        )

    print("\n===================================================")
    print(f"Analyzing load shedding: {scenario}")
    print("===================================================")

    n = pypsa.Network(network_path)

    # --------------------------------------------------------
    # Extract hourly time series
    # --------------------------------------------------------
    df = pd.DataFrame(index=n.snapshots)

    # Demand
    df["demand_mw"] = n.loads_t.p_set["demand"]

    # Solar dispatch
    if "solar" in n.generators_t.p.columns:
        df["solar_mw"] = n.generators_t.p["solar"]
    else:
        df["solar_mw"] = 0.0

    # Gas dispatch
    if "natural_gas_cc" in n.generators_t.p.columns:
        df["gas_mw"] = n.generators_t.p["natural_gas_cc"]
    else:
        df["gas_mw"] = 0.0

    # Load shedding dispatch
    if "load_shedding" in n.generators_t.p.columns:
        df["load_shedding_mw"] = n.generators_t.p["load_shedding"]
    else:
        df["load_shedding_mw"] = 0.0

    # Solar availability / CF if stored in generator p_max_pu
    if "solar" in n.generators_t.p_max_pu.columns:
        df["solar_cf"] = n.generators_t.p_max_pu["solar"]
    else:
        df["solar_cf"] = None

    # Add calendar features
    df["timestamp"] = df.index
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    df["date"] = df.index.date
    df["period"] = df["hour"].apply(classify_period)

    # Residual demand before load shedding
    df["served_by_solar_gas_mw"] = df["solar_mw"] + df["gas_mw"]
    df["unserved_share_of_demand"] = df["load_shedding_mw"] / df["demand_mw"]

    # Filter only hours where load shedding occurs
    shed = df[df["load_shedding_mw"] > 1e-6].copy()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    total_shed_mwh = shed["load_shedding_mw"].sum()
    total_load_mwh = df["demand_mw"].sum()
    shed_hours = len(shed)

    if shed_hours == 0:
        print("No load shedding found.")
        return {
            "scenario": scenario,
            "total_load_shedding_mwh": 0,
            "load_shedding_hours": 0,
            "load_shedding_pct_of_load": 0,
            "peak_load_shedding_mw": 0,
            "dominant_period": "none",
            "avg_solar_cf_during_shedding": 0,
            "avg_hour_during_shedding": None,
        }

    dominant_period = shed.groupby("period")["load_shedding_mw"].sum().idxmax()

    summary = {
        "scenario": scenario,
        "total_load_shedding_mwh": total_shed_mwh,
        "load_shedding_hours": shed_hours,
        "load_shedding_pct_of_load": total_shed_mwh / total_load_mwh * 100,
        "peak_load_shedding_mw": shed["load_shedding_mw"].max(),
        "dominant_period": dominant_period,
        "avg_solar_cf_during_shedding": shed["solar_cf"].mean(),
        "avg_hour_during_shedding": shed["hour"].mean(),
    }

    print(f"Total load shedding: {total_shed_mwh:,.0f} MWh")
    print(f"Load shedding hours: {shed_hours:,} hours")
    print(f"Share of annual load: {total_shed_mwh / total_load_mwh * 100:.4f}%")
    print(f"Peak load shedding: {shed['load_shedding_mw'].max():,.0f} MW")
    print(f"Dominant period: {dominant_period}")
    print(f"Average solar CF during shedding: {shed['solar_cf'].mean():.3f}")
    print(f"Average demand during shedding: {shed['demand_mw'].mean():,.0f} MW")
    print(f"Average gas during shedding: {shed['gas_mw'].mean():,.0f} MW")
    print(f"Average solar during shedding: {shed['solar_mw'].mean():,.0f} MW")

    # --------------------------------------------------------
    # Top 20 worst hours
    # --------------------------------------------------------
    top = shed.sort_values("load_shedding_mw", ascending=False).head(20)

    print("\nTop 20 load shedding hours:")
    print(
        top[
            [
                "timestamp",
                "month",
                "hour",
                "period",
                "demand_mw",
                "solar_cf",
                "solar_mw",
                "gas_mw",
                "load_shedding_mw",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Group by hour of day
    # --------------------------------------------------------
    by_hour = (
        shed.groupby("hour")
        .agg(
            load_shedding_mwh=("load_shedding_mw", "sum"),
            load_shedding_hours=("load_shedding_mw", "count"),
            avg_solar_cf=("solar_cf", "mean"),
            avg_demand_mw=("demand_mw", "mean"),
            avg_gas_mw=("gas_mw", "mean"),
            avg_solar_mw=("solar_mw", "mean"),
        )
        .reset_index()
        .sort_values("hour")
    )

    print("\nLoad shedding by hour of day:")
    print(by_hour.to_string(index=False))

    # --------------------------------------------------------
    # Group by period
    # --------------------------------------------------------
    by_period = (
        shed.groupby("period")
        .agg(
            load_shedding_mwh=("load_shedding_mw", "sum"),
            load_shedding_hours=("load_shedding_mw", "count"),
            avg_solar_cf=("solar_cf", "mean"),
            avg_demand_mw=("demand_mw", "mean"),
        )
        .reset_index()
        .sort_values("load_shedding_mwh", ascending=False)
    )

    print("\nLoad shedding by time period:")
    print(by_period.to_string(index=False))

    # --------------------------------------------------------
    # Group by month
    # --------------------------------------------------------
    by_month = (
        shed.groupby("month")
        .agg(
            load_shedding_mwh=("load_shedding_mw", "sum"),
            load_shedding_hours=("load_shedding_mw", "count"),
            avg_solar_cf=("solar_cf", "mean"),
            avg_demand_mw=("demand_mw", "mean"),
            peak_load_shedding_mw=("load_shedding_mw", "max"),
        )
        .reset_index()
        .sort_values("month")
    )

    print("\nLoad shedding by month:")
    print(by_month.to_string(index=False))

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    shed.to_csv(output_dir / f"load_shedding_hours_{scenario}.csv", index=False)
    by_hour.to_csv(output_dir / f"load_shedding_by_hour_{scenario}.csv", index=False)
    by_month.to_csv(output_dir / f"load_shedding_by_month_{scenario}.csv", index=False)
    by_period.to_csv(output_dir / f"load_shedding_by_period_{scenario}.csv", index=False)

    print(f"\nSaved hourly shedding file:")
    print(output_dir / f"load_shedding_hours_{scenario}.csv")

    return summary

def interpret_shedding(row):
    """
    Creates a direct plain-English explanation of why load shedding occurs.
    """

    scenario = row["scenario"]
    shed_mwh = row["total_load_shedding_mwh"]
    shed_pct = row["load_shedding_pct_of_load"]
    dominant_period = row["dominant_period"]
    avg_solar_cf = row["avg_solar_cf_during_shedding"]
    avg_hour = row["avg_hour_during_shedding"]

    if shed_mwh == 0:
        return "No load shedding occurs in this scenario."

    # Time-based explanation
    if avg_solar_cf <= 0.05:
        solar_condition = "solar availability is near zero"
    elif avg_solar_cf <= 0.20:
        solar_condition = "solar availability is low"
    else:
        solar_condition = "solar is available, but not enough to meet residual demand"

    if "night" in dominant_period:
        time_explanation = "mostly during nighttime hours"
    elif "evening" in dominant_period:
        time_explanation = "mostly during evening peak hours"
    elif "morning" in dominant_period:
        time_explanation = "mostly during morning ramp hours"
    elif "midday" in dominant_period:
        time_explanation = "mostly during midday hours"
    else:
        time_explanation = f"mostly during {dominant_period}"

    # Scenario-specific explanation
    if scenario == "solar_gas":
        if shed_pct < 0.01:
            cause = (
                "This is very small economic load shedding. Without the PRM constraint, "
                "the optimizer builds slightly less gas capacity and accepts a tiny amount "
                "of unserved load because building extra gas for only a few rare hours is "
                "more expensive than paying VOLL."
            )
        else:
            cause = (
                "The optimizer is relying on load shedding because gas capacity is not high "
                "enough to meet all residual demand hours."
            )

    elif scenario == "solar_gas_co2cap":
        cause = (
            "This is mainly caused by the binding CO2 cap. Gas cannot generate more without "
            "violating the emissions limit, while solar cannot cover many low-solar or night hours. "
            "Because this scenario has no battery or other clean firm resource, the model sheds load."
        )

    elif scenario == "solar_gas_rps":
        if shed_pct < 0.01:
            cause = (
                "This is very small residual load shedding. The RPS constraint forces high annual "
                "solar generation, but in a few low-solar peak hours the optimizer accepts tiny "
                "unserved energy instead of building extra rarely used gas capacity."
            )
        else:
            cause = (
                "The RPS constraint changes the generation mix, and remaining residual demand is "
                "not fully covered by gas in some hours."
            )

    else:
        cause = "Load shedding occurs because available generation is insufficient in some hours."

    return (
        f"Load shedding occurs {time_explanation}, around average hour {avg_hour:.1f}, when "
        f"{solar_condition} with average solar CF {avg_solar_cf:.3f}. {cause}"
    )

def main():
    config = load_config()
    _, networks_dir, output_dir = get_paths(config)

    summaries = []

    for scenario in SCENARIOS:
        summary = analyze_scenario(config, scenario, networks_dir, output_dir)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    # Add direct interpretation column
    summary_df["direct_answer"] = summary_df.apply(interpret_shedding, axis=1)

    # Reorder columns so the answer is easy to read
    summary_df = summary_df[
        [
            "scenario",
            "total_load_shedding_mwh",
            "load_shedding_hours",
            "load_shedding_pct_of_load",
            "peak_load_shedding_mw",
            "dominant_period",
            "avg_solar_cf_during_shedding",
            "avg_hour_during_shedding",
            "direct_answer",
        ]
    ]

    summary_path = output_dir / "load_shedding_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n===================================================")
    print("DIRECT LOAD SHEDDING ANSWER")
    print("===================================================")

    for _, row in summary_df.iterrows():
        print(f"\nScenario: {row['scenario']}")
        print(f"Total load shedding: {row['total_load_shedding_mwh']:,.0f} MWh/year")
        print(f"Load shedding share: {row['load_shedding_pct_of_load']:.4f}% of annual load")
        print(f"Load shedding hours: {row['load_shedding_hours']:,.0f} hours")
        print(f"Dominant period: {row['dominant_period']}")
        print(f"Average solar CF during shedding: {row['avg_solar_cf_during_shedding']:.3f}")
        print(f"Average hour during shedding: {row['avg_hour_during_shedding']:.1f}")
        print(f"Answer: {row['direct_answer']}")

    print("\n===================================================")
    print("Final load shedding summary table")
    print("===================================================")
    print(summary_df.to_string(index=False))

    print(f"\nSaved: {summary_path}")

if __name__ == "__main__":
    main()