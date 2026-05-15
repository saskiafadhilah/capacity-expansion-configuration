"""
sensitivity_2.py — Carbon Sensitivity WITHOUT PRM
=================================================

Outputs:
    results_no_prm/carbon_sensitivity_2_no_prm.csv
    results_no_prm/figures/15_sensitivity_capacity_no_prm_only.png
    results_no_prm/figures/16_sensitivity_cost_emissions_no_prm_only.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa
import yaml
from pathlib import Path


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "dataset" / "data.yaml"
NO_PRM_RESULTS_DIR = "results_no_prm"


# ============================================================
# SCC values to test
# ============================================================

SCC_VALUES = [0, 15, 30, 51, 75, 100, 130, 150, 190, 200]


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ============================================================
# Helpers
# ============================================================

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Save sensitivity results separately in no-PRM folder
    config["paths"]["results_dir"] = NO_PRM_RESULTS_DIR

    return config


def annuity(r, n):
    if r == 0:
        return 1 / n
    return r / (1 - (1 + r) ** (-n))


def annualized_capital_cost_per_mw(
    capex_per_kw,
    fixed_om_per_kw_year,
    lifetime_years,
    discount_rate,
):
    capex_per_mw = capex_per_kw * 1000
    fixed_om_per_mw_year = fixed_om_per_kw_year * 1000

    return (
        annuity(discount_rate, lifetime_years) * capex_per_mw
        + fixed_om_per_mw_year
    )


def load_timeseries(config):
    processed_dir = BASE_DIR / config["paths"]["processed_data_dir"]
    path = processed_dir / config["processed_files"]["merged_timeseries"]

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run data loader first:\n"
            "  python src/data_loader.py"
        )

    ts = pd.read_csv(path)
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])

    required_cols = ["timestamp", "demand_mw", "solar_cf"]
    missing = [c for c in required_cols if c not in ts.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ts["demand_mw"] = pd.to_numeric(ts["demand_mw"], errors="coerce")
    ts["solar_cf"] = pd.to_numeric(ts["solar_cf"], errors="coerce")

    ts = (
        ts.dropna(subset=["timestamp", "demand_mw", "solar_cf"])
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    ts["solar_cf"] = ts["solar_cf"].clip(lower=0, upper=1)

    if ts["demand_mw"].max() <= 1:
        raise ValueError(
            "Demand looks normalized. Use actual MW demand, not normalized demand."
        )

    print("\nLoaded time series")
    print("------------------")
    print(f"Rows:          {len(ts)}")
    print(f"Start:         {ts.index.min()}")
    print(f"End:           {ts.index.max()}")
    print(f"Peak demand:   {ts['demand_mw'].max():,.0f} MW")
    print(f"Mean demand:   {ts['demand_mw'].mean():,.0f} MW")
    print(f"Mean solar CF: {ts['solar_cf'].mean():.4f}")

    return ts


# ============================================================
# Build solar + gas network WITHOUT PRM
# ============================================================

def build_solar_gas_network_no_prm(config, ts, scc):
    """
    Builds solar + gas + load_shedding network at a given SCC.

    NO PRM:
        No ELCC-weighted planning reserve margin is added.

    Power balance:
        solar + gas + load_shedding = demand

    Solar weather constraint:
        solar_t <= solar_cf_t * solar_capacity

    Gas marginal cost changes with SCC:
        gas_marginal = variable O&M + fuel cost + SCC * CO2 intensity
    """

    solar = config["technologies"]["solar"]
    gas = config["technologies"]["natural_gas_cc"]
    r = config["project"]["discount_rate"]

    fuel_cost = gas["fuel_price_per_mmbtu"] * gas["heat_rate_mmbtu_per_mwh"]
    carbon_cost = scc * gas["co2_intensity_t_per_mwh"]
    gas_marginal_cost = gas["variable_om_per_mwh"] + fuel_cost + carbon_cost

    n = pypsa.Network()
    n.add("Bus", "California", carrier="AC")
    n.set_snapshots(ts.index)
    n.snapshot_weightings.loc[:, :] = 1.0

    # Carriers
    n.add("Carrier", "solar", co2_emissions=0)
    n.add(
        "Carrier",
        "natural_gas_cc",
        co2_emissions=gas["co2_intensity_t_per_mwh"],
    )
    n.add("Carrier", "load_shedding", co2_emissions=0)

    # Load
    n.add(
        "Load",
        "demand",
        bus="California",
        p_set=ts["demand_mw"],
    )

    # Solar
    solar_capital_cost = annualized_capital_cost_per_mw(
        solar["capex_per_kw"],
        solar["fixed_om_per_kw_year"],
        solar["lifetime_years"],
        r,
    )

    n.add(
        "Generator",
        "solar",
        bus="California",
        carrier="solar",
        p_max_pu=ts["solar_cf"],
        capital_cost=solar_capital_cost,
        marginal_cost=solar["variable_om_per_mwh"],
        p_nom_extendable=True,
    )

    # Gas
    gas_capital_cost = annualized_capital_cost_per_mw(
        gas["capex_per_kw"],
        gas["fixed_om_per_kw_year"],
        gas["lifetime_years"],
        r,
    )

    n.add(
        "Generator",
        "natural_gas_cc",
        bus="California",
        carrier="natural_gas_cc",
        capital_cost=gas_capital_cost,
        marginal_cost=gas_marginal_cost,
        efficiency=1.0,
        p_nom_extendable=True,
    )

    # Load shedding / VOLL
    voll = config["scenario_settings"]["value_of_lost_load_per_mwh"]

    n.add(
        "Generator",
        "load_shedding",
        bus="California",
        carrier="load_shedding",
        capital_cost=0,
        marginal_cost=voll,
        p_nom_extendable=True,
    )

    return n, gas_marginal_cost


# ============================================================
# Extract results
# ============================================================

def extract_results(n, scc, gas_marginal_cost):
    weights = n.snapshot_weightings.generators
    generation_mwh = weights @ n.generators_t.p

    load_mwh = float(weights @ n.loads_t.p_set.sum(axis=1))

    solar_capacity_gw = n.generators.at["solar", "p_nom_opt"] / 1000
    gas_capacity_gw = n.generators.at["natural_gas_cc", "p_nom_opt"] / 1000

    solar_generation_twh = generation_mwh.get("solar", 0) / 1e6
    gas_generation_twh = generation_mwh.get("natural_gas_cc", 0) / 1e6
    load_shedding_mwh = generation_mwh.get("load_shedding", 0)

    emissions_tco2 = float(
        weights @ n.generators_t.p.multiply(
            n.generators.carrier.map(n.carriers.co2_emissions),
            axis=1,
        ).sum(axis=1)
    )

    solar_cf = (
        solar_generation_twh * 1e6 / (solar_capacity_gw * 1000 * 8760)
        if solar_capacity_gw > 0
        else 0
    )

    gas_cf = (
        gas_generation_twh * 1e6 / (gas_capacity_gw * 1000 * 8760)
        if gas_capacity_gw > 0
        else 0
    )

    return {
        "scc_per_tco2": scc,
        "gas_marginal_cost_per_mwh": round(gas_marginal_cost, 2),

        "solar_capacity_gw": round(solar_capacity_gw, 3),
        "gas_capacity_gw": round(gas_capacity_gw, 3),

        "solar_capacity_factor_pct": round(solar_cf * 100, 2),
        "gas_capacity_factor_pct": round(gas_cf * 100, 2),

        "solar_generation_twh": round(solar_generation_twh, 3),
        "gas_generation_twh": round(gas_generation_twh, 3),

        "operational_emissions_mtco2": round(emissions_tco2 / 1e6, 3),

        "objective_total_cost_billion_usd": round(n.objective / 1e9, 3),
        "avg_system_cost_per_mwh": round(n.objective / load_mwh, 2),

        "load_shedding_mwh": round(load_shedding_mwh, 3),
        "load_shedding_pct": round(load_shedding_mwh / load_mwh * 100, 5),
    }


# ============================================================
# Run one SCC
# ============================================================

def run_without_prm(config, ts, scc):
    n, gas_marginal_cost = build_solar_gas_network_no_prm(config, ts, scc)

    status = n.optimize(solver_name="highs")

    print(f"    Solver status: {status}")

    return extract_results(n, scc, gas_marginal_cost)


# ============================================================
# Plots
# ============================================================

def plot_capacity_sensitivity(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["scc_per_tco2"],
        df["solar_capacity_gw"],
        marker="o",
        linewidth=2.2,
        label="Solar capacity",
    )

    ax.plot(
        df["scc_per_tco2"],
        df["gas_capacity_gw"],
        marker="s",
        linewidth=2.2,
        label="Gas capacity",
    )

    ax.set_title("Carbon Sensitivity WITHOUT PRM — Capacity")
    ax.set_xlabel("Social Cost of Carbon ($/tCO₂)")
    ax.set_ylabel("Optimized capacity (GW)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"  Saved: {save_path}")


def plot_cost_emissions_sensitivity(df, save_path):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(
        df["scc_per_tco2"],
        df["avg_system_cost_per_mwh"],
        marker="o",
        linewidth=2.2,
        label="Average system cost",
    )

    ax1.set_xlabel("Social Cost of Carbon ($/tCO₂)")
    ax1.set_ylabel("Average system cost ($/MWh)")

    ax2 = ax1.twinx()

    ax2.plot(
        df["scc_per_tco2"],
        df["operational_emissions_mtco2"],
        marker="s",
        linewidth=2.2,
        linestyle="--",
        label="Operational emissions",
    )

    ax2.set_ylabel("Operational emissions (MtCO₂/year)")

    ax1.set_title("Carbon Sensitivity WITHOUT PRM — Cost and Emissions")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"  Saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    config = load_config()

    results_dir = BASE_DIR / config["paths"]["results_dir"]
    figures_dir = results_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    voll = config["scenario_settings"]["value_of_lost_load_per_mwh"]

    print("\n============================================================")
    print("  Social Cost of Carbon Sensitivity — NO PRM")
    print("============================================================")
    print(f"  SCC values:  {SCC_VALUES}")
    print(f"  VOLL:        ${voll:,}/MWh")
    print("  Removed:     C6 ELCC-weighted planning reserve margin")
    print("  Scenario:    solar_gas only")
    print(f"  Output:      {results_dir}")

    ts = load_timeseries(config)

    rows = []

    for scc in SCC_VALUES:
        print(f"\n  Solving SCC = ${scc}/tCO2 WITHOUT PRM...")
        result = run_without_prm(config, ts, scc)
        rows.append(result)

    df = pd.DataFrame(rows)

    output_csv = results_dir / "carbon_sensitivity_2_no_prm.csv"
    df.to_csv(output_csv, index=False)

    print(f"\n  Saved: {output_csv}")

    plot_capacity_sensitivity(
        df,
        figures_dir / "15_sensitivity_capacity_no_prm_only.png",
    )

    plot_cost_emissions_sensitivity(
        df,
        figures_dir / "16_sensitivity_cost_emissions_no_prm_only.png",
    )

    print("\n============================================================")
    print("  NO-PRM Sensitivity Results")
    print("============================================================")

    cols = [
        "scc_per_tco2",
        "gas_marginal_cost_per_mwh",
        "solar_capacity_gw",
        "gas_capacity_gw",
        "solar_generation_twh",
        "gas_generation_twh",
        "operational_emissions_mtco2",
        "avg_system_cost_per_mwh",
        "load_shedding_pct",
    ]

    print(df[cols].to_string(index=False))

    print(f"\n  Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()