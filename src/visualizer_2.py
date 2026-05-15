"""
visualizer_2.py — NO-PRM Main Scenario Visualizer
=================================================

Fixes:
  1. Excludes capped sensitivity cases from all main-scenario plots:
        solar_bess_capped_2x
        solar_bess_capped_5x
     even if they still exist inside scenario_comparison.csv.

  2. Cost-emissions tradeoff uses lifecycle-inclusive emissions:
        total_emissions_with_lca_tco2e
     instead of operational_emissions_tco2.

  3. Saves report tables without requiring the optional tabulate package.

Run:
    python src/visualizer_2.py
or:
    python src/run_2.py --skip-data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypsa
import yaml
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "dataset" / "data.yaml"
NO_PRM_RESULTS_DIR = "results_no_prm"

MAIN_SCENARIOS = [
    "solar_bess",
    "solar_bess_8h",
    "solar_gas",
    "solar_gas_co2cap",
    "solar_gas_rps",
]

SCENARIO_LABELS = {
    "solar_bess": "Solar +\nBESS",
    "solar_bess_8h": "Solar +\nBESS (8h)",
    "solar_gas": "Solar +\nNat. Gas",
    "solar_gas_co2cap": "Solar + Gas\n(CO₂ Cap)",
    "solar_gas_rps": "Solar + Gas\n(60% RPS)",
}

SCENARIO_COLORS = {
    "solar_bess": "#2ca6b8",
    "solar_bess_8h": "#55a6d9",
    "solar_gas": "#e66b6b",
    "solar_gas_co2cap": "#f0a23a",
    "solar_gas_rps": "#78c578",
}

TECH_COLORS = {
    "solar": "#f4c430",
    "battery": "#78c578",
    "gas": "#e66b6b",
    "load_shedding": "#222222",
    "lifecycle": "#b66fd3",
}

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
# Loading
# ============================================================

def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    config["paths"]["results_dir"] = NO_PRM_RESULTS_DIR
    config["scenario_settings"]["scenarios"] = MAIN_SCENARIOS.copy()
    return config


def get_paths(config):
    results_dir = BASE_DIR / config["paths"]["results_dir"]
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    networks_dir = results_dir / "networks"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    return results_dir, figures_dir, tables_dir, networks_dir


def scenario_label(s):
    return SCENARIO_LABELS.get(s, s)


def scenario_color(s):
    return SCENARIO_COLORS.get(s, "#999999")


def load_comparison(config):
    results_dir, _, _, _ = get_paths(config)
    path = results_dir / "scenario_comparison.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run optimizer_2.py first."
        )

    comparison = pd.read_csv(path)

    # CRITICAL FIX:
    # Remove capped sensitivity cases from main report plots/tables.
    comparison = comparison[comparison["scenario"].isin(MAIN_SCENARIOS)].copy()

    comparison["scenario"] = pd.Categorical(
        comparison["scenario"],
        categories=MAIN_SCENARIOS,
        ordered=True,
    )
    comparison = comparison.sort_values("scenario").reset_index(drop=True)
    comparison["scenario"] = comparison["scenario"].astype(str)

    if comparison.empty:
        raise ValueError("No main scenarios found in scenario_comparison.csv.")

    return comparison


# ============================================================
# Tables
# ============================================================

def save_metric_tables(df, tables_dir):
    df = df.copy()
    df["Scenario"] = df["scenario"].map(lambda s: SCENARIO_LABELS.get(s, s).replace("\n", " "))

    total_system_cost = pd.DataFrame({
        "Scenario": df["Scenario"],
        "Total system cost ($/year)": df["objective_total_cost_per_year_$"],
        "Total system cost ($B/year)": df["objective_total_cost_billion_$"],
        "Average system cost ($/MWh)": df["average_system_cost_$_per_mwh"],
        "Annual load served (TWh)": df["load_mwh"] / 1e6,
    })

    optimized_capacity = pd.DataFrame({
        "Scenario": df["Scenario"],
        "Solar capacity (GW)": df["solar_capacity_gw"],
        "Battery power capacity (GW)": df["battery_power_gw"],
        "Battery energy capacity (GWh)": df["battery_energy_gwh"],
        "Gas capacity (GW)": df["gas_capacity_gw"],
    })

    annual_generation = pd.DataFrame({
        "Scenario": df["Scenario"],
        "Solar generation (TWh/year)": df["solar_generation_twh"],
        "Gas generation (TWh/year)": df["gas_generation_twh"],
        "Solar share of annual load (%)": df["solar_generation_mwh"] / df["load_mwh"] * 100,
        "Gas share of annual load (%)": df["gas_generation_mwh"] / df["load_mwh"] * 100,
    })

    load_shedding = pd.DataFrame({
        "Scenario": df["Scenario"],
        "Gas residual unserved load (MWh/year)": df.get("gas_unserved_residual_load_mwh", 0),
        "Total load shedding (MWh/year)": df["load_shedding_mwh"],
        "Load shedding (% of annual load)": df["load_shedding_pct_of_load"],
    })

    emission_metrics = pd.DataFrame({
        "Scenario": df["Scenario"],
        "Operational CO₂ (MtCO₂/year)": df["operational_emissions_tco2"] / 1e6,
        "Solar lifecycle CO₂e (MtCO₂e)": df["solar_lifecycle_emissions_tco2e"] / 1e6,
        "Battery lifecycle CO₂e (MtCO₂e)": df["battery_lifecycle_emissions_tco2e"] / 1e6,
        "Gas lifecycle CO₂e (MtCO₂e)": df["gas_lifecycle_emissions_tco2e"] / 1e6,
        "Total incl. lifecycle CO₂e (MtCO₂e)": df["total_emissions_with_lca_tco2e"] / 1e6,
        "Lifecycle carbon cost ($M)": df["lifecycle_carbon_cost_usd"] / 1e6,
        "Total carbon cost incl. LCA ($M)": df["total_carbon_cost_with_lca_usd"] / 1e6,
    })

    tables = {
        "01_total_system_cost.csv": total_system_cost,
        "02_optimized_capacity.csv": optimized_capacity,
        "03_annual_generation.csv": annual_generation,
        "04_load_shedding.csv": load_shedding,
        "05_emission_metrics.csv": emission_metrics,
    }

    for filename, table in tables.items():
        table.round(4).to_csv(tables_dir / filename, index=False)

    # No tabulate dependency.
    md_path = tables_dir / "00_all_summary_tables.md"
    with open(md_path, "w") as f:
        f.write("# NO-PRM Main Scenario Summary Tables\\n\\n")
        for title, table in [
            ("Total System Cost", total_system_cost),
            ("Optimized Capacity", optimized_capacity),
            ("Annual Generation", annual_generation),
            ("Load Shedding", load_shedding),
            ("Emission Metrics", emission_metrics),
        ]:
            f.write(f"## {title}\\n\\n")
            f.write("```csv\\n")
            f.write(table.round(4).to_csv(index=False))
            f.write("```\\n\\n")

    print(f"  Saved tables to: {tables_dir}")


# ============================================================
# Plot functions
# ============================================================

def plot_average_system_cost(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [scenario_label(s) for s in df["scenario"]]
    colors = [scenario_color(s) for s in df["scenario"]]
    values = df["average_system_cost_$_per_mwh"]

    bars = ax.bar(labels, values, color=colors, width=0.65)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + values.max() * 0.015,
            f"${val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Average System Cost by Scenario")
    ax.set_ylabel("$/MWh")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_optimized_capacity(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [scenario_label(s) for s in df["scenario"]]
    x = np.arange(len(labels))
    width = 0.65

    solar = df["solar_capacity_gw"].fillna(0).values
    battery = df["battery_power_gw"].fillna(0).values
    gas = df["gas_capacity_gw"].fillna(0).values

    ax.bar(x, solar, width, label="Solar PV", color=TECH_COLORS["solar"])
    ax.bar(x, battery, width, bottom=solar, label="BESS power", color=TECH_COLORS["battery"])
    ax.bar(x, gas, width, bottom=solar + battery, label="Natural Gas CC", color=TECH_COLORS["gas"])

    ax.set_title("Optimized Installed Capacity by Scenario")
    ax.set_ylabel("Capacity (GW)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_annual_generation(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [scenario_label(s) for s in df["scenario"]]
    x = np.arange(len(labels))
    width = 0.65

    solar = df["solar_generation_twh"].fillna(0).values
    gas = df["gas_generation_twh"].fillna(0).values
    shed = df["load_shedding_mwh"].fillna(0).values / 1e6

    ax.bar(x, solar, width, label="Solar PV", color=TECH_COLORS["solar"])
    ax.bar(x, gas, width, bottom=solar, label="Natural Gas CC", color=TECH_COLORS["gas"])
    ax.bar(x, -shed, width, label="Unserved load", color=TECH_COLORS["load_shedding"], alpha=0.55)

    annual_load = df["load_mwh"].iloc[0] / 1e6
    ax.axhline(
        annual_load,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.65,
        label=f"Annual load ({annual_load:.0f} TWh)",
    )

    ax.set_title("Annual Generation by Scenario")
    ax.set_ylabel("TWh/year")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_load_shedding(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [scenario_label(s) for s in df["scenario"]]
    colors = [scenario_color(s) for s in df["scenario"]]
    values = df["load_shedding_pct_of_load"].fillna(0)

    bars = ax.bar(labels, values, color=colors, width=0.65)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values.max() * 0.02, 0.0001),
                f"{val:.3f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title("Load Shedding by Scenario")
    ax.set_ylabel("Unserved load (% of annual load)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_emissions(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [scenario_label(s) for s in df["scenario"]]
    x = np.arange(len(labels))
    width = 0.36

    operational = df["operational_emissions_tco2"].fillna(0).values / 1e6
    lifecycle = df["total_emissions_with_lca_tco2e"].fillna(0).values / 1e6

    ax.bar(
        x - width / 2,
        operational,
        width,
        label="Operational CO₂",
        color=TECH_COLORS["gas"],
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        lifecycle,
        width,
        label="Total incl. lifecycle CO₂e",
        color=TECH_COLORS["lifecycle"],
        alpha=0.85,
    )

    ax.axhline(30, color="gray", linestyle="--", linewidth=1, alpha=0.8, label="CPUC 2030 target ~30 MtCO₂")
    ax.axhline(20, color="gray", linestyle=":", linewidth=1, alpha=0.8, label="CPUC 2035 target ~20 MtCO₂")

    ax.set_title("Annual CO₂ Emissions by Scenario")
    ax.set_ylabel("MtCO₂e/year")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_cost_emissions_tradeoff_lifecycle(df, save_path):
    """
    CRITICAL FIX:
    Uses total lifecycle-inclusive emissions instead of operational emissions.
    X-axis = total_emissions_with_lca_tco2e / 1e6.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = df["total_emissions_with_lca_tco2e"].fillna(0) / 1e6
    y = df["average_system_cost_$_per_mwh"].fillna(0)
    shed = df["load_shedding_pct_of_load"].fillna(0)

    # Bubble size uses unserved energy percent, but keep minimum size visible.
    sizes = 80 + shed * 80

    for i, row in df.iterrows():
        scenario = row["scenario"]
        xi = row["total_emissions_with_lca_tco2e"] / 1e6
        yi = row["average_system_cost_$_per_mwh"]
        si = 80 + row["load_shedding_pct_of_load"] * 80

        ax.scatter(
            xi,
            yi,
            s=si,
            color=scenario_color(scenario),
            alpha=0.85,
            edgecolor="white",
            linewidth=1.0,
        )
        ax.text(
            xi,
            yi + max(y.max() * 0.015, 5),
            scenario_label(scenario).replace("\n", " "),
            fontsize=8,
            ha="center",
            color=scenario_color(scenario),
            fontweight="bold",
        )

    ax.axvline(30, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="30 MtCO₂e reference")
    ax.axvline(20, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="20 MtCO₂e reference")

    ax.set_title("Cost vs. Lifecycle-Inclusive Emissions Tradeoff")
    ax.set_xlabel("Total emissions incl. lifecycle CO₂e (MtCO₂e/year)")
    ax.set_ylabel("Average System Cost ($/MWh)")
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_dashboard(df, save_path):
    labels = [scenario_label(s) for s in df["scenario"]]
    colors = [scenario_color(s) for s in df["scenario"]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("NO-PRM Main Scenario Dashboard", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(x, df["average_system_cost_$_per_mwh"], color=colors)
    ax.set_title("Average System Cost")
    ax.set_ylabel("$/MWh")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax = axes[0, 1]
    ax.bar(x, df["total_emissions_with_lca_tco2e"] / 1e6, color=colors)
    ax.set_title("Total Lifecycle-Inclusive Emissions")
    ax.set_ylabel("MtCO₂e/year")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax = axes[1, 0]
    solar = df["solar_generation_twh"]
    gas = df["gas_generation_twh"]
    ax.bar(x, solar, color=TECH_COLORS["solar"], label="Solar")
    ax.bar(x, gas, bottom=solar, color=TECH_COLORS["gas"], label="Gas")
    ax.set_title("Annual Generation")
    ax.set_ylabel("TWh/year")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax = axes[1, 1]
    ax.bar(x, df["load_shedding_pct_of_load"], color=colors)
    ax.set_title("Load Shedding")
    ax.set_ylabel("% annual load")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# Main
# ============================================================

def main():
    config = load_config()
    results_dir, figures_dir, tables_dir, _ = get_paths(config)
    comparison = load_comparison(config)

    print("\n============================================================")
    print("  NO-PRM MAIN SCENARIO VISUALIZER")
    print("============================================================")
    print(f"  Results folder: {results_dir}")
    print(f"  Figures folder: {figures_dir}")
    print(f"  Tables folder:  {tables_dir}")
    print("  Excluding capped sensitivity cases from main plots.")
    print("  Tradeoff plot uses lifecycle-inclusive emissions.")

    print("\nGenerating report tables...")
    save_metric_tables(comparison, tables_dir)

    print("\nGenerating figures...")
    plot_average_system_cost(comparison, figures_dir / "01_average_system_cost_no_prm.png")
    plot_optimized_capacity(comparison, figures_dir / "02_optimized_capacity_no_prm.png")
    plot_annual_generation(comparison, figures_dir / "03_annual_generation_no_prm.png")
    plot_load_shedding(comparison, figures_dir / "04_load_shedding_no_prm.png")
    plot_emissions(comparison, figures_dir / "05_emission_metrics_no_prm.png")
    plot_cost_emissions_tradeoff_lifecycle(comparison, figures_dir / "06_cost_lifecycle_emissions_tradeoff_no_prm.png")
    plot_dashboard(comparison, figures_dir / "07_dashboard_no_prm.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
