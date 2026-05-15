"""
solarbess_sensitivity.py — Capped Solar+BESS Sensitivity Only
=============================================================

This is NOT part of the five main scenarios.

Purpose:
    Tests how capped Solar+BESS performs under hourly solar/weather constraints.

Model:
    solar + battery + bess_unserved_load = demand

Capacity caps:
    solar_capacity <= current_solar_cap_mw * multiplier
    battery_power_capacity <= current_battery_power_cap_mw * multiplier

VOLL here is diagnostic only:
    It is used to measure unmet demand when Solar+BESS capacity is capped.
    It is separate from the main optimizer_2.py results.

Run:
    python src/solarbess_sensitivity.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import optimizer_2 as opt


CAP_MULTIPLIERS = [0.5, 1, 2, 3, 4, 5, 7.5, 10]

BATTERY_CASES = {
    "4h_BESS": "bess",
    "8h_BESS": "bess_8h",
}

DEFAULT_CURRENT_SOLAR_CAP_MW = 25_000
DEFAULT_CURRENT_BATTERY_CAP_MW = 17_000


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


def apply_solar_bess_caps(n, config, multiplier):
    solar_base = config["scenario_settings"].get(
        "current_solar_cap_mw",
        DEFAULT_CURRENT_SOLAR_CAP_MW,
    )
    battery_base = config["scenario_settings"].get(
        "current_battery_power_cap_mw",
        DEFAULT_CURRENT_BATTERY_CAP_MW,
    )

    solar_cap_mw = solar_base * multiplier
    battery_cap_mw = battery_base * multiplier

    n.generators.at["solar", "p_nom_max"] = solar_cap_mw

    if "battery" in n.storage_units.index:
        n.storage_units.at["battery", "p_nom_max"] = battery_cap_mw

    return solar_cap_mw, battery_cap_mw


def add_bess_unserved_load(n, config):
    n.add(
        "Generator",
        "bess_unserved_load",
        bus="California",
        carrier="load_shedding",
        capital_cost=0,
        marginal_cost=config["scenario_settings"]["value_of_lost_load_per_mwh"],
        p_nom_extendable=True,
    )


def build_capped_solar_bess_network(config, ts, multiplier, tech_key):
    n = opt.initialize_network(ts, config)
    opt.add_carriers(n, config)
    opt.add_load(n, ts)
    opt.add_solar(n, ts, config)
    opt.add_battery_custom(n, config, tech_key)

    solar_cap_mw, battery_cap_mw = apply_solar_bess_caps(n, config, multiplier)
    add_bess_unserved_load(n, config)

    return n, solar_cap_mw, battery_cap_mw


def run_one_case(config, ts, multiplier, case_name, tech_key):
    print("\n----------------------------------------------------")
    print(f"Running capped Solar+BESS sensitivity: {case_name}, {multiplier}x")
    print("----------------------------------------------------")

    n, solar_cap_mw, battery_cap_mw = build_capped_solar_bess_network(
        config,
        ts,
        multiplier,
        tech_key,
    )

    extra_func = opt.make_soc_max_constraint(config, tech_key=tech_key)

    status = n.optimize(solver_name="highs", extra_functionality=extra_func)
    print(f"Solver status: {status}")

    summary = opt.extract_summary(n, f"solarbess_{case_name}_{multiplier}x", config)

    generation_mwh = opt.weighted_generation_by_generator(n)
    load_mwh = summary["load_mwh"]
    bess_unserved_mwh = generation_mwh.get("bess_unserved_load", 0)

    duration_hours = config["technologies"][tech_key]["duration_hours"]
    battery_energy_cap_gwh = battery_cap_mw * duration_hours / 1000

    row = {
        "case": case_name,
        "battery_tech_key": tech_key,
        "duration_hours": duration_hours,
        "cap_multiplier": multiplier,

        "solar_cap_limit_gw": solar_cap_mw / 1000,
        "battery_power_cap_limit_gw": battery_cap_mw / 1000,
        "battery_energy_cap_limit_gwh": battery_energy_cap_gwh,

        "solar_capacity_gw": summary["solar_capacity_gw"],
        "battery_power_gw": summary["battery_power_gw"],
        "battery_energy_gwh": summary["battery_energy_gwh"],

        "solar_generation_twh": summary["solar_generation_twh"],
        "battery_throughput_twh": summary["battery_throughput_mwh"] / 1e6,

        "bess_unserved_load_mwh": bess_unserved_mwh,
        "bess_unserved_load_twh": bess_unserved_mwh / 1e6,
        "load_shedding_pct_of_load": bess_unserved_mwh / load_mwh * 100 if load_mwh > 0 else 0,

        "average_system_cost_per_mwh": summary["average_system_cost_$_per_mwh"],
        "objective_total_cost_billion_usd": summary["objective_total_cost_billion_$"],
    }

    return row, n


def plot_load_shedding(df, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    for case, sub in df.groupby("case"):
        ax.plot(
            sub["cap_multiplier"],
            sub["load_shedding_pct_of_load"],
            marker="o",
            linewidth=2.2,
            label=case,
        )

    ax.set_title("Capped Solar+BESS Adequacy Sensitivity")
    ax.set_xlabel("Capacity cap multiplier vs. baseline")
    ax.set_ylabel("Unserved load (% of annual demand)")
    ax.legend()

    save_path = figures_dir / "01_load_shedding_vs_cap.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_cost(df, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    for case, sub in df.groupby("case"):
        ax.plot(
            sub["cap_multiplier"],
            sub["average_system_cost_per_mwh"],
            marker="o",
            linewidth=2.2,
            label=case,
        )

    ax.set_title("Average System Cost vs. Solar+BESS Cap")
    ax.set_xlabel("Capacity cap multiplier vs. baseline")
    ax.set_ylabel("Average system cost ($/MWh)")
    ax.legend()

    save_path = figures_dir / "02_cost_vs_cap.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_capacity(df, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    for case, sub in df.groupby("case"):
        ax.plot(
            sub["cap_multiplier"],
            sub["solar_capacity_gw"],
            marker="o",
            linewidth=2.2,
            label=f"{case} solar",
        )
        ax.plot(
            sub["cap_multiplier"],
            sub["battery_power_gw"],
            marker="s",
            linestyle="--",
            linewidth=2.2,
            label=f"{case} battery power",
        )

    ax.set_title("Optimized Capacity Under Caps")
    ax.set_xlabel("Capacity cap multiplier vs. baseline")
    ax.set_ylabel("Capacity (GW)")
    ax.legend(ncol=2)

    save_path = figures_dir / "03_capacity_vs_cap.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_generation(df, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    for case, sub in df.groupby("case"):
        ax.plot(
            sub["cap_multiplier"],
            sub["solar_generation_twh"],
            marker="o",
            linewidth=2.2,
            label=f"{case} solar generation",
        )
        ax.plot(
            sub["cap_multiplier"],
            sub["bess_unserved_load_twh"],
            marker="s",
            linestyle="--",
            linewidth=2.2,
            label=f"{case} unserved load",
        )

    ax.set_title("Solar Generation and Unserved Load vs. Cap")
    ax.set_xlabel("Capacity cap multiplier vs. baseline")
    ax.set_ylabel("TWh/year")
    ax.legend(ncol=2)

    save_path = figures_dir / "04_generation_vs_cap.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    config = opt.load_config()

    # Save sensitivity outputs separately from main scenario outputs.
    results_dir = BASE_DIR / config["paths"]["results_dir"] / "solarbess_sensitivity"
    figures_dir = results_dir / "figures"
    networks_dir = results_dir / "networks"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    networks_dir.mkdir(parents=True, exist_ok=True)

    config["scenario_settings"].setdefault("current_solar_cap_mw", DEFAULT_CURRENT_SOLAR_CAP_MW)
    config["scenario_settings"].setdefault("current_battery_power_cap_mw", DEFAULT_CURRENT_BATTERY_CAP_MW)

    print("\n============================================================")
    print("  Solar+BESS Capped Weather Adequacy Sensitivity")
    print("============================================================")
    print(f"  Output: {results_dir}")
    print(f"  Multipliers: {CAP_MULTIPLIERS}")
    print(f"  Battery cases: {list(BATTERY_CASES.keys())}")
    print(
        "  Baseline caps: "
        f"solar={config['scenario_settings']['current_solar_cap_mw']/1000:.1f} GW, "
        f"battery={config['scenario_settings']['current_battery_power_cap_mw']/1000:.1f} GW"
    )
    print(f"  VOLL diagnostic: ${config['scenario_settings']['value_of_lost_load_per_mwh']:,}/MWh")

    ts = opt.load_timeseries(config)
    rows = []

    for case_name, tech_key in BATTERY_CASES.items():
        for multiplier in CAP_MULTIPLIERS:
            row, n = run_one_case(config, ts, multiplier, case_name, tech_key)
            rows.append(row)

            safe_mult = str(multiplier).replace(".", "p")
            n.export_to_netcdf(networks_dir / f"{case_name}_{safe_mult}x.nc")

    df = pd.DataFrame(rows)

    csv_path = results_dir / "solarbess_cap_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved sensitivity CSV: {csv_path}")

    plot_load_shedding(df, figures_dir)
    plot_cost(df, figures_dir)
    plot_capacity(df, figures_dir)
    plot_generation(df, figures_dir)

    print("\n============================================================")
    print("  Sensitivity Summary")
    print("============================================================")

    cols = [
        "case",
        "cap_multiplier",
        "solar_cap_limit_gw",
        "battery_power_cap_limit_gw",
        "battery_energy_cap_limit_gwh",
        "solar_capacity_gw",
        "battery_power_gw",
        "battery_energy_gwh",
        "bess_unserved_load_twh",
        "load_shedding_pct_of_load",
        "average_system_cost_per_mwh",
    ]

    print(df[cols].to_string(index=False))
    print(f"\nFigures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
