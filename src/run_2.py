"""
run_2.py — California Capacity Expansion Model WITHOUT PRM
==========================================================

Runs only the five main scenarios:
    1. solar_bess
    2. solar_bess_8h
    3. solar_gas
    4. solar_gas_co2cap
    5. solar_gas_rps

Capped Solar+BESS sensitivity is NOT run here.
Run it separately:
    python src/solarbess_sensitivity.py

This pipeline prints the same high-level result sections as the original:
    - Total System Cost
    - Optimized Capacity
    - Annual Generation
    - Load Shedding
    - Emission Metrics
"""

import argparse
import sys
import time
import pandas as pd
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


def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["paths"]["results_dir"] = NO_PRM_RESULTS_DIR
    config["scenario_settings"]["scenarios"] = MAIN_SCENARIOS.copy()
    return config


def banner(title):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def step(msg):
    print(f"\n  → {msg}")


def ok(msg):
    print(f"  ✓ {msg}")


def warn(msg):
    print(f"  ⚠  {msg}")


def run_data_loader():
    banner("STEP 1 — Data Loader")
    t0 = time.time()

    sys.path.insert(0, str(BASE_DIR / "src"))
    import importlib
    loader = importlib.import_module("data_loader")
    loader.main()

    ok(f"Data loader finished in {time.time() - t0:.1f}s")


def run_optimizer_2():
    banner("STEP 2 — Optimizer 2 — NO PRM MAIN SCENARIOS")
    t0 = time.time()

    sys.path.insert(0, str(BASE_DIR / "src"))
    import importlib
    optimizer_2 = importlib.import_module("optimizer_2")
    optimizer_2.main()

    ok(f"Optimizer 2 finished in {time.time() - t0:.1f}s")


def run_visualizer_2():
    banner("STEP 3 — Visualizer 2 — Figures + Tables")
    t0 = time.time()

    sys.path.insert(0, str(BASE_DIR / "src"))
    import importlib
    visualizer_2 = importlib.import_module("visualizer_2")
    visualizer_2.main()

    ok(f"Visualizer 2 finished in {time.time() - t0:.1f}s")


def fmt(v, decimals=1, scale=1):
    if v is None or pd.isna(v):
        return "—"
    return f"{v * scale:,.{decimals}f}"


def print_table(df, rows):
    scenarios = df["scenario"].tolist()
    print(f"  {'Metric':<45} " + "  ".join(f"{s:>18}" for s in scenarios))
    print(f"  {'-'*45} " + "  ".join(f"{'─'*18}" for _ in scenarios))

    for label, col, decimals, scale in rows:
        vals = []
        for _, r in df.iterrows():
            vals.append(f"{fmt(r.get(col, None), decimals=decimals, scale=scale):>18}")
        print(f"  {label:<45} " + "  ".join(vals))


def print_summary(config):
    banner("RESULTS SUMMARY — NO PRM MAIN SCENARIOS")

    results_dir = BASE_DIR / config["paths"]["results_dir"]
    comparison_path = results_dir / "scenario_comparison.csv"

    if not comparison_path.exists():
        warn("scenario_comparison.csv not found. Run optimizer_2 first.")
        return

    df = pd.read_csv(comparison_path)
    df = df[df["scenario"].isin(MAIN_SCENARIOS)].copy()
    df["scenario"] = pd.Categorical(df["scenario"], categories=MAIN_SCENARIOS, ordered=True)
    df = df.sort_values("scenario").reset_index(drop=True)
    df["scenario"] = df["scenario"].astype(str)

    print("\n  ── TOTAL SYSTEM COST ─────────────────────────────────")
    print_table(df, [
        ("Total system cost ($/year)", "objective_total_cost_per_year_$", 0, 1),
        ("Total system cost ($B/year)", "objective_total_cost_billion_$", 2, 1),
        ("Average system cost ($/MWh)", "average_system_cost_$_per_mwh", 1, 1),
        ("Annual load served (TWh)", "load_mwh", 1, 1e-6),
    ])

    print("\n  ── OPTIMIZED CAPACITY ────────────────────────────────")
    print_table(df, [
        ("Solar capacity (GW)", "solar_capacity_gw", 1, 1),
        ("Battery power capacity (GW)", "battery_power_gw", 1, 1),
        ("Battery energy capacity (GWh)", "battery_energy_gwh", 1, 1),
        ("Gas capacity (GW)", "gas_capacity_gw", 1, 1),
    ])

    print("\n  ── ANNUAL GENERATION ─────────────────────────────────")
    df["solar_share_pct"] = df["solar_generation_mwh"] / df["load_mwh"] * 100
    df["gas_share_pct"] = df["gas_generation_mwh"] / df["load_mwh"] * 100
    print_table(df, [
        ("Solar generation (TWh/year)", "solar_generation_twh", 1, 1),
        ("Gas generation (TWh/year)", "gas_generation_twh", 1, 1),
        ("Solar share of annual load (%)", "solar_share_pct", 1, 1),
        ("Gas share of annual load (%)", "gas_share_pct", 1, 1),
    ])

    print("\n  ── LOAD SHEDDING ─────────────────────────────────────")
    print_table(df, [
        ("Gas residual unserved load (MWh/year)", "gas_unserved_residual_load_mwh", 0, 1),
        ("Total load shedding (MWh/year)", "load_shedding_mwh", 0, 1),
        ("Load shedding (% of annual load)", "load_shedding_pct_of_load", 4, 1),
    ])

    print("\n  ── EMISSION METRICS ──────────────────────────────────")
    print_table(df, [
        ("Operational CO2 (MtCO2/year)", "operational_emissions_tco2", 2, 1e-6),
        ("Solar lifecycle CO2e (MtCO2e)", "solar_lifecycle_emissions_tco2e", 2, 1e-6),
        ("Battery lifecycle CO2e (MtCO2e)", "battery_lifecycle_emissions_tco2e", 2, 1e-6),
        ("Gas lifecycle CO2e (MtCO2e)", "gas_lifecycle_emissions_tco2e", 2, 1e-6),
        ("Total incl. lifecycle CO2e (MtCO2e)", "total_emissions_with_lca_tco2e", 2, 1e-6),
        ("Lifecycle carbon cost ($M)", "lifecycle_carbon_cost_usd", 1, 1e-6),
        ("Total carbon cost incl. LCA ($M)", "total_carbon_cost_with_lca_usd", 1, 1e-6),
    ])

    print(f"\n  Results saved to: {results_dir}")
    print("  Tables and figures are generated by visualizer_2.py.")
    print("  NOTE: No PRM. VOLL is only on gas_unserved_residual_load in gas scenarios.\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the NO-PRM main California capacity expansion pipeline."
    )
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--only-summary", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    t_start = time.time()

    banner("CALIFORNIA CAPACITY EXPANSION MODEL — NO PRM MAIN SCENARIOS")
    print(f"  Project:       {config['project']['name']}")
    print(f"  Scenarios:     {', '.join(config['scenario_settings']['scenarios'])}")
    print(f"  Discount rate: {config['project']['discount_rate'] * 100:.0f}%")
    print("  Removed:       planning reserve margin")
    print("  VOLL:          gas_unserved_residual_load only")
    print(f"  Output folder: {NO_PRM_RESULTS_DIR}")

    if args.only_summary:
        print_summary(config)
        return

    if not args.skip_data:
        run_data_loader()
    else:
        step("Skipping data loader (--skip-data)")

    run_optimizer_2()

    if not args.skip_viz:
        run_visualizer_2()
    else:
        step("Skipping visualizer (--skip-viz)")

    print_summary(config)

    banner("NO-PRM MAIN PIPELINE COMPLETE")
    print(f"  Total runtime: {time.time() - t_start:.1f}s\n")


if __name__ == "__main__":
    main()
