"""
run.py — California Capacity Expansion Model
=============================================
Runs the full pipeline in order:
  1. data_loader.py   — process raw CSV into 8760-hour timeseries
  2. optimizer.py     — solve PyPSA capacity expansion for each scenario
  3. visualizer.py    — generate all figures
  4. Summary          — print a clean human-readable results table

Usage:
    python src/run.py
    python src/run.py --skip-data      (skip data loader if already processed)
    python src/run.py --skip-viz       (skip figure generation)
    python src/run.py --only-summary   (just reprint the summary from existing results)
"""

import argparse
import sys
import time
import pandas as pd
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "dataset" / "data.yaml"

# ============================================================
# Helpers
# ============================================================

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def banner(title):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def step(msg):
    print(f"\n  → {msg}")


def ok(msg):
    print(f"  ✓ {msg}")


def warn(msg):
    print(f"  ⚠  {msg}")


# ============================================================
# Pipeline stages
# ============================================================

def run_data_loader():
    banner("STEP 1 — Data Loader")
    t0 = time.time()

    # Import and run inline so errors surface cleanly
    sys.path.insert(0, str(BASE_DIR / "src"))
    import importlib
    loader = importlib.import_module("data_loader")
    loader.main()

    ok(f"Data loader finished in {time.time() - t0:.1f}s")


def run_optimizer():
    banner("STEP 2 — Optimizer")
    t0 = time.time()

    sys.path.insert(0, str(BASE_DIR / "src"))
    import importlib
    optimizer = importlib.import_module("optimizer")
    optimizer.main()

    ok(f"Optimizer finished in {time.time() - t0:.1f}s")


def run_visualizer():
    banner("STEP 3 — Visualizer")
    t0 = time.time()

    sys.path.insert(0, str(BASE_DIR / "src"))
    import importlib
    viz = importlib.import_module("visualizer")
    viz.main()

    ok(f"Visualizer finished in {time.time() - t0:.1f}s")


# ============================================================
# Summary printer
# ============================================================

def fmt(val, decimals=1):
    """Format a number, return '—' if zero or NaN."""
    try:
        if pd.isna(val) or val == 0:
            return "—"
        return f"{val:,.{decimals}f}"
    except Exception:
        return str(val)


def print_summary(config):
    banner("RESULTS SUMMARY")

    results_dir = BASE_DIR / config["paths"]["results_dir"]
    comparison_path = results_dir / "scenario_comparison.csv"

    if not comparison_path.exists():
        warn("scenario_comparison.csv not found. Run optimizer first.")
        return

    df = pd.read_csv(comparison_path)
    scenarios = df["scenario"].tolist()

    # --------------------------------------------------------
    # Section 1: Cost
    # --------------------------------------------------------
    print("\n  ── SYSTEM COST ─────────────────────────────────────")
    print(f"  {'Metric':<40} " + "  ".join(f"{s:>15}" for s in scenarios))
    print(f"  {'-'*40} " + "  ".join(f"{'─'*15}" for _ in scenarios))

    def row(label, col, decimals=1, scale=1, suffix=""):
        vals = []
        for _, r in df.iterrows():
            v = r.get(col, None)
            if v is not None and not pd.isna(v):
                vals.append(f"{v * scale:>{15},.{decimals}f}{suffix}")
            else:
                vals.append(f"{'—':>15}")
        print(f"  {label:<40} " + "  ".join(vals))

    row("Total system cost ($/year)",          "objective_total_cost_per_year_$",  decimals=0)
    row("Total system cost ($B/year)",         "objective_total_cost_billion_$",   decimals=2)
    row("Average system cost ($/MWh)",         "average_system_cost_$_per_mwh",    decimals=1)
    row("Annual load served (TWh)",            "load_mwh",                         decimals=1, scale=1e-6)

    # --------------------------------------------------------
    # Section 2: Capacity
    # --------------------------------------------------------
    print("\n  ── OPTIMIZED CAPACITY ───────────────────────────────")
    print(f"  {'Metric':<40} " + "  ".join(f"{s:>15}" for s in scenarios))
    print(f"  {'-'*40} " + "  ".join(f"{'─'*15}" for _ in scenarios))

    row("Solar capacity (GW)",                 "solar_capacity_gw",                decimals=1)
    row("Solar capacity factor (%)",           "solar_capacity_factor",            decimals=1, scale=100)
    row("Solar curtailment (TWh)",             "solar_curtailment_mwh",            decimals=2, scale=1e-6)
    row("Battery power capacity (GW)",         "battery_power_gw",                 decimals=1)
    row("Battery energy capacity (GWh)",       "battery_energy_gwh",               decimals=1)
    row("Battery throughput (TWh)",            "battery_throughput_mwh",           decimals=2, scale=1e-6)
    row("Gas capacity (GW)",                   "gas_capacity_gw",                  decimals=1)
    row("Gas capacity factor (%)",             "gas_capacity_factor",              decimals=1, scale=100)

    # --------------------------------------------------------
    # Section 3: Generation mix
    # --------------------------------------------------------
    print("\n  ── ANNUAL GENERATION MIX ────────────────────────────")
    print(f"  {'Metric':<40} " + "  ".join(f"{s:>15}" for s in scenarios))
    print(f"  {'-'*40} " + "  ".join(f"{'─'*15}" for _ in scenarios))

    row("Solar generation (TWh)",              "solar_generation_twh",             decimals=1)
    row("Gas generation (TWh)",               "gas_generation_twh",               decimals=1)

    # Solar share of load
    for _, r in df.iterrows():
        pass  # handled below as computed column

    print(f"\n  {'Solar share of load (%)':<40} ", end="")
    shares = []
    for _, r in df.iterrows():
        load = r.get("load_mwh", None)
        solar = r.get("solar_generation_mwh", None)
        if load and solar and load > 0:
            shares.append(f"{solar / load * 100:>15.1f}")
        else:
            shares.append(f"{'—':>15}")
    print("  ".join(shares))

    # --------------------------------------------------------
    # Section 4: Emissions
    # --------------------------------------------------------
    print("\n  ── EMISSIONS ────────────────────────────────────────")
    print(f"  {'Metric':<40} " + "  ".join(f"{s:>15}" for s in scenarios))
    print(f"  {'-'*40} " + "  ".join(f"{'─'*15}" for _ in scenarios))

    row("Operational CO₂ (MtCO₂/year)",       "operational_emissions_tco2",       decimals=2, scale=1e-6)
    row("Solar lifecycle CO₂e (MtCO₂e)",      "solar_lifecycle_emissions_tco2e",  decimals=2, scale=1e-6)
    row("Battery lifecycle CO₂e (MtCO₂e)",    "battery_lifecycle_emissions_tco2e",decimals=2, scale=1e-6)
    row("Gas lifecycle CO₂e (MtCO₂e)",        "gas_lifecycle_emissions_tco2e",    decimals=2, scale=1e-6)
    row("Total incl. lifecycle CO₂e (MtCO₂e)","total_emissions_with_lca_tco2e",   decimals=2, scale=1e-6)
    row("Lifecycle carbon cost ($M)",          "lifecycle_carbon_cost_usd",        decimals=1, scale=1e-6)
    row("Total carbon cost incl. LCA ($M)",    "total_carbon_cost_with_lca_usd",   decimals=1, scale=1e-6)

    # --------------------------------------------------------
    # Section 5: Reliability
    # --------------------------------------------------------
    print("\n  ── RELIABILITY ──────────────────────────────────────")
    print(f"  {'Metric':<40} " + "  ".join(f"{s:>15}" for s in scenarios))
    print(f"  {'-'*40} " + "  ".join(f"{'─'*15}" for _ in scenarios))

    row("Load shedding (MWh/year)",            "load_shedding_mwh",                decimals=0)

    print(f"\n  {'Load shedding (% of load)':<40} ", end="")
    shed_shares = []
    for _, r in df.iterrows():
        load = r.get("load_mwh", None)
        shed = r.get("load_shedding_mwh", None)
        if load and shed is not None and not pd.isna(shed) and load > 0:
            shed_shares.append(f"{shed / load * 100:>15.3f}")
        else:
            shed_shares.append(f"{'—':>15}")
    print("  ".join(shed_shares))

    # --------------------------------------------------------
    # Section 6: Interpretation
    # --------------------------------------------------------
    print("\n  ── INTERPRETATION ───────────────────────────────────")

    costs = {r["scenario"]: r.get("average_system_cost_$_per_mwh")
             for _, r in df.iterrows()}
    costs = {k: v for k, v in costs.items() if v and not pd.isna(v)}
    if costs:
        cheapest = min(costs, key=costs.get)
        costliest = max(costs, key=costs.get)
        pct = (costs[costliest] - costs[cheapest]) / costs[costliest] * 100
        print(f"\n  Cost:        {cheapest} is cheapest at ${costs[cheapest]:,.1f}/MWh.")
        print(f"               {costliest} is most expensive — {pct:.0f}% higher.")

    emissions = {r["scenario"]: r.get("operational_emissions_tco2")
                 for _, r in df.iterrows()}
    emissions = {k: v for k, v in emissions.items()
                 if v is not None and not pd.isna(v)}
    if emissions:
        cleanest = min(emissions, key=emissions.get)
        print(f"  Emissions:   {cleanest} has lowest operational CO₂.")

    for _, r in df.iterrows():
        shed = r.get("load_shedding_mwh", 0) or 0
        load = r.get("load_mwh", 1) or 1
        if shed > 1000:
            print(f"  Reliability: {r['scenario']} — {shed:,.0f} MWh load shedding "
                  f"({shed/load*100:.3f}% of load).")
    figures_dir = BASE_DIR / config["paths"]["results_dir"] / "figures"
    print(f"\n  Figures saved to: {figures_dir}")
    print(f"  Results saved to: {results_dir}")
    print()


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full California capacity expansion pipeline."
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data_loader.py (use if processed files already exist)",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualizer.py (skip figure generation)",
    )
    parser.add_argument(
        "--only-summary",
        action="store_true",
        help="Only print summary from existing scenario_comparison.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    t_start = time.time()

    banner("CALIFORNIA CAPACITY EXPANSION MODEL")
    print(f"  Project:  {config['project']['name']}")
    print(f"  Scenarios: {', '.join(config['scenario_settings']['scenarios'])}")
    print(f"  Discount rate: {config['project']['discount_rate']*100:.0f}%")

    if args.only_summary:
        print_summary(config)
        return

    if not args.skip_data:
        run_data_loader()
    else:
        step("Skipping data loader (--skip-data)")

    run_optimizer()

    if not args.skip_viz:
        run_visualizer()
    else:
        step("Skipping visualizer (--skip-viz)")

    print_summary(config)

    banner("PIPELINE COMPLETE")
    print(f"  Total runtime: {time.time() - t_start:.1f}s\n")


if __name__ == "__main__":
    main()
