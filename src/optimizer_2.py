"""
optimizer_2.py — NO-PRM Five-Scenario Capacity Expansion Model
==============================================================

Final main model for report methodology.

Main scenarios only:
    1. solar_bess
    2. solar_bess_8h
    3. solar_gas
    4. solar_gas_co2cap
    5. solar_gas_rps

Important modeling choices:
    - NO capped Solar+BESS 2x/5x scenarios in this main optimizer.
      Those belong only in solarbess_sensitivity.py.
    - NO planning reserve margin in this main optimizer.
    - Strict Solar+BESS scenarios have no load shedding and no VOLL.
      Balance: solar + battery = demand
    - Solar+Gas scenarios have gas residual unserved load.
      Balance: solar + gas + gas_unserved_residual_load = demand
      VOLL is applied only to gas_unserved_residual_load.
"""

import pandas as pd
import numpy as np
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


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    config["paths"]["results_dir"] = NO_PRM_RESULTS_DIR
    config["scenario_settings"]["scenarios"] = MAIN_SCENARIOS.copy()
    return config


def get_paths(config):
    processed_dir = BASE_DIR / config["paths"]["processed_data_dir"]
    results_dir = BASE_DIR / config["paths"]["results_dir"]
    networks_dir = results_dir / "networks"

    results_dir.mkdir(parents=True, exist_ok=True)
    networks_dir.mkdir(parents=True, exist_ok=True)

    return processed_dir, results_dir, networks_dir


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
    return annuity(discount_rate, lifetime_years) * capex_per_mw + fixed_om_per_mw_year


def load_timeseries(config):
    processed_dir, _, _ = get_paths(config)
    path = processed_dir / config["processed_files"]["merged_timeseries"]

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run data_loader.py first:\n"
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

    # IMPORTANT:
    # Do NOT manually force solar to zero from 9 PM to 5 AM.
    # We use the observed/processed CAISO solar profile and only clip physically
    # impossible normalized values outside [0, 1].
    ts["solar_cf"] = ts["solar_cf"].clip(lower=0, upper=1)

    if ts["demand_mw"].max() <= 1:
        raise ValueError("Demand looks normalized. Use actual MW demand, not normalized demand.")

    print("\nLoaded time series")
    print("------------------")
    print(f"Rows:          {len(ts)}")
    print(f"Start:         {ts.index.min()}")
    print(f"End:           {ts.index.max()}")
    print(f"Peak demand:   {ts['demand_mw'].max():,.0f} MW")
    print(f"Mean demand:   {ts['demand_mw'].mean():,.0f} MW")
    print(f"Mean solar CF: {ts['solar_cf'].mean():.4f}")

    return ts


def initialize_network(ts, config):
    n = pypsa.Network()
    n.add("Bus", "California", carrier="AC")
    n.set_snapshots(ts.index)

    if config["scenario_settings"]["full_year_hourly"]:
        n.snapshot_weightings.loc[:, :] = 1.0

    return n


def add_carriers(n, config):
    tech = config["technologies"]

    n.add("Carrier", "solar", co2_emissions=0, color="gold")
    n.add("Carrier", "battery", co2_emissions=0, color="yellowgreen")
    n.add("Carrier", "load_shedding", co2_emissions=0, color="black")

    n.add(
        "Carrier",
        "natural_gas_cc",
        co2_emissions=tech["natural_gas_cc"]["co2_intensity_t_per_mwh"],
        color="indianred",
    )


def add_load(n, ts):
    n.add("Load", "demand", bus="California", p_set=ts["demand_mw"])


def add_solar(n, ts, config):
    tech = config["technologies"]["solar"]
    r = config["project"]["discount_rate"]

    capital_cost = annualized_capital_cost_per_mw(
        tech["capex_per_kw"],
        tech["fixed_om_per_kw_year"],
        tech["lifetime_years"],
        r,
    )

    n.add(
        "Generator",
        "solar",
        bus="California",
        carrier="solar",
        p_max_pu=ts["solar_cf"],
        capital_cost=capital_cost,
        marginal_cost=tech["variable_om_per_mwh"],
        p_nom_extendable=True,
    )


def add_natural_gas(n, config):
    tech = config["technologies"]["natural_gas_cc"]
    r = config["project"]["discount_rate"]

    capital_cost = annualized_capital_cost_per_mw(
        tech["capex_per_kw"],
        tech["fixed_om_per_kw_year"],
        tech["lifetime_years"],
        r,
    )

    n.add(
        "Generator",
        "natural_gas_cc",
        bus="California",
        carrier="natural_gas_cc",
        capital_cost=capital_cost,
        marginal_cost=tech["marginal_cost_per_mwh"],
        efficiency=1.0,
        p_nom_extendable=True,
    )


def add_battery(n, config):
    return add_battery_custom(n, config, "bess")


def add_battery_custom(n, config, tech_key):
    tech = config["technologies"][tech_key]
    r = config["project"]["discount_rate"]

    capital_cost = annualized_capital_cost_per_mw(
        tech["capex_per_kw"],
        tech["fixed_om_per_kw_year"],
        tech["lifetime_years"],
        r,
    )

    eta = np.sqrt(tech["roundtrip_efficiency"])

    n.add(
        "StorageUnit",
        "battery",
        bus="California",
        carrier="battery",
        max_hours=tech["duration_hours"],
        capital_cost=capital_cost,
        marginal_cost=tech["variable_om_per_mwh"] + tech["degradation_cost_per_mwh_throughput"],
        efficiency_store=eta,
        efficiency_dispatch=eta,
        p_nom_extendable=True,
        cyclic_state_of_charge=tech["cyclic_state_of_charge"],
        state_of_charge_min=tech["min_state_of_charge"],
    )


def add_gas_residual_load_shedding(n, config):
    """
    Adds VOLL only for gas residual shortfall in Solar+Gas scenarios.

    This is NOT a penalty on solar generation.
    This is NOT a penalty on gas generation.
    This is the penalty on unmet residual demand after solar and gas:
        solar + gas + gas_unserved_residual_load = demand
    """
    if not config["scenario_settings"].get("use_load_shedding", True):
        return

    n.add(
        "Generator",
        "gas_unserved_residual_load",
        bus="California",
        carrier="load_shedding",
        capital_cost=0,
        marginal_cost=config["scenario_settings"]["value_of_lost_load_per_mwh"],
        p_nom_extendable=True,
    )


def make_soc_max_constraint(config, tech_key="bess"):
    soc_max = config["technologies"][tech_key]["max_state_of_charge"]

    def extra_functionality(n, snapshots):
        if "battery" not in n.storage_units.index:
            return

        max_h = n.storage_units.at["battery", "max_hours"]
        e_t = n.model["StorageUnit-state_of_charge"].sel(name="battery")
        p_nom = n.model["StorageUnit-p_nom"].sel(name="battery")

        n.model.add_constraints(
            e_t <= soc_max * max_h * p_nom,
            name="soc_max_constraint",
        )

    return extra_functionality


def make_rps_constraint(config, ts):
    min_fraction = config["constraints"]["min_renewable_fraction"]
    total_load_mwh = ts["demand_mw"].sum()

    def extra_functionality(n, snapshots):
        solar_p = n.model["Generator-p"].sel(name="solar")
        weightings = n.snapshot_weightings.generators

        lhs = (solar_p * weightings).sum()
        rhs = min_fraction * total_load_mwh

        n.model.add_constraints(lhs >= rhs, name="rps_constraint")

    return extra_functionality


def build_network_for_scenario(config, scenario):
    ts = load_timeseries(config)
    return build_network_for_scenario_from_ts(config, ts, scenario)


def build_network_for_scenario_from_ts(config, ts, scenario):
    n = initialize_network(ts, config)
    add_carriers(n, config)
    add_load(n, ts)
    add_solar(n, ts, config)

    if scenario == "solar_bess":
        add_battery(n, config)

    elif scenario == "solar_bess_8h":
        add_battery_custom(n, config, "bess_8h")

    elif scenario == "solar_gas":
        add_natural_gas(n, config)
        add_gas_residual_load_shedding(n, config)

    elif scenario == "solar_gas_co2cap":
        add_natural_gas(n, config)
        add_gas_residual_load_shedding(n, config)
        n.add(
            "GlobalConstraint",
            "co2_cap",
            sense="<=",
            constant=config["constraints"]["co2_cap_tco2_per_year"],
            carrier_attribute="co2_emissions",
        )

    elif scenario == "solar_gas_rps":
        add_natural_gas(n, config)
        add_gas_residual_load_shedding(n, config)

    else:
        raise ValueError(f"Unknown scenario: {scenario}. Valid scenarios: {MAIN_SCENARIOS}")

    return n, ts


def weighted_generation_by_generator(n):
    return n.snapshot_weightings.generators @ n.generators_t.p


def calculate_emissions(n):
    emissions = n.generators_t.p.multiply(
        n.generators.carrier.map(n.carriers.co2_emissions),
        axis=1,
    )
    return n.snapshot_weightings.generators @ emissions.sum(axis=1)


def extract_summary(n, scenario, config):
    load_mwh = n.snapshot_weightings.generators @ n.loads_t.p_set.sum(axis=1)
    generation_mwh = weighted_generation_by_generator(n)

    solar_capacity_mw = n.generators.at["solar", "p_nom_opt"]
    solar_generation_mwh = generation_mwh.get("solar", 0)
    solar_available_mwh = n.snapshot_weightings.generators @ (
        n.generators_t.p_max_pu["solar"] * solar_capacity_mw
    )
    solar_curtailment_mwh = max(solar_available_mwh - solar_generation_mwh, 0)
    solar_cf = solar_generation_mwh / (solar_capacity_mw * 8760) if solar_capacity_mw > 0 else 0

    gas_capacity_mw = 0
    gas_generation_mwh = 0
    gas_cf = 0
    if "natural_gas_cc" in n.generators.index:
        gas_capacity_mw = n.generators.at["natural_gas_cc", "p_nom_opt"]
        gas_generation_mwh = generation_mwh.get("natural_gas_cc", 0)
        gas_cf = gas_generation_mwh / (gas_capacity_mw * 8760) if gas_capacity_mw > 0 else 0

    battery_power_mw = 0
    battery_energy_mwh = 0
    battery_throughput_mwh = 0
    if "battery" in n.storage_units.index:
        battery_power_mw = n.storage_units.at["battery", "p_nom_opt"]
        battery_energy_mwh = battery_power_mw * n.storage_units.at["battery", "max_hours"]
        battery_throughput_mwh = n.snapshot_weightings.stores @ n.storage_units_t.p["battery"].abs()

    gas_unserved_mwh = generation_mwh.get("gas_unserved_residual_load", 0)
    load_shedding_mwh = gas_unserved_mwh

    emissions_tco2 = calculate_emissions(n)

    s_lca = config["technologies"]["solar"]["lifecycle_emissions_gco2e_per_kwh"]
    g_lca = config["technologies"]["natural_gas_cc"]["lifecycle_emissions_gco2e_per_kwh"]
    b_lca = config["technologies"]["bess"]["lifecycle_emissions_gco2e_per_kwh"]

    solar_lca_tco2e = solar_generation_mwh * s_lca / 1000
    gas_lca_tco2e = gas_generation_mwh * g_lca / 1000
    battery_lca_tco2e = battery_throughput_mwh * b_lca / 1000
    total_lca_tco2e = solar_lca_tco2e + gas_lca_tco2e + battery_lca_tco2e
    total_with_lca = emissions_tco2 + total_lca_tco2e

    scc = config["emissions_accounting"]["social_cost_of_carbon_per_tco2"]

    return pd.Series({
        "scenario": scenario,
        "objective_total_cost_per_year_$": n.objective,
        "objective_total_cost_billion_$": n.objective / 1e9,
        "load_mwh": load_mwh,
        "average_system_cost_$_per_mwh": n.objective / load_mwh,

        "solar_capacity_mw": solar_capacity_mw,
        "solar_capacity_gw": solar_capacity_mw / 1000,
        "solar_generation_mwh": solar_generation_mwh,
        "solar_generation_twh": solar_generation_mwh / 1e6,
        "solar_capacity_factor": solar_cf,
        "solar_available_mwh": solar_available_mwh,
        "solar_curtailment_mwh": solar_curtailment_mwh,

        "battery_power_mw": battery_power_mw,
        "battery_power_gw": battery_power_mw / 1000,
        "battery_energy_mwh": battery_energy_mwh,
        "battery_energy_gwh": battery_energy_mwh / 1000,
        "battery_throughput_mwh": battery_throughput_mwh,

        "gas_capacity_mw": gas_capacity_mw,
        "gas_capacity_gw": gas_capacity_mw / 1000,
        "gas_generation_mwh": gas_generation_mwh,
        "gas_generation_twh": gas_generation_mwh / 1e6,
        "gas_capacity_factor": gas_cf,

        "operational_emissions_tco2": emissions_tco2,
        "solar_lifecycle_emissions_tco2e": solar_lca_tco2e,
        "battery_lifecycle_emissions_tco2e": battery_lca_tco2e,
        "gas_lifecycle_emissions_tco2e": gas_lca_tco2e,
        "total_lifecycle_emissions_tco2e": total_lca_tco2e,
        "total_emissions_with_lca_tco2e": total_with_lca,
        "lifecycle_carbon_cost_usd": total_lca_tco2e * scc,
        "total_carbon_cost_with_lca_usd": total_with_lca * scc,

        "gas_unserved_residual_load_mwh": gas_unserved_mwh,
        "load_shedding_mwh": load_shedding_mwh,
        "load_shedding_pct_of_load": load_shedding_mwh / load_mwh * 100 if load_mwh > 0 else 0,
    })


def save_scenario_outputs(n, summary, scenario, config):
    _, results_dir, networks_dir = get_paths(config)

    summary.to_csv(results_dir / f"summary_{scenario}.csv")
    n.generators.to_csv(results_dir / f"generators_{scenario}.csv")
    n.generators_t.p.to_csv(results_dir / f"generator_dispatch_{scenario}.csv")
    n.loads_t.p_set.to_csv(results_dir / f"load_{scenario}.csv")

    if not n.storage_units.empty:
        n.storage_units.to_csv(results_dir / f"storage_units_{scenario}.csv")
        n.storage_units_t.p.to_csv(results_dir / f"storage_dispatch_{scenario}.csv")
        n.storage_units_t.state_of_charge.to_csv(results_dir / f"storage_soc_{scenario}.csv")

    n.export_to_netcdf(networks_dir / f"{scenario}.nc")


def solve_scenario(config, scenario):
    print("\n===================================================")
    print(f"  Solving NO-PRM main scenario: {scenario}")
    print("===================================================")

    n, ts = build_network_for_scenario(config, scenario)
    extra_funcs = []

    if scenario == "solar_bess":
        extra_funcs.append(make_soc_max_constraint(config, tech_key="bess"))
    elif scenario == "solar_bess_8h":
        extra_funcs.append(make_soc_max_constraint(config, tech_key="bess_8h"))

    if scenario == "solar_gas_rps":
        extra_funcs.append(make_rps_constraint(config, ts))

    def combined_extra_functionality(n, snapshots):
        for func in extra_funcs:
            func(n, snapshots)

    print("\n  Active constraints:")
    print("    C1  Hourly power balance")
    print("    C2  Solar weather/CF bound")
    print("    C16 No planning reserve margin")

    if "battery" in n.storage_units.index:
        tech_key = "bess_8h" if scenario == "solar_bess_8h" else "bess"
        print("    C3  Battery SOC dynamics")
        print(f"    C4  Min SOC {config['technologies'][tech_key]['min_state_of_charge'] * 100:.0f}%")
        print(f"    C4b Max SOC {config['technologies'][tech_key]['max_state_of_charge'] * 100:.0f}%")
        print("    C5  Cyclic SOC")
        print(f"    C6  BESS duration = {config['technologies'][tech_key]['duration_hours']} hours")
        print("    C9  Strict Solar+BESS balance, no load shedding, no VOLL")

    if "natural_gas_cc" in n.generators.index:
        print("    C7  Natural gas dispatch bound")

    if "gas_unserved_residual_load" in n.generators.index:
        voll = config["scenario_settings"]["value_of_lost_load_per_mwh"]
        print(f"    C8  VOLL ${voll:,}/MWh on gas_unserved_residual_load only")

    if scenario == "solar_gas_co2cap":
        print(f"    C13 CO2 cap <= {config['constraints']['co2_cap_tco2_per_year'] / 1e6:.0f} MtCO2/year")

    if scenario == "solar_gas_rps":
        print(f"    C14 RPS >= {config['constraints']['min_renewable_fraction'] * 100:.0f}% of annual load")

    if extra_funcs:
        status = n.optimize(solver_name="highs", extra_functionality=combined_extra_functionality)
    else:
        status = n.optimize(solver_name="highs")

    print(f"\n  Solver status: {status}")

    summary = extract_summary(n, scenario, config)
    print("\n  Summary:")
    print(summary.to_string())

    save_scenario_outputs(n, summary, scenario, config)
    return summary


def main():
    config = load_config()
    _, results_dir, _ = get_paths(config)

    voll = config["scenario_settings"]["value_of_lost_load_per_mwh"]

    if voll <= 0:
        raise ValueError("VOLL must be greater than zero.")

    print("\n===================================================")
    print("  NO-PRM MAIN CAPACITY EXPANSION MODEL")
    print("===================================================")
    print(f"  Output folder: {results_dir}")
    print(f"  VOLL: ${voll:,}/MWh")
    print("  Removed: planning reserve margin")
    print("  Main scenarios only; capped Solar+BESS cases are in solarbess_sensitivity.py")
    print("  VOLL appears only in gas scenarios as gas_unserved_residual_load.")
    print("\n  Scenarios:")
    for s in config["scenario_settings"]["scenarios"]:
        print(f"    - {s}")

    summaries = [solve_scenario(config, scenario) for scenario in config["scenario_settings"]["scenarios"]]
    comparison = pd.DataFrame(summaries)

    comparison_path = results_dir / "scenario_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n===================================================")
    print("  FINAL COMPARISON — NO PRM MAIN SCENARIOS")
    print("===================================================")

    key_cols = [
        "scenario",
        "average_system_cost_$_per_mwh",
        "solar_capacity_gw",
        "battery_power_gw",
        "battery_energy_gwh",
        "gas_capacity_gw",
        "solar_generation_twh",
        "gas_generation_twh",
        "operational_emissions_tco2",
        "gas_unserved_residual_load_mwh",
        "load_shedding_mwh",
        "load_shedding_pct_of_load",
    ]
    existing_cols = [c for c in key_cols if c in comparison.columns]
    print(comparison[existing_cols].to_string(index=False))

    print(f"\n  Saved: {comparison_path}")


if __name__ == "__main__":
    main()
