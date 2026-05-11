import pandas as pd
import numpy as np
import pypsa
import yaml
from pathlib import Path


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "dataset" / "data.yaml"


# ============================================================
# Load configuration
# ============================================================

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def get_paths(config):
    processed_dir = BASE_DIR / config["paths"]["processed_data_dir"]
    results_dir = BASE_DIR / config["paths"]["results_dir"]
    networks_dir = results_dir / "networks"

    results_dir.mkdir(parents=True, exist_ok=True)
    networks_dir.mkdir(parents=True, exist_ok=True)

    return processed_dir, results_dir, networks_dir

# ============================================================
# SINGLE NODE MODEL - PyPSA tutorial style, with custom cost processing and time series loading.
# https://docs.pypsa.org/v0.29.0/examples/capacity-expansion-planning-single-node.html
# ============================================================

# ============================================================
# Cost processing 
# ============================================================

def annuity(r, n):
    """
    Annualization factor:
    a(r,n) = r / (1 - (1+r)^(-n))
    """
    if r == 0:
        return 1 / n

    return r / (1 - (1 + r) ** (-n))


def annualized_capital_cost_per_mw(
    capex_per_kw,
    fixed_om_per_kw_year,
    lifetime_years,
    discount_rate,
):
    """
    PyPSA uses capital_cost in $/MW-year.

    Your YAML has:
    CAPEX = $/kW
    Fixed O&M = $/kW-year

    This converts both into annualized $/MW-year.
    """
    capex_per_mw = capex_per_kw * 1000
    fixed_om_per_mw_year = fixed_om_per_kw_year * 1000

    return annuity(discount_rate, lifetime_years) * capex_per_mw + fixed_om_per_mw_year


# ============================================================
# Load time series
# ============================================================

def load_timeseries(config):
    processed_dir, _, _ = get_paths(config)
    path = processed_dir / config["processed_files"]["merged_timeseries"]

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run data loader first:\n"
            "python src/data_loader.py"
        )

    ts = pd.read_csv(path)
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])

    required_cols = ["timestamp", "demand_mw", "solar_cf"]
    missing_cols = [c for c in required_cols if c not in ts.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    ts["demand_mw"] = pd.to_numeric(ts["demand_mw"], errors="coerce")
    ts["solar_cf"] = pd.to_numeric(ts["solar_cf"], errors="coerce")

    ts = ts.dropna(subset=["timestamp", "demand_mw", "solar_cf"])
    ts = ts.drop_duplicates(subset=["timestamp"])
    ts = ts.sort_values("timestamp")
    ts = ts.set_index("timestamp")

    ts["solar_cf"] = ts["solar_cf"].clip(lower=0, upper=1)

    if ts["demand_mw"].max() <= 1:
        raise ValueError(
            "Demand looks normalized. Use actual Demand in MW, not Demand_Normalized."
        )

    print("\nLoaded time series")
    print("------------------")
    print("Rows:", len(ts))
    print("Start:", ts.index.min())
    print("End:", ts.index.max())
    print("Peak demand MW:", ts["demand_mw"].max())
    print("Mean demand MW:", ts["demand_mw"].mean())
    print("Max solar CF:", ts["solar_cf"].max())
    print("Mean solar CF:", ts["solar_cf"].mean())

    return ts


# ============================================================
# Build network - PyPSA single-node example
# ============================================================

def initialize_network(ts, config):
    """
    Model Initialisation:
    - empty PyPSA network
    - single bus
    - hourly snapshots
    - snapshot weighting
    """
    n = pypsa.Network()

    # Single bus -  tutorial's "electricity" bus
    n.add("Bus", "California", carrier="AC")

    # Set snapshots
    n.set_snapshots(ts.index)

    # Full year hourly data: each snapshot represents 1 hour.
    # This corresponds to w_t in the PyPSA formulation.
    if config["scenario_settings"]["full_year_hourly"]:
        n.snapshot_weightings.loc[:, :] = 1.0

    return n


def add_carriers(n, config):
    """
    Add technology carriers.
    Carriers are labels used for grouping, emissions, and plotting.
    """
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
    """
    Add demand time series.
    PyPSA Load.p_set should be in MW.
    """
    n.add(
        "Load",
        "demand",
        bus="California",
        p_set=ts["demand_mw"],
    )


def add_solar(n, ts, config):
    """
    Add utility-scale solar.
    This matches the tutorial's variable renewable generator:
    p_max_pu = hourly capacity factor
    p_nom_extendable = True
    """
    tech = config["technologies"]["solar"]
    discount_rate = config["project"]["discount_rate"]

    capital_cost = annualized_capital_cost_per_mw(
        capex_per_kw=tech["capex_per_kw"],
        fixed_om_per_kw_year=tech["fixed_om_per_kw_year"],
        lifetime_years=tech["lifetime_years"],
        discount_rate=discount_rate,
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
    """
    Add natural gas combined cycle generator.
    This is like the tutorial's OCGT/dispatchable generator.
    """
    tech = config["technologies"]["natural_gas_cc"]
    discount_rate = config["project"]["discount_rate"]

    capital_cost = annualized_capital_cost_per_mw(
        capex_per_kw=tech["capex_per_kw"],
        fixed_om_per_kw_year=tech["fixed_om_per_kw_year"],
        lifetime_years=tech["lifetime_years"],
        discount_rate=discount_rate,
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
    """
    Add BESS as StorageUnit.
    This follows the PyPSA tutorial pattern:
    StorageUnit + max_hours + p_nom_extendable + cyclic SOC.
    """
    tech = config["technologies"]["bess"]
    discount_rate = config["project"]["discount_rate"]

    capital_cost = annualized_capital_cost_per_mw(
        capex_per_kw=tech["capex_per_kw"],
        fixed_om_per_kw_year=tech["fixed_om_per_kw_year"],
        lifetime_years=tech["lifetime_years"],
        discount_rate=discount_rate,
    )

    one_way_efficiency = np.sqrt(tech["roundtrip_efficiency"])

    n.add(
        "StorageUnit",
        "battery",
        bus="California",
        carrier="battery",
        max_hours=tech["duration_hours"],
        capital_cost=capital_cost,
        marginal_cost=tech["variable_om_per_mwh"]
        + tech["degradation_cost_per_mwh_throughput"],
        efficiency_store=one_way_efficiency,
        efficiency_dispatch=one_way_efficiency,
        p_nom_extendable=True,
        cyclic_state_of_charge=tech["cyclic_state_of_charge"],        
        state_of_charge_min=tech["min_state_of_charge"],  
    )


def add_load_shedding(n, config):
    """
    Add expensive backup generator to represent Value of Lost Load.
    This keeps the model feasible but strongly penalizes unmet demand.
    """
    if not config["scenario_settings"]["use_load_shedding"]:
        return

    n.add(
        "Generator",
        "load_shedding",
        bus="California",
        carrier="load_shedding",
        capital_cost=0,
        marginal_cost=config["scenario_settings"]["value_of_lost_load_per_mwh"],
        p_nom_extendable=True,
    )


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
    n = initialize_network(ts, config)
    add_carriers(n, config)
    add_load(n, ts)
    add_solar(n, ts, config)

    if scenario == "solar_bess":
        add_battery(n, config)

    elif scenario == "solar_gas":
        add_natural_gas(n, config)
        reserve_margin = config["scenario_settings"].get("planning_reserve_margin", 0.15)
        peak_demand_mw = ts["demand_mw"].max()
        n.generators.at["natural_gas_cc", "p_nom_min"] = peak_demand_mw * (1 + reserve_margin)
        
    elif scenario == "solar_gas_co2cap":
        add_natural_gas(n, config)
        reserve_margin = config["scenario_settings"].get("planning_reserve_margin", 0.15)
        peak_demand_mw = ts["demand_mw"].max()
        n.generators.at["natural_gas_cc", "p_nom_min"] = peak_demand_mw * (1 + reserve_margin)
        # CO2 cap added here — before optimize() builds the model
        co2_cap = config["constraints"]["co2_cap_tco2_per_year"]
        n.add(
            "GlobalConstraint",
            "co2_cap",
            sense="<=",
            constant=co2_cap,
            carrier_attribute="co2_emissions",
        )

    elif scenario == "solar_gas_rps":
        add_natural_gas(n, config)
        reserve_margin = config["scenario_settings"].get("planning_reserve_margin", 0.15)
        peak_demand_mw = ts["demand_mw"].max()
        n.generators.at["natural_gas_cc", "p_nom_min"] = peak_demand_mw * (1 + reserve_margin)
    
    

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    add_load_shedding(n, config)
    return n, ts   # <-- return ts too, needed for RPS constraint

# ============================================================
# Model evaluation, like PyPSA tutorial
# ============================================================

def weighted_generation_by_generator(n):
    """
    MWh generated by generator.
    Similar to tutorial:
    n.snapshot_weightings.generators @ n.generators_t.p
    """
    return n.snapshot_weightings.generators @ n.generators_t.p


def weighted_storage_dispatch(n):
    """
    MWh discharged/charged by storage.
    Positive = discharge, negative = charge.
    """
    if n.storage_units.empty:
        return pd.Series(dtype=float)

    return n.snapshot_weightings.stores @ n.storage_units_t.p


def calculate_emissions(n):
    """
    Calculate emissions from generator dispatch.
    Since gas co2_intensity is already tCO2/MWh electricity in your YAML,
    this uses direct electricity output.
    """
    emissions = n.generators_t.p.multiply(
        n.generators.carrier.map(n.carriers.co2_emissions), axis=1
    )

    return n.snapshot_weightings.generators @ emissions.sum(axis=1)


def extract_summary(n, scenario, config):
    load_mwh = n.snapshot_weightings.generators @ n.loads_t.p_set.sum(axis=1)

    generation_mwh = weighted_generation_by_generator(n)

    solar_capacity_mw = n.generators.at["solar", "p_nom_opt"]
    solar_generation_mwh = generation_mwh.get("solar", 0)

    gas_capacity_mw = 0
    gas_generation_mwh = 0

    if "natural_gas_cc" in n.generators.index:
        gas_capacity_mw = n.generators.at["natural_gas_cc", "p_nom_opt"]
        gas_generation_mwh = generation_mwh.get("natural_gas_cc", 0)

    battery_power_mw = 0
    battery_energy_mwh = 0
    battery_throughput_mwh = 0

    if "battery" in n.storage_units.index:
        battery_power_mw = n.storage_units.at["battery", "p_nom_opt"]
        battery_energy_mwh = battery_power_mw * n.storage_units.at["battery", "max_hours"]
        battery_throughput_mwh = (
            n.snapshot_weightings.stores @ n.storage_units_t.p["battery"].abs()
        )

    load_shedding_mwh = generation_mwh.get("load_shedding", 0)

    solar_available_mwh = (
        n.snapshot_weightings.generators
        @ (n.generators_t.p_max_pu["solar"] * solar_capacity_mw)
    )
    solar_curtailment_mwh = max(solar_available_mwh - solar_generation_mwh, 0)

    emissions_tco2 = calculate_emissions(n)

    solar_capacity_factor = (
        solar_generation_mwh / (solar_capacity_mw * 8760)
        if solar_capacity_mw > 0
        else 0
    )

    gas_capacity_factor = (
        gas_generation_mwh / (gas_capacity_mw * 8760)
        if gas_capacity_mw > 0
        else 0
    )

    # --------------------------------------------------------
    # Life-cycle emissions accounting
    # LCA factor = gCO2e/kWh, generation = MWh
    # MWh * g/kWh / 1000 = tCO2e
    # --------------------------------------------------------
    solar_lca_factor = config["technologies"]["solar"]["lifecycle_emissions_gco2e_per_kwh"]
    gas_lca_factor = config["technologies"]["natural_gas_cc"]["lifecycle_emissions_gco2e_per_kwh"]
    bess_lca_factor = config["technologies"]["bess"]["lifecycle_emissions_gco2e_per_kwh"]

    solar_lifecycle_emissions_tco2e = solar_generation_mwh * solar_lca_factor / 1000
    gas_lifecycle_emissions_tco2e = gas_generation_mwh * gas_lca_factor / 1000
    battery_lifecycle_emissions_tco2e = battery_throughput_mwh * bess_lca_factor / 1000

    total_lifecycle_emissions_tco2e = (
        solar_lifecycle_emissions_tco2e
        + gas_lifecycle_emissions_tco2e
        + battery_lifecycle_emissions_tco2e
    )

    total_emissions_with_lca_tco2e = emissions_tco2 + total_lifecycle_emissions_tco2e

    social_cost_carbon = config["emissions_accounting"]["social_cost_of_carbon_per_tco2"]

    lifecycle_carbon_cost_usd = total_lifecycle_emissions_tco2e * social_cost_carbon
    total_carbon_cost_with_lca_usd = total_emissions_with_lca_tco2e * social_cost_carbon

    return pd.Series(
        {
            "scenario": scenario,
            "objective_total_cost_per_year_$": n.objective,
            "objective_total_cost_billion_$": n.objective / 1e9,
            "load_mwh": load_mwh,
            "average_system_cost_$_per_mwh": n.objective / load_mwh,

            "solar_capacity_mw": solar_capacity_mw,
            "solar_capacity_gw": solar_capacity_mw / 1000,
            "solar_generation_mwh": solar_generation_mwh,
            "solar_generation_twh": solar_generation_mwh / 1e6,
            "solar_capacity_factor": solar_capacity_factor,
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
            "gas_capacity_factor": gas_capacity_factor,

            "operational_emissions_tco2": emissions_tco2,
            "solar_lifecycle_emissions_tco2e": solar_lifecycle_emissions_tco2e,
            "battery_lifecycle_emissions_tco2e": battery_lifecycle_emissions_tco2e,
            "gas_lifecycle_emissions_tco2e": gas_lifecycle_emissions_tco2e,
            "total_lifecycle_emissions_tco2e": total_lifecycle_emissions_tco2e,
            "total_emissions_with_lca_tco2e": total_emissions_with_lca_tco2e,
            "lifecycle_carbon_cost_usd": lifecycle_carbon_cost_usd,
            "total_carbon_cost_with_lca_usd": total_carbon_cost_with_lca_usd,

            "load_shedding_mwh": load_shedding_mwh,
        }
    )

def save_scenario_outputs(n, summary, scenario, config):
    _, results_dir, networks_dir = get_paths(config)

    summary.to_csv(results_dir / f"summary_{scenario}.csv")

    n.generators.to_csv(results_dir / f"generators_{scenario}.csv")
    n.generators_t.p.to_csv(results_dir / f"generator_dispatch_{scenario}.csv")
    n.loads_t.p_set.to_csv(results_dir / f"load_{scenario}.csv")

    if not n.storage_units.empty:
        n.storage_units.to_csv(results_dir / f"storage_units_{scenario}.csv")
        n.storage_units_t.p.to_csv(results_dir / f"storage_dispatch_{scenario}.csv")
        n.storage_units_t.state_of_charge.to_csv(
            results_dir / f"storage_soc_{scenario}.csv"
        )

    # Save solved PyPSA network for visualizer.py
    n.export_to_netcdf(networks_dir / f"{scenario}.nc")


def solve_scenario(config, scenario):
    print("\n===================================================")
    print(f"Solving scenario: {scenario}")
    print("===================================================")

    n, ts = build_network_for_scenario(config, scenario)

    extra_func = None
    if scenario == "solar_gas_rps":
        extra_func = make_rps_constraint(config, ts)

    if extra_func:
        status = n.optimize(
            solver_name="highs",
            extra_functionality=extra_func,
        )
    else:
        status = n.optimize(solver_name="highs")

    print("Solver status:", status)

    summary = extract_summary(n, scenario, config)
    print("\nSummary:")
    print(summary)

    save_scenario_outputs(n, summary, scenario, config)
    return summary

def main():
    config = load_config()
    _, results_dir, _ = get_paths(config)

    summaries = []

    for scenario in config["scenario_settings"]["scenarios"]:
        summaries.append(solve_scenario(config, scenario))

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(results_dir / "scenario_comparison.csv", index=False)

    print("\n===================================================")
    print("Final comparison")
    print("===================================================")
    print(comparison)

    print("\nSaved:")
    print(results_dir / "scenario_comparison.csv")


if __name__ == "__main__":
    main()