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
    results_dir   = BASE_DIR / config["paths"]["results_dir"]
    networks_dir  = results_dir / "networks"
    results_dir.mkdir(parents=True, exist_ok=True)
    networks_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir, results_dir, networks_dir


# ============================================================
# SINGLE NODE CAPACITY EXPANSION MODEL
# Reference: https://docs.pypsa.org/v0.29.0/examples/
#            capacity-expansion-planning-single-node.html
#
# ── OBJECTIVE FUNCTION (all scenarios) ──────────────────────
#
#   min  SUM_g [ c_g_cap * P_g_nom ]
#      + SUM_g SUM_t [ w_t * c_g_marg * p_{g,t} ]
#      + VOLL * SUM_t [ L_t ]          <- gas scenarios only
#
#   c_g_cap = annuity(r, n) * CAPEX_per_MW + Fixed_OM_per_MW_yr
#   annuity(r, n) = r / (1 - (1+r)^-n),  r = 0.07
#
# ── CONSTRAINTS BY SCENARIO ─────────────────────────────────
#
#  ALL SCENARIOS — PyPSA automatic:
#   C1  Power balance every hour t
#   C2  Solar CF bound: 0 <= p_solar_t <= CF_t * P_solar_nom
#
#  solar_bess / solar_bess_8h — additional:
#   C3  SOC dynamics:  E_t = E_{t-1} + eta*p_ch_t - p_dis_t/eta
#   C4  Min SOC:       E_t >= soc_min * P_bess * max_hours
#   C4b Max SOC:       E_t <= soc_max * P_bess * max_hours  [linopy]
#   C5  Cyclic SOC:    E_0 = E_T
#   C6  ELCC-PRM:      0.21*P_solar + 0.85*P_bess >= 1.17*D_peak  (4h)
#                      0.21*P_solar + 1.00*P_bess >= 1.17*D_peak  (8h)
#
#  solar_gas — additional:
#   C6  ELCC-PRM:      0.95*P_gas + 0.21*P_solar >= 1.17*D_peak
#   VOLL applied in objective (gas is last resort before load shedding)
#
#  solar_gas_co2cap — additional:
#   C6  ELCC-PRM (same as solar_gas)
#   C7  CO2 cap:       SUM_t(p_gas_t * 0.36) <= 25,000,000 tCO2/yr
#   VOLL applied in objective
#
#  solar_gas_rps — additional:
#   C6  ELCC-PRM (same as solar_gas)
#   C8  RPS 60%:       SUM_t(p_solar_t) >= 0.60 * SUM_t(D_t)
#   VOLL applied in objective
# ============================================================


# ============================================================
# Cost processing
# ============================================================

def annuity(r, n):
    """
    Annualization factor: a(r,n) = r / (1 - (1+r)^(-n))
    Converts one-time CAPEX into equivalent annual payment.
    r=0 returns uniform spread: 1/n.
    """
    if r == 0:
        return 1 / n
    return r / (1 - (1 + r) ** (-n))


def annualized_capital_cost_per_mw(capex_per_kw, fixed_om_per_kw_year,
                                    lifetime_years, discount_rate):
    """
    Converts $/kW costs from YAML into $/MW-year for PyPSA capital_cost.
    Formula: annuity(r,n) * CAPEX_per_MW + Fixed_OM_per_MW_yr
    CAPEX is annuitized (one-time); Fixed O&M is already annual (no annuity).
    """
    capex_per_mw       = capex_per_kw * 1000
    fixed_om_per_mw_yr = fixed_om_per_kw_year * 1000
    return annuity(discount_rate, lifetime_years) * capex_per_mw + fixed_om_per_mw_yr


# ============================================================
# Load time series
# ============================================================

def load_timeseries(config):
    processed_dir, _, _ = get_paths(config)
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
    ts["solar_cf"]  = pd.to_numeric(ts["solar_cf"],  errors="coerce")
    ts = (ts.dropna(subset=["timestamp", "demand_mw", "solar_cf"])
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .set_index("timestamp"))
    ts["solar_cf"] = ts["solar_cf"].clip(lower=0, upper=1)

    if ts["demand_mw"].max() <= 1:
        raise ValueError(
            "Demand looks normalized. Use actual Demand in MW, not Demand_Normalized."
        )

    print("\nLoaded time series")
    print("------------------")
    print(f"Rows:          {len(ts)}")
    print(f"Start:         {ts.index.min()}")
    print(f"End:           {ts.index.max()}")
    print(f"Peak demand:   {ts['demand_mw'].max():,.0f} MW")
    print(f"Mean demand:   {ts['demand_mw'].mean():,.0f} MW")
    print(f"Max solar CF:  {ts['solar_cf'].max():.4f}")
    print(f"Mean solar CF: {ts['solar_cf'].mean():.4f}")
    return ts


# ============================================================
# Network initialization
# ============================================================

def initialize_network(ts, config):
    """
    Single-bus PyPSA network (copper-plate model).
    All generators connect to one bus — no transmission constraints.
    Each snapshot = 1 hour, weighting = 1.0.
    """
    n = pypsa.Network()
    n.add("Bus", "California", carrier="AC")
    n.set_snapshots(ts.index)
    if config["scenario_settings"]["full_year_hourly"]:
        n.snapshot_weightings.loc[:, :] = 1.0
    return n


def add_carriers(n, config):
    """
    Carrier co2_emissions = direct operational intensity in tCO2/MWh.
    Solar and battery = 0. Gas CC = 0.36 tCO2/MWh (NREL ATB 2026).
    """
    tech = config["technologies"]
    n.add("Carrier", "solar",         co2_emissions=0,   color="gold")
    n.add("Carrier", "battery",       co2_emissions=0,   color="yellowgreen")
    n.add("Carrier", "load_shedding", co2_emissions=0,   color="black")
    n.add("Carrier", "natural_gas_cc",
          co2_emissions=tech["natural_gas_cc"]["co2_intensity_t_per_mwh"],
          color="indianred")


def add_load(n, ts):
    """Demand as fixed load time series in actual MW."""
    n.add("Load", "demand", bus="California", p_set=ts["demand_mw"])


# ============================================================
# Technology adders
# ============================================================

def add_solar(n, ts, config):
    """
    CONSTRAINT C2 enforced here via p_max_pu:
        0 <= p_solar_t <= CF_t * P_solar_nom   for all t

    CF_t = CAISO solar generation / 22,380 MW nameplate (2025 data).
    marginal_cost = $0/MWh (no fuel, no variable O&M).
    Source: NREL ATB 2026 Utility PV, Market scenario.
    CAPEX: $1,405.5/kW. Fixed O&M: $20/kW-yr. Lifetime: 30 yr.
    """
    tech          = config["technologies"]["solar"]
    discount_rate = config["project"]["discount_rate"]
    capital_cost  = annualized_capital_cost_per_mw(
        tech["capex_per_kw"], tech["fixed_om_per_kw_year"],
        tech["lifetime_years"], discount_rate)

    n.add("Generator", "solar",
          bus="California", carrier="solar",
          p_max_pu=ts["solar_cf"],
          capital_cost=capital_cost,
          marginal_cost=tech["variable_om_per_mwh"],
          p_nom_extendable=True)


def add_natural_gas(n, config):
    """
    Dispatchable gas CC. NO p_nom_min set here — reserve margin is
    enforced via C6 ELCC-weighted PRM in extra_functionality instead.

    marginal_cost = $42.5/MWh:
        Variable O&M: $4.5/MWh
        Fuel:         $27.2/MWh  (6.8 MMBtu/MWh * $4.00/MMBtu)
        Carbon:       $10.8/MWh  (0.36 tCO2/MWh * $30/tCO2 CA C&T)

    efficiency=1.0: marginal_cost already includes fuel, avoids double-count.
    Source: NREL ATB 2026 Natural Gas CC, Market scenario.
    CAPEX: $1,513/kW. Fixed O&M: $32.5/kW-yr. Lifetime: 30 yr.
    """
    tech          = config["technologies"]["natural_gas_cc"]
    discount_rate = config["project"]["discount_rate"]
    capital_cost  = annualized_capital_cost_per_mw(
        tech["capex_per_kw"], tech["fixed_om_per_kw_year"],
        tech["lifetime_years"], discount_rate)

    n.add("Generator", "natural_gas_cc",
          bus="California", carrier="natural_gas_cc",
          capital_cost=capital_cost,
          marginal_cost=tech["marginal_cost_per_mwh"],
          efficiency=1.0,
          p_nom_extendable=True)


def add_battery(n, config):
    """
    4-hour BESS as PyPSA StorageUnit.

    CONSTRAINTS enforced here:
        C3 SOC dynamics (PyPSA auto):
            E_t = E_{t-1} + eta*p_charge_t - p_discharge_t/eta
            eta = sqrt(0.90) = 0.9487

        C4 Min SOC (state_of_charge_min, PyPSA treats as fraction for extendable):
            E_t >= 0.10 * P_bess * max_hours = 0.10 * P_bess * 4h

        C5 Cyclic SOC (cyclic_state_of_charge=True):
            E_0 = E_T  (annual closure condition)

    NOTE: max_hours = 4 is the PHYSICAL duration, not the usable window.
    C4b (SOC max 90%) is enforced separately via make_soc_max_constraint()
    in extra_functionality, because PyPSA has no state_of_charge_max param.

    Source: NREL ATB 2026 Utility-Scale Battery Storage, Market scenario.
    CAPEX: $2,171.5/kW. Fixed O&M: $50/kW-yr. Lifetime: 15 yr.
    """
    tech          = config["technologies"]["bess"]
    discount_rate = config["project"]["discount_rate"]
    capital_cost  = annualized_capital_cost_per_mw(
        tech["capex_per_kw"], tech["fixed_om_per_kw_year"],
        tech["lifetime_years"], discount_rate)
    eta = np.sqrt(tech["roundtrip_efficiency"])

    n.add("StorageUnit", "battery",
          bus="California", carrier="battery",
          max_hours=tech["duration_hours"],       # physical duration = 4h
          capital_cost=capital_cost,
          marginal_cost=tech["variable_om_per_mwh"] + tech["degradation_cost_per_mwh_throughput"],
          efficiency_store=eta,
          efficiency_dispatch=eta,
          p_nom_extendable=True,
          cyclic_state_of_charge=tech["cyclic_state_of_charge"],
          state_of_charge_min=tech["min_state_of_charge"])   # C4: 10% floor


def add_battery_custom(n, config, tech_key):
    """
    Adds BESS using named tech block (e.g. 'bess_8h').
    Same constraints C3/C4/C5 as add_battery() — duration differs.
    C4b (SOC max) enforced via make_soc_max_constraint() in extra_functionality.
    """
    tech          = config["technologies"][tech_key]
    discount_rate = config["project"]["discount_rate"]
    capital_cost  = annualized_capital_cost_per_mw(
        tech["capex_per_kw"], tech["fixed_om_per_kw_year"],
        tech["lifetime_years"], discount_rate)
    eta = np.sqrt(tech["roundtrip_efficiency"])

    n.add("StorageUnit", "battery",
          bus="California", carrier="battery",
          max_hours=tech["duration_hours"],       # physical duration = 8h
          capital_cost=capital_cost,
          marginal_cost=tech["variable_om_per_mwh"] + tech["degradation_cost_per_mwh_throughput"],
          efficiency_store=eta,
          efficiency_dispatch=eta,
          p_nom_extendable=True,
          cyclic_state_of_charge=tech["cyclic_state_of_charge"],
          state_of_charge_min=tech["min_state_of_charge"])   # C4: 5% floor for 8h


def add_load_shedding(n, config):
    """
    VOLL penalty generator — applied to GAS scenarios only.

    Physical rationale:
        In gas scenarios, gas is the last dispatchable resource. When demand
        exceeds solar + gas capacity, L_t MW of load goes unserved at cost VOLL.
        This adds VOLL * SUM_t(L_t) to the objective function, making the
        optimizer weigh building more gas vs. accepting load shedding.

        In BESS scenarios, the battery physically handles all residual demand.
        No VOLL generator is needed — if the optimizer builds enough solar+battery
        to always meet demand, the LP is always feasible without it.

    Power balance with VOLL (gas scenarios):
        G_solar_t + G_gas_t + L_t = D_t,   L_t >= 0

    VOLL = $30,000/MWh (blended CA commercial-industrial, Gorman & Callaway 2024).
    """
    n.add("Generator", "load_shedding",
          bus="California", carrier="load_shedding",
          capital_cost=0,
          marginal_cost=config["scenario_settings"]["value_of_lost_load_per_mwh"],
          p_nom_extendable=True)


# ============================================================
# CONSTRAINT C4b — Battery SOC Maximum (90% / 95%)
# ============================================================

def make_soc_max_constraint(config, tech_key="bess"):
    """
    CONSTRAINT C4b: Maximum battery state of charge

    Formulation:
        E_t <= soc_max * max_hours * P_bess_nom   for all t

    Where:
        soc_max   = 0.90 for 4h BESS (from data.yaml bess.max_state_of_charge)
                  = 0.95 for 8h BESS (from data.yaml bess_8h.max_state_of_charge)
        max_hours = physical duration (4h or 8h)
        P_bess_nom = installed battery power capacity (MW) — decision variable

    Combined with C4 (state_of_charge_min), the full SOC window is:
        4h BESS: 0.10 * P_nom * 4h <= E_t <= 0.90 * P_nom * 4h  (80% usable)
        8h BESS: 0.05 * P_nom * 8h <= E_t <= 0.95 * P_nom * 8h  (90% usable)

    Why max_hours is NOT reduced to usable hours:
        Setting max_hours = 3.2h (usable) incorrectly represents the physical
        battery — the ELCC calculation and capital cost both reference the
        physical 4h duration. The SOC window must be enforced as a separate
        constraint while keeping max_hours at the physical duration.

    Implementation: linopy constraint via extra_functionality, because PyPSA
    has no state_of_charge_max parameter for extendable StorageUnits.
    The constraint is linear: E_t <= constant * P_nom (no variable product).
    """
    soc_max = config["technologies"][tech_key]["max_state_of_charge"]

    def extra_functionality(n, snapshots):
        if "battery" not in n.storage_units.index:
            return
        max_h  = n.storage_units.at["battery", "max_hours"]   # 4 or 8
        e_t    = n.model["StorageUnit-state_of_charge"].sel(name="battery")
        p_nom  = n.model["StorageUnit-p_nom"].sel(name="battery")
        # E_t <= soc_max * max_h * p_nom  for every snapshot
        n.model.add_constraints(
            e_t <= soc_max * max_h * p_nom,
            name="soc_max_constraint"
        )

    return extra_functionality


# ============================================================
# CONSTRAINT C6 — ELCC-Weighted Planning Reserve Margin
# ============================================================

def make_prm_elcc_constraint(config, ts):
    """
    CONSTRAINT C6: ELCC-Weighted Planning Reserve Margin

    Formulation:
        SUM_g [ ELCC_g * P_g_nom ] >= (1 + PRM) * D_peak

    Expanded by scenario:
        solar_gas / co2cap / rps:
            0.95 * P_gas + 0.21 * P_solar >= 1.17 * 43,923 = 51,390 MW

        solar_bess (4h):
            0.21 * P_solar + 0.85 * P_bess >= 1.17 * 43,923 = 51,390 MW

        solar_bess_8h:
            0.21 * P_solar + 1.00 * P_bess >= 1.17 * 43,923 = 51,390 MW

    Parameters:
        D_peak  = 43,923 MW   2025 CAISO peak demand (from timeseries)
        PRM     = 0.17        CPUC System RA, 17% effective 2024 (R.23-10-011)
        Required = 43,923 * 1.17 = 51,390 MW firm capacity

        ELCC_gas   = 0.95  CAISO RA qualifying capacity (CPUC R.23-10-011)
        ELCC_solar = 0.21  CAISO 2026 median (Pham et al., 2024, Fig. 4 panel 1)
        ELCC_bess4 = 0.85  CAISO 2026 4-hr battery (Pham et al., 2024, Fig. 4 panel 4)
        ELCC_bess8 = 1.00  CAISO 2026 8-hr battery (Pham et al., 2024, Fig. 4 panel 5)

    Derivation:
        Step 1: CPUC requires Total firm capacity >= (1 + PRM) * D_peak
        Step 2: Firm_g = ELCC_g * P_g_nom  (Pham et al., 2024, p.9)
        Step 3: SUM_g [ELCC_g * P_g_nom] >= (1 + PRM) * D_peak
        Step 4: Substitute CAISO values -> equation above

    References:
        Pham, A., Cole, W., & Gagnon, P. (2024). NREL/TP-7A40-89587.
        Ho, J. et al. (2021). ReEDS Model Documentation. NREL/TP-6A20-78195.
        CPUC Resource Adequacy Program, R.23-10-011, D.24-06-004, D.24-12-003.
    """
    prm              = config["scenario_settings"]["planning_reserve_margin"]  # 0.17
    peak_demand_mw   = ts["demand_mw"].max()
    required_firm_mw = peak_demand_mw * (1 + prm)

    elcc       = config["scenario_settings"]["elcc"]
    elcc_gas   = elcc["natural_gas_cc"]   # 0.95
    elcc_solar = elcc["solar"]            # 0.21
    elcc_bess4 = elcc["bess_4h"]          # 0.85
    elcc_bess8 = elcc["bess_8h"]          # 1.00

    def extra_functionality(n, snapshots):
        p_solar = n.model["Generator-p_nom"].sel(name="solar")
        lhs     = elcc_solar * p_solar

        # Gas contribution (gas scenarios)
        if "natural_gas_cc" in n.generators.index:
            p_gas = n.model["Generator-p_nom"].sel(name="natural_gas_cc")
            lhs   = lhs + elcc_gas * p_gas

        # Battery contribution (BESS scenarios)
        # Select ELCC based on actual physical duration
        if "battery" in n.storage_units.index:
            p_bess    = n.model["StorageUnit-p_nom"].sel(name="battery")
            bess_hrs  = n.storage_units.at["battery", "max_hours"]
            elcc_bess = elcc_bess8 if bess_hrs >= 8 else elcc_bess4
            lhs       = lhs + elcc_bess * p_bess

        n.model.add_constraints(
            lhs >= required_firm_mw,
            name="prm_elcc_constraint"
        )

    return extra_functionality


# ============================================================
# CONSTRAINT C8 — Renewable Portfolio Standard
# ============================================================

def make_rps_constraint(config, ts):
    """
    CONSTRAINT C8: Renewable Portfolio Standard (solar_gas_rps only)

    Formulation:
        SUM_t [ w_t * p_solar_t ] >= 0.60 * SUM_t [ D_t ]

    Numbers:
        Annual demand ~ 225 TWh
        Required solar generation >= 0.60 * 225 = 135 TWh/yr

    Policy: California SB 100 (2018) — 60% renewable electricity by 2030.
    Solar is the only renewable in this model, so the RPS forces solar to
    provide at least 60% of annual energy.

    Physical implication:
        At solar CF ~ 30%, meeting 135 TWh requires ~51 GW minimum solar.
        In practice, the optimizer builds much more because solar must
        overbuild to hit annual totals when no storage or wind is available.
        This reveals the structural gap in a solar+gas only system trying
        to meet RPS without additional zero-carbon firm resources.
    """
    min_fraction   = config["constraints"]["min_renewable_fraction"]
    total_load_mwh = ts["demand_mw"].sum()

    def extra_functionality(n, snapshots):
        solar_p    = n.model["Generator-p"].sel(name="solar")
        weightings = n.snapshot_weightings.generators
        lhs        = (solar_p * weightings).sum()
        rhs        = min_fraction * total_load_mwh
        n.model.add_constraints(lhs >= rhs, name="rps_constraint")

    return extra_functionality


# ============================================================
# Build network per scenario
# ============================================================

def build_network_for_scenario(config, scenario):
    """
    Assembles the PyPSA network for a given scenario.
    Returns (n, ts) — ts is needed for ELCC-PRM and RPS constraints.

    Constraint map:
    ──────────────────────────────────────────────────────────
    solar_bess:
        C1  Power balance (PyPSA auto)
            G_solar_t + G_bess_t = D_t
        C2  Solar CF bound (p_max_pu)
            0 <= p_solar_t <= CF_t * P_solar_nom
        C3  SOC dynamics (PyPSA StorageUnit auto)
            E_t = E_{t-1} + eta*p_ch_t - p_dis_t/eta, eta=sqrt(0.90)
        C4  Min SOC (state_of_charge_min=0.10)
            E_t >= 0.10 * P_bess * 4h
        C4b Max SOC (extra_functionality linopy)
            E_t <= 0.90 * P_bess * 4h
        C5  Cyclic SOC (cyclic_state_of_charge=True)
            E_0 = E_T
        C6  ELCC-PRM (extra_functionality linopy)
            0.21*P_solar + 0.85*P_bess >= 51,390 MW
        NO VOLL — battery is the physical backup

    solar_bess_8h:
        C1 C2 C3 C4 C4b C5 — same as solar_bess, duration=8h
        C6  ELCC-PRM
            0.21*P_solar + 1.00*P_bess >= 51,390 MW
        NO VOLL

    solar_gas:
        C1  Power balance (PyPSA auto)
            G_solar_t + G_gas_t + L_t = D_t,  L_t >= 0
        C2  Solar CF bound (p_max_pu)
        C6  ELCC-PRM (extra_functionality linopy)
            0.95*P_gas + 0.21*P_solar >= 51,390 MW
        VOLL = $30,000/MWh on L_t (load_shedding generator)

    solar_gas_co2cap:
        C1 C2 C6 VOLL — same as solar_gas
        C7  CO2 cap (GlobalConstraint, added before optimize())
            SUM_t(p_gas_t * 0.36) <= 25,000,000 tCO2/yr
            Source: CPUC Decision 24-02-047 (Feb 2024), 2035 target

    solar_gas_rps:
        C1 C2 C6 VOLL — same as solar_gas
        C8  RPS 60% (extra_functionality linopy)
            SUM_t(p_solar_t) >= 0.60 * 225 TWh
            Source: California SB 100 (2018)
    ──────────────────────────────────────────────────────────
    """
    ts = load_timeseries(config)
    n  = initialize_network(ts, config)
    add_carriers(n, config)
    add_load(n, ts)
    add_solar(n, ts, config)

    if scenario == "solar_bess":
        add_battery(n, config)
        # VOLL NOT added — battery handles all residual demand

    elif scenario == "solar_bess_8h":
        add_battery_custom(n, config, "bess_8h")
        # VOLL NOT added — battery handles all residual demand

    elif scenario == "solar_gas":
        add_natural_gas(n, config)
        add_load_shedding(n, config)   # VOLL on gas scenarios only

    elif scenario == "solar_gas_co2cap":
        add_natural_gas(n, config)
        add_load_shedding(n, config)   # VOLL
        # C7 CO2 cap — must be added BEFORE n.optimize() builds the LP
        co2_cap = config["constraints"]["co2_cap_tco2_per_year"]
        n.add("GlobalConstraint", "co2_cap",
              sense="<=",
              constant=co2_cap,
              carrier_attribute="co2_emissions")

    elif scenario == "solar_gas_rps":
        add_natural_gas(n, config)
        add_load_shedding(n, config)   # VOLL
        # C8 RPS — added via extra_functionality in solve_scenario()

    else:
        raise ValueError(
            f"Unknown scenario: '{scenario}'. "
            f"Valid: solar_bess, solar_bess_8h, solar_gas, "
            f"solar_gas_co2cap, solar_gas_rps"
        )

    return n, ts


# ============================================================
# Model evaluation
# ============================================================

def weighted_generation_by_generator(n):
    """Annual MWh per generator: snapshot_weightings @ dispatch."""
    return n.snapshot_weightings.generators @ n.generators_t.p


def calculate_emissions(n):
    """
    Annual operational CO2 in tCO2.
    Dispatch (MW) * CO2 intensity (tCO2/MWh) * weighting (h) = tCO2.
    """
    emissions = n.generators_t.p.multiply(
        n.generators.carrier.map(n.carriers.co2_emissions), axis=1)
    return n.snapshot_weightings.generators @ emissions.sum(axis=1)


def extract_summary(n, scenario, config):
    """
    Extracts all results from the solved network.
    Handles missing load_shedding gracefully (BESS scenarios have none).
    """
    load_mwh       = n.snapshot_weightings.generators @ n.loads_t.p_set.sum(axis=1)
    generation_mwh = weighted_generation_by_generator(n)

    solar_capacity_mw    = n.generators.at["solar", "p_nom_opt"]
    solar_generation_mwh = generation_mwh.get("solar", 0)

    gas_capacity_mw    = 0
    gas_generation_mwh = 0
    if "natural_gas_cc" in n.generators.index:
        gas_capacity_mw    = n.generators.at["natural_gas_cc", "p_nom_opt"]
        gas_generation_mwh = generation_mwh.get("natural_gas_cc", 0)

    battery_power_mw       = 0
    battery_energy_mwh     = 0
    battery_throughput_mwh = 0
    if "battery" in n.storage_units.index:
        battery_power_mw       = n.storage_units.at["battery", "p_nom_opt"]
        battery_energy_mwh     = battery_power_mw * n.storage_units.at["battery", "max_hours"]
        battery_throughput_mwh = (
            n.snapshot_weightings.stores @ n.storage_units_t.p["battery"].abs())

    # load_shedding only exists in gas scenarios; defaults to 0 for BESS
    load_shedding_mwh     = generation_mwh.get("load_shedding", 0)
    solar_available_mwh   = (n.snapshot_weightings.generators
                             @ (n.generators_t.p_max_pu["solar"] * solar_capacity_mw))
    solar_curtailment_mwh = max(solar_available_mwh - solar_generation_mwh, 0)
    emissions_tco2        = calculate_emissions(n)

    solar_cf = (solar_generation_mwh / (solar_capacity_mw * 8760)
                if solar_capacity_mw > 0 else 0)
    gas_cf   = (gas_generation_mwh   / (gas_capacity_mw   * 8760)
                if gas_capacity_mw   > 0 else 0)

    # Life-cycle emissions: LCA_factor (gCO2e/kWh) * generation (MWh) / 1000 = tCO2e
    s_lca = config["technologies"]["solar"]["lifecycle_emissions_gco2e_per_kwh"]
    g_lca = config["technologies"]["natural_gas_cc"]["lifecycle_emissions_gco2e_per_kwh"]
    b_lca = config["technologies"]["bess"]["lifecycle_emissions_gco2e_per_kwh"]

    solar_lca_tco2e   = solar_generation_mwh  * s_lca / 1000
    gas_lca_tco2e     = gas_generation_mwh    * g_lca / 1000
    battery_lca_tco2e = battery_throughput_mwh * b_lca / 1000
    total_lca_tco2e   = solar_lca_tco2e + gas_lca_tco2e + battery_lca_tco2e
    total_with_lca    = emissions_tco2 + total_lca_tco2e

    scc               = config["emissions_accounting"]["social_cost_of_carbon_per_tco2"]
    lca_carbon_cost   = total_lca_tco2e * scc
    total_carbon_cost = total_with_lca  * scc

    return pd.Series({
        "scenario":                           scenario,
        "objective_total_cost_per_year_$":    n.objective,
        "objective_total_cost_billion_$":     n.objective / 1e9,
        "load_mwh":                           load_mwh,
        "average_system_cost_$_per_mwh":      n.objective / load_mwh,

        "solar_capacity_mw":                  solar_capacity_mw,
        "solar_capacity_gw":                  solar_capacity_mw / 1000,
        "solar_generation_mwh":               solar_generation_mwh,
        "solar_generation_twh":               solar_generation_mwh / 1e6,
        "solar_capacity_factor":              solar_cf,
        "solar_curtailment_mwh":              solar_curtailment_mwh,

        "battery_power_mw":                   battery_power_mw,
        "battery_power_gw":                   battery_power_mw / 1000,
        "battery_energy_mwh":                 battery_energy_mwh,
        "battery_energy_gwh":                 battery_energy_mwh / 1000,
        "battery_throughput_mwh":             battery_throughput_mwh,

        "gas_capacity_mw":                    gas_capacity_mw,
        "gas_capacity_gw":                    gas_capacity_mw / 1000,
        "gas_generation_mwh":                 gas_generation_mwh,
        "gas_generation_twh":                 gas_generation_mwh / 1e6,
        "gas_capacity_factor":                gas_cf,

        "operational_emissions_tco2":         emissions_tco2,
        "solar_lifecycle_emissions_tco2e":    solar_lca_tco2e,
        "battery_lifecycle_emissions_tco2e":  battery_lca_tco2e,
        "gas_lifecycle_emissions_tco2e":      gas_lca_tco2e,
        "total_lifecycle_emissions_tco2e":    total_lca_tco2e,
        "total_emissions_with_lca_tco2e":     total_with_lca,
        "lifecycle_carbon_cost_usd":          lca_carbon_cost,
        "total_carbon_cost_with_lca_usd":     total_carbon_cost,

        "load_shedding_mwh":                  load_shedding_mwh,
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
        n.storage_units_t.state_of_charge.to_csv(
            results_dir / f"storage_soc_{scenario}.csv")
    n.export_to_netcdf(networks_dir / f"{scenario}.nc")


# ============================================================
# Solve
# ============================================================

def solve_scenario(config, scenario):
    """
    Builds network, combines all constraints, solves with HiGHS.

    All extra_functionality callbacks are merged into one function
    because PyPSA only accepts a single extra_functionality argument.

    Constraint activation per scenario:
        solar_bess:       C1 C2 C3 C4 C4b C5 C6
        solar_bess_8h:    C1 C2 C3 C4 C4b C5 C6
        solar_gas:        C1 C2 C6 + VOLL
        solar_gas_co2cap: C1 C2 C6 C7 + VOLL
        solar_gas_rps:    C1 C2 C6 C8 + VOLL
    """
    print("\n===================================================")
    print(f"  Solving: {scenario}")
    print("===================================================")

    n, ts = build_network_for_scenario(config, scenario)

    # ── Assemble extra_functionality callbacks ───────────────

    extra_funcs = []

    # C6 — ELCC-weighted PRM: applies to ALL scenarios
    extra_funcs.append(make_prm_elcc_constraint(config, ts))

    # C4b — SOC max: applies to BESS scenarios only
    if scenario == "solar_bess":
        extra_funcs.append(make_soc_max_constraint(config, tech_key="bess"))
    elif scenario == "solar_bess_8h":
        extra_funcs.append(make_soc_max_constraint(config, tech_key="bess_8h"))

    # C8 — RPS: applies to solar_gas_rps only
    if scenario == "solar_gas_rps":
        extra_funcs.append(make_rps_constraint(config, ts))

    # Combine all into one callback (PyPSA accepts only one)
    def combined_extra_functionality(n, snapshots):
        for func in extra_funcs:
            func(n, snapshots)

    # ── Print active constraints ─────────────────────────────
    print(f"\n  Active constraints:")
    print(f"    C1  Power balance (PyPSA auto)")
    print(f"    C2  Solar CF bound (p_max_pu)")
    if "battery" in [c for c in getattr(n, "storage_units", pd.DataFrame()).index]:
        print(f"    C3  SOC dynamics (PyPSA StorageUnit auto)")
        print(f"    C4  Min SOC {config['technologies']['bess' if scenario == 'solar_bess' else 'bess_8h']['min_state_of_charge']*100:.0f}% (state_of_charge_min)")
        print(f"    C4b Max SOC {config['technologies']['bess' if scenario == 'solar_bess' else 'bess_8h']['max_state_of_charge']*100:.0f}% (extra_functionality linopy)")
        print(f"    C5  Cyclic SOC (cyclic_state_of_charge)")
    print(f"    C6  ELCC-weighted PRM 17% (extra_functionality linopy)")
    if scenario == "solar_gas_co2cap":
        print(f"    C7  CO2 cap <= {config['constraints']['co2_cap_tco2_per_year']/1e6:.0f} MtCO2/yr (GlobalConstraint)")
    if scenario == "solar_gas_rps":
        print(f"    C8  RPS >= {config['constraints']['min_renewable_fraction']*100:.0f}% of load (extra_functionality linopy)")
    if scenario in ["solar_gas", "solar_gas_co2cap", "solar_gas_rps"]:
        voll = config["scenario_settings"]["value_of_lost_load_per_mwh"]
        print(f"    VOLL ${voll:,}/MWh on unserved load L_t (gas scenarios only)")
    print()

    status = n.optimize(
        solver_name="highs",
        extra_functionality=combined_extra_functionality,
    )
    print(f"\n  Solver status: {status}")

    summary = extract_summary(n, scenario, config)
    print("\n  Summary:")
    print(summary.to_string())

    save_scenario_outputs(n, summary, scenario, config)
    return summary


# ============================================================
# Main
# ============================================================

def main():
    config = load_config()
    _, results_dir, _ = get_paths(config)

    # Validate VOLL
    voll = config["scenario_settings"]["value_of_lost_load_per_mwh"]
    if voll < 10000:
        print(f"\n  WARNING: VOLL = ${voll:,}/MWh is below $10,000/MWh.")
        print(f"  Gas scenarios will show excessive load shedding.")
        print(f"  Set value_of_lost_load_per_mwh: 30000 in data.yaml\n")

    # Validate ELCC section
    if "elcc" not in config["scenario_settings"]:
        raise KeyError(
            "Missing 'elcc' section in scenario_settings in data.yaml.\n"
            "Add:\n"
            "  elcc:\n"
            "    natural_gas_cc: 0.95\n"
            "    solar: 0.21\n"
            "    bess_4h: 0.85\n"
            "    bess_8h: 1.00\n"
        )

    summaries = []
    for scenario in config["scenario_settings"]["scenarios"]:
        summaries.append(solve_scenario(config, scenario))

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(results_dir / "scenario_comparison.csv", index=False)

    print("\n===================================================")
    print("  Final comparison")
    print("===================================================")
    print(comparison.T.to_string())
    print(f"\n  Saved: {results_dir / 'scenario_comparison.csv'}")


if __name__ == "__main__":
    main()
