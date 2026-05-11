# California Grid Capacity Expansion Optimization (2026)
# 
# A single-node LP capacity expansion model built with PyPSA that determines
# the minimum-cost investment portfolio to reliably serve CAISO across 8,758 hours.
#
# ER254: Electric Power Systems, UC Berkeley, Spring 2026
# Author: Saskia Fadhilah Kusnadi
# GitHub: github.com/saskiafadhilah/capacity-expansion-configuration

# RESEARCH QUESTION
# Can solar PV + short-duration BESS reliably and cost-effectively replace
# natural gas peaker plants under California's decarbonization policy targets?

# SCENARIOS
# 1. solar_bess         Solar PV + 4-hr BESS            Free optimization
# 2. solar_gas          Solar PV + Natural Gas CC        15% reserve margin
# 3. solar_gas_co2cap   Solar PV + Natural Gas CC        CO2 <= 25 MtCO2/yr (CPUC D.24-02-047)
# 4. solar_gas_rps      Solar PV + Natural Gas CC        Solar >= 60% load (SB 100, 2018)

# RESULTS SUMMARY (Current Run)
# Scenario             Avg Cost ($/MWh)   CO2 (MtCO2/yr)   Load Shedding
# solar_bess           $419               0.00             0.047%
# solar_gas            $77                81.0             0.000%
# solar_gas_co2cap     $5,267*            25.0             15.8%*
# solar_gas_rps        $13,991*           32.4             0.000%
#
# * = infeasibility signal. CO2 cap needs storage. RPS needs wind/geothermal.
#
# Key finding: no 2-technology combination achieves low cost + low emissions +
# full reliability simultaneously. Consistent with CPUC 2023 Preferred System Plan.
#
# OBJECTIVE FUNCTION
# min  Σ_g [ annuity(r,n) * CAPEX/MW + FixedOM/MW ] * P_nom_g
#    + Σ_g Σ_t [ w_t * marginal_cost_g * p_g_t ]
#
# annuity(r, n) = r / (1 - (1+r)^-n),   r=0.07, n=lifetime years
# w_t = 1.0 for all hours (full-year hourly weighting)
#
# =============================================================================
# CONSTRAINTS
# =============================================================================
# C1  Power balance (every hour t):
#       p_solar_t + p_discharge_t + p_gas_t + p_shed_t - p_charge_t = D_t
#
# C2  Solar generation upper bound:
#       0 <= p_solar_t <= CF_t * P_solar_nom
#       CF_t = CAISO solar generation / 22,380 MW nameplate in [0,1]
#
# C3  Battery SOC dynamics:
#       E_t = E_{t-1} + eta_store * p_charge_t - p_discharge_t / eta_dispatch
#       eta = sqrt(0.90) = 0.9487  (from 90% roundtrip efficiency, NREL ATB 2026)
#       E_t >= 0.10 * E_max        (min SOC 10%)
#       E_0 = E_T                  (cyclic annual boundary)
#       E_max = P_batt_nom * 4     (4-hour duration)
#
# C4  Planning reserve margin (gas scenarios):
#       P_gas_nom >= 1.15 * D_peak = 50,512 MW  (CPUC RA standard)
#
# C5  CO2 emissions cap (solar_gas_co2cap only):
#       Σ_t w_t * p_gas_t * 0.36 <= 25,000,000 tCO2/yr
#       Source: CPUC Decision 24-02-047, Feb 2024 (CAISO share of 2035 target)
#
# C6  Renewable portfolio standard (solar_gas_rps only):
#       Σ_t w_t * p_solar_t >= 0.60 * Σ_t D_t = 135 TWh
#       Source: California SB 100 (2018), 60% RPS by 2030
#
# =============================================================================
# TECHNOLOGY COSTS  (NREL ATB 2026, Market scenario)
# =============================================================================
# Technology          CAPEX ($/kW)   Fixed OM ($/kW-yr)   Marginal ($/MWh)
# Utility Solar PV    1,405.5        20                   $0
# 4-hr BESS           2,171.5        50                   $0
# Natural Gas CC      1,513.0        32.5                 $42.5
# Load Shedding VOLL  -              -                    $30,000
#
# Gas marginal = VarOM $4.5 + Fuel $27.2 (6.8 MMBtu/MWh * $4/MMBtu)
#              + Carbon $10.8 (0.36 tCO2/MWh * $30/tCO2 CA Cap-and-Trade)
#
# VOLL $30,000/MWh = blended CA commercial-industrial average
# Sources: Gorman & Callaway (2024), CAISO Price Formation WG (Jan 2025)
#
# =============================================================================
# DATA SOURCES
# =============================================================================
# Hourly demand (MW)    CAISO 2025 operational data        8,758 hours
# Hourly solar CF       CAISO solar gen / 22,380 MW        Normalized [0,1]
# Technology costs      NREL ATB 2026 Market scenario      CAPEX, O&M, OCC
# CO2 intensity         NREL ATB 2026                      0.36 tCO2/MWh NGCC
# Carbon price          CA Cap-and-Trade 2025              $30/tCO2
# CO2 policy target     CPUC Decision 24-02-047 (2024)     25 MtCO2/yr CAISO
# RPS requirement       California SB 100 (2018)           60% by 2030
# Solar nameplate       CAISO Key Statistics Nov 2025       22,380 MW
#
# =============================================================================
# REPOSITORY STRUCTURE
# =============================================================================
# capacity-expansion-configuration/
#   dataset/
#     data.yaml                      <- ALL parameters: costs, scenarios, constraints
#     raw/
#       aggregated_demand_and_solar_2025_normalized.csv
#     processed/                     <- auto-generated by data_loader.py
#       demand_8760.csv
#       solar_cf_8760.csv
#       timeseries_8760.csv
#   src/
#     data_loader.py                 <- Step 1: clean and validate CAISO data
#     optimizer.py                   <- Step 2: build PyPSA network, solve LP
#     visualizer.py                  <- Step 3: generate all report figures
#     run.py                         <- Runs full pipeline (one command)
#     lcoe_analysis.py               <- Standalone LCOE comparison by technology
#   results/                         <- auto-generated, not tracked in git
#     scenario_comparison.csv
#     lcoe_comparison.csv
#     figures/
#     networks/
#   README.md
#
# =============================================================================
# HOW TO RUN
# =============================================================================
# 1. Activate virtual environment:
#       source pypsa-env/bin/activate
#
# 2. Full pipeline (data + optimize + visualize):
#       python src/run.py
#
# 3. Common flags:
#       python src/run.py --skip-data       fastest re-run after changing data.yaml
#       python src/run.py --skip-viz        optimize only, skip figures
#       python src/run.py --only-summary    reprint results from last run
#
# 4. LCOE analysis (standalone, no PyPSA needed):
#       python src/lcoe_analysis.py
#
# =============================================================================
# MODEL LIMITATIONS
# =============================================================================
# - Single-node (copper plate): no transmission constraints
# - Single year (2025): underestimates multi-year weather variability
# - Solar only: no wind, geothermal, imports — RPS result is unrealistic
# - 4-hour BESS fixed duration: optimizer does not choose duration endogenously
# - Greenfield: no existing capacity assumed
#
# =============================================================================
# REFERENCES
# =============================================================================
# Brown et al. (2018). PyPSA. J. Open Research Software 6(1). pypsa.org
# CPUC (2024). Decision 24-02-047: 2023 Preferred System Plan.
# Gorman & Callaway (2024). Value of lost load in California. UC Berkeley.
# Mallapragada et al. (2020). Long-run value of battery storage. Applied Energy 275.
# NREL (2026). Annual Technology Baseline 2026. atb.nrel.gov
# Ruhnau & Qvist (2022). Storage requirements 100% renewable. ERL 17(4).
# Sepulveda et al. (2018). Role of firm low-carbon resources. Joule 2(11).
# State of California (2018). Senate Bill 100.