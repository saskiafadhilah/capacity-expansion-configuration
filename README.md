# CAISO Grid Capacity Expansion Optimization (2026)

This repository contains a linear programming (LP) capacity expansion model built with PyPSA. It determines the minimum-cost investment portfolio required to reliably serve the California ISO (CAISO) electrical grid for a full 8,760-hour simulation year.

## Project Scope
The model evaluates and compares two competing grid decarbonization pathways:
1. **Scenario A (Solar + BESS):** High CAPEX, near-zero OPEX grid relying on photovoltaic generation paired with Lithium-ion battery energy storage systems.
2. **Scenario B (Solar + Natural Gas CC):** Lower CAPEX, high OPEX grid utilizing firm capacity from Natural Gas Combined Cycle turbines, heavily penalized by California Cap-and-Trade carbon pricing.

## Mathematical Formulation
The objective function minimizes the total annualized system cost, subject to strict hourly power balance constraints, capacity factor limits, and inter-temporal State of Charge (SOC) battery dynamics.

* **Objective:** `min(Annualized CAPEX + Fixed O&M + Variable OPEX + Load Shedding Penalty)`
* **Reliability:** Enforced via a Value of Lost Load (VOLL) soft constraint set to $30,000/MWh, allowing the solver to organically size the system for worst-case weather events without artificial heuristics.

## Data Sources
* **Load Profiles:** CAISO 2025 hourly aggregated demand.
* **Weather Profiles:** CAISO 2025 hourly solar generation arrays.
* **Economics:** NREL Annual Technology Baseline (ATB) 2024 and EPA FLIGHT emissions data.

## Installation & Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the data cleaner: `python src/data_pipeline.py`
3. Execute the solver: `python src/pypsa_model.py`
