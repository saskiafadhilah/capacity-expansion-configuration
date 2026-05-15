"""
carbon_sensitivity.py — Social Cost of Carbon Sensitivity Analysis
===================================================================
Runs the solar_gas scenario at each SCC value TWICE:
    1. With ELCC-weighted PRM (17%, CPUC R.23-10-011) — realistic
    2. Without PRM — theoretical free-optimization baseline

This produces two sets of results for direct comparison:
    - With PRM:    gas cannot go below ~32 GW even at $200/tCO2
                   (firm capacity floor from ELCC constraint)
    - Without PRM: gas can approach zero at high SCC
                   (shows pure carbon price effect)

The comparison reveals how much of the gas retirement in the
unconstrained model is policy-real vs. theoretically possible.

Run from project root:
    python src/carbon_sensitivity.py

Outputs:
    results/carbon_sensitivity_with_prm.csv
    results/carbon_sensitivity_no_prm.csv
    results/figures/15_sensitivity_capacity_with_prm.png
    results/figures/16_sensitivity_capacity_no_prm.png
    results/figures/17_sensitivity_comparison.png
    results/figures/18_sensitivity_cost_emissions.png
    results/figures/19_sensitivity_breakeven.png

References for SCC range $0-$200/tCO2:
    - $0:   No carbon pricing (theoretical baseline)
    - $30:  California Cap-and-Trade allowance price (CARB, 2025)
    - $51:  EPA IWG central estimate (2016), 3% discount rate
    - $190: EPA IWG central estimate (2023), 2.5% discount rate
    - $200: Upper bound per Rennert et al. (2022), Nature 610, 687-692

ELCC values (Pham, Cole & Gagnon, 2024, NREL/TP-7A40-89587):
    - Gas CC:   0.95 (CAISO RA qualifying capacity, CPUC R.23-10-011)
    - Solar PV: 0.21 (CAISO 2026 median, Figure 4 panel 1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa
import yaml
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "dataset" / "data.yaml"

plt.rcParams.update({
    "font.family":       "DejaVu Sans", "font.size": 11,
    "axes.titlesize":    13,            "axes.titleweight": "bold",
    "axes.spines.top":   False,         "axes.spines.right": False,
    "axes.grid":         True,          "axes.grid.axis": "y",
    "grid.alpha":        0.25,          "legend.frameon": False,
    "savefig.dpi":       200,           "savefig.bbox": "tight",
})

# ============================================================
# SCC values to test
# ============================================================

SCC_VALUES = [0, 15, 30, 51, 75, 100, 130, 150, 190, 200]

SCC_REFERENCES = {
    0:   "No carbon\npricing",
    30:  "CA C&T 2025\n(CARB, 2025)",
    51:  "EPA IWG 2016\ncentral est.",
    190: "EPA IWG 2023\ncentral est.",
    200: "Rennert et al.\n(2022), Nature",
}

# Plot colors
COLOR_SOLAR   = "#f5c518"
COLOR_GAS     = "#e05c5c"
COLOR_PRM     = "#2196a8"
COLOR_NO_PRM  = "#e8912d"


# ============================================================
# Helpers
# ============================================================

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def annuity(r, n):
    if r == 0:
        return 1 / n
    return r / (1 - (1 + r) ** (-n))


def annualized_capital_cost_per_mw(capex_per_kw, fixed_om_per_kw_year,
                                    lifetime_years, discount_rate):
    return (annuity(discount_rate, lifetime_years) * capex_per_kw * 1000
            + fixed_om_per_kw_year * 1000)


def load_timeseries(config):
    processed_dir = BASE_DIR / config["paths"]["processed_data_dir"]
    path = processed_dir / config["processed_files"]["merged_timeseries"]
    if not path.exists():
        raise FileNotFoundError(f"Run data_loader.py first: {path}")
    ts = pd.read_csv(path)
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])
    ts = ts.set_index("timestamp")
    ts["solar_cf"] = ts["solar_cf"].clip(0, 1)
    return ts


# ============================================================
# Build base network (shared between PRM and no-PRM runs)
# ============================================================

def build_base_network(config, ts, scc):
    """
    Builds solar+gas network at a given SCC.
    Gas marginal cost is recalculated from SCC, overriding YAML value.
    Returns network n ready for optimization (no extra_functionality yet).
    """
    g = config["technologies"]["natural_gas_cc"]
    s = config["technologies"]["solar"]
    r = config["project"]["discount_rate"]

    fuel_cost    = g["fuel_price_per_mmbtu"] * g["heat_rate_mmbtu_per_mwh"]
    carbon_cost  = scc * g["co2_intensity_t_per_mwh"]
    gas_marginal = g["variable_om_per_mwh"] + fuel_cost + carbon_cost

    n = pypsa.Network()
    n.add("Bus", "California", carrier="AC")
    n.set_snapshots(ts.index)
    n.snapshot_weightings.loc[:, :] = 1.0

    n.add("Carrier", "solar",          co2_emissions=0)
    n.add("Carrier", "natural_gas_cc", co2_emissions=g["co2_intensity_t_per_mwh"])
    n.add("Carrier", "load_shedding",  co2_emissions=0)

    n.add("Load", "demand", bus="California", p_set=ts["demand_mw"])

    solar_cap_cost = annualized_capital_cost_per_mw(
        s["capex_per_kw"], s["fixed_om_per_kw_year"], s["lifetime_years"], r)
    n.add("Generator", "solar",
          bus="California", carrier="solar",
          p_max_pu=ts["solar_cf"],
          capital_cost=solar_cap_cost,
          marginal_cost=0,
          p_nom_extendable=True)

    gas_cap_cost = annualized_capital_cost_per_mw(
        g["capex_per_kw"], g["fixed_om_per_kw_year"], g["lifetime_years"], r)
    n.add("Generator", "natural_gas_cc",
          bus="California", carrier="natural_gas_cc",
          capital_cost=gas_cap_cost,
          marginal_cost=gas_marginal,
          p_nom_extendable=True)

    # VOLL — gas is the last resort before load shedding
    voll = config["scenario_settings"]["value_of_lost_load_per_mwh"]
    n.add("Generator", "load_shedding",
          bus="California", carrier="load_shedding",
          capital_cost=0,
          marginal_cost=voll,
          p_nom_extendable=True)

    return n, gas_marginal


def extract_results(n, scc, gas_marginal):
    """Extracts results dict from a solved network."""
    wt    = n.snapshot_weightings.generators
    gen   = wt @ n.generators_t.p

    solar_gw  = n.generators.at["solar", "p_nom_opt"] / 1000
    gas_gw    = n.generators.at["natural_gas_cc", "p_nom_opt"] / 1000
    solar_twh = gen.get("solar", 0) / 1e6
    gas_twh   = gen.get("natural_gas_cc", 0) / 1e6
    shed_mwh  = gen.get("load_shedding", 0)
    load_mwh  = float(wt @ n.loads_t.p_set.sum(axis=1))
    avg_cost  = n.objective / load_mwh

    emissions_tco2 = float(
        wt @ n.generators_t.p.multiply(
            n.generators.carrier.map(n.carriers.co2_emissions), axis=1
        ).sum(axis=1)
    )

    solar_cf = solar_twh * 1e6 / (solar_gw * 1000 * 8760) if solar_gw > 0 else 0
    gas_cf   = gas_twh   * 1e6 / (gas_gw   * 1000 * 8760) if gas_gw   > 0 else 0

    return {
        "scc_per_tco2":                scc,
        "gas_marginal_cost_per_mwh":   round(gas_marginal, 2),
        "solar_capacity_gw":           round(solar_gw, 2),
        "gas_capacity_gw":             round(gas_gw, 2),
        "solar_capacity_factor_pct":   round(solar_cf * 100, 1),
        "gas_capacity_factor_pct":     round(gas_cf   * 100, 1),
        "solar_generation_twh":        round(solar_twh, 2),
        "gas_generation_twh":          round(gas_twh, 2),
        "operational_emissions_mtco2": round(emissions_tco2 / 1e6, 2),
        "avg_system_cost_per_mwh":     round(avg_cost, 1),
        "load_shedding_mwh":           round(shed_mwh, 0),
        "load_shedding_pct":           round(shed_mwh / load_mwh * 100, 3),
    }


# ============================================================
# RUN A — With ELCC-weighted PRM (realistic)
# ============================================================

def run_with_prm(config, ts, scc):
    """
    Solves solar+gas at given SCC with ELCC-weighted PRM constraint.

    CONSTRAINT C6:
        0.95 * P_gas + 0.21 * P_solar >= (1 + 0.17) * D_peak
        0.95 * P_gas + 0.21 * P_solar >= 51,390 MW

    This is the REALISTIC run. Gas cannot go to zero because
    solar's ELCC of 0.21 means it contributes only 21 cents of
    firm capacity per MW installed. Gas stays high enough to
    fill the remaining firm capacity gap.

    Key insight: even at $200/tCO2, gas does NOT retire completely
    because solar cannot provide firm nighttime capacity. This is
    the correct policy-grounded result.

    ELCC sources (Pham et al., 2024, NREL/TP-7A40-89587):
        ELCC_gas   = 0.95  (CAISO RA qualifying capacity, CPUC R.23-10-011)
        ELCC_solar = 0.21  (CAISO 2026 median, Figure 4 panel 1)
    PRM source:
        17% per CPUC R.23-10-011, effective 2024.
    """
    n, gas_marginal = build_base_network(config, ts, scc)

    prm              = config["scenario_settings"]["planning_reserve_margin"]
    peak_demand_mw   = ts["demand_mw"].max()
    required_firm_mw = peak_demand_mw * (1 + prm)
    elcc             = config["scenario_settings"]["elcc"]
    elcc_gas         = elcc["natural_gas_cc"]   # 0.95
    elcc_solar       = elcc["solar"]            # 0.21

    def prm_extra_functionality(n, snapshots):
        p_gas   = n.model["Generator-p_nom"].sel(name="natural_gas_cc")
        p_solar = n.model["Generator-p_nom"].sel(name="solar")
        lhs     = elcc_gas * p_gas + elcc_solar * p_solar
        n.model.add_constraints(
            lhs >= required_firm_mw,
            name="prm_elcc_constraint"
        )

    n.optimize(solver_name="highs", extra_functionality=prm_extra_functionality)
    return extract_results(n, scc, gas_marginal)


# ============================================================
# RUN B — Without PRM (theoretical)
# ============================================================

def run_without_prm(config, ts, scc):
    """
    Solves solar+gas at given SCC with NO reserve margin constraint.

    Gas optimizes freely — can go to zero if solar is cheaper.
    This isolates the pure carbon price effect on investment.

    This is the THEORETICAL run. It shows what cost-optimal
    investment looks like ignoring reliability policy entirely.
    No real California utility operates without a reserve margin.

    Useful for:
        - Showing the theoretical breakeven SCC
        - Comparing against the PRM run to quantify the
          reliability premium (extra cost imposed by PRM)
        - Academic comparison with unrestricted LP models
    """
    n, gas_marginal = build_base_network(config, ts, scc)
    n.optimize(solver_name="highs")
    return extract_results(n, scc, gas_marginal)


# ============================================================
# Analytical breakeven
# ============================================================

def compute_breakeven(config):
    """
    Analytical breakeven SCC where solar LCOE = gas LCOE (no PRM).

    Formula:
        breakeven = (solar_lcoe - gas_no_carbon) / co2_intensity

    CF values (hardcoded from 2025 CAISO data and solar_gas results):
        solar_cf_base = 0.302  (2025 CAISO observed mean)
        gas_cf_base   = 0.509  (gas CF when serving all load)

    Limitation: gas CF drops as solar grows (endogenous in PyPSA).
    This formula holds CF fixed → OVERESTIMATES true breakeven.
    PyPSA results capture this correctly.
    """
    g = config["technologies"]["natural_gas_cc"]
    s = config["technologies"]["solar"]
    r = config["project"]["discount_rate"]

    solar_cap_cost = annualized_capital_cost_per_mw(
        s["capex_per_kw"], s["fixed_om_per_kw_year"], s["lifetime_years"], r)
    solar_cf_base = 0.302
    solar_lcoe    = solar_cap_cost / (solar_cf_base * 8760)

    gas_cap_cost = annualized_capital_cost_per_mw(
        g["capex_per_kw"], g["fixed_om_per_kw_year"], g["lifetime_years"], r)
    gas_cf_base   = 0.509
    gas_cap_lcoe  = gas_cap_cost / (gas_cf_base * 8760)
    gas_no_carbon = (gas_cap_lcoe
                     + g["variable_om_per_mwh"]
                     + g["fuel_price_per_mmbtu"] * g["heat_rate_mmbtu_per_mwh"])

    breakeven_scc = (solar_lcoe - gas_no_carbon) / g["co2_intensity_t_per_mwh"]
    return solar_lcoe, gas_no_carbon, breakeven_scc


# ============================================================
# Plots
# ============================================================

def plot_capacity_single(df, title, save_path, with_prm=True):
    """
    Figures 15 & 16: Capacity vs SCC for one run (PRM or no-PRM).
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(df["scc_per_tco2"], df["solar_capacity_gw"],
            color=COLOR_SOLAR, lw=2.5, marker="o", markersize=6,
            label="Solar PV capacity (GW)")
    ax.plot(df["scc_per_tco2"], df["gas_capacity_gw"],
            color=COLOR_GAS, lw=2.5, marker="s", markersize=6,
            label="Natural Gas CC capacity (GW)")

    ymax = max(df["solar_capacity_gw"].max(), df["gas_capacity_gw"].max()) * 1.15
    ax.set_ylim(0, ymax)

    for scc, label in SCC_REFERENCES.items():
        if scc in df["scc_per_tco2"].values:
            ax.axvline(scc, color="gray", lw=0.8, ls="--", alpha=0.5)
            ax.text(scc + 1.5, ymax * 0.92,
                    label, fontsize=7.5, color="gray", va="top")

    constraint_note = ("17% ELCC-weighted PRM enforced — gas cannot retire fully"
                       if with_prm else
                       "No reserve margin — pure carbon price effect")
    ax.set_xlabel("Social Cost of Carbon ($/tCO2)")
    ax.set_ylabel("Optimized Capacity (GW)")
    ax.set_title(f"{title}\n{constraint_note}")
    ax.legend(loc="center right")
    ax.set_xlim(-5, 210)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_comparison(df_prm, df_no_prm, save_path):
    """
    Figure 17: Side-by-side gas and solar capacity comparison
    between PRM and no-PRM runs at each SCC value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Gas capacity
    axes[0].plot(df_prm["scc_per_tco2"], df_prm["gas_capacity_gw"],
                 color=COLOR_PRM, lw=2.5, marker="s", markersize=6,
                 label="With PRM (17% ELCC-weighted)")
    axes[0].plot(df_no_prm["scc_per_tco2"], df_no_prm["gas_capacity_gw"],
                 color=COLOR_NO_PRM, lw=2.5, marker="s", markersize=6,
                 ls="--", label="Without PRM (theoretical)")
    axes[0].set_xlabel("Social Cost of Carbon ($/tCO2)")
    axes[0].set_ylabel("Gas Capacity (GW)")
    axes[0].set_title("Natural Gas Capacity")
    axes[0].set_xlim(-5, 210)
    axes[0].legend()

    # Right: Solar capacity
    axes[1].plot(df_prm["scc_per_tco2"], df_prm["solar_capacity_gw"],
                 color=COLOR_PRM, lw=2.5, marker="o", markersize=6,
                 label="With PRM (17% ELCC-weighted)")
    axes[1].plot(df_no_prm["scc_per_tco2"], df_no_prm["solar_capacity_gw"],
                 color=COLOR_NO_PRM, lw=2.5, marker="o", markersize=6,
                 ls="--", label="Without PRM (theoretical)")
    axes[1].set_xlabel("Social Cost of Carbon ($/tCO2)")
    axes[1].set_ylabel("Solar Capacity (GW)")
    axes[1].set_title("Solar PV Capacity")
    axes[1].set_xlim(-5, 210)
    axes[1].legend()

    for ax in axes:
        for scc in [30, 190]:
            ax.axvline(scc, color="gray", lw=0.8, ls="--", alpha=0.4)

    fig.suptitle(
        "PRM vs. No-PRM Capacity Comparison\n"
        "Gap between lines = reliability premium imposed by CPUC 17% reserve margin",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_cost_emissions(df_prm, df_no_prm, save_path):
    """
    Figure 18: System cost and emissions comparison PRM vs no-PRM.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: system cost
    axes[0].plot(df_prm["scc_per_tco2"], df_prm["avg_system_cost_per_mwh"],
                 color=COLOR_PRM, lw=2.5, marker="o", markersize=6,
                 label="With PRM (realistic)")
    axes[0].plot(df_no_prm["scc_per_tco2"], df_no_prm["avg_system_cost_per_mwh"],
                 color=COLOR_NO_PRM, lw=2.5, marker="o", markersize=6,
                 ls="--", label="Without PRM (theoretical)")
    axes[0].set_xlabel("Social Cost of Carbon ($/tCO2)")
    axes[0].set_ylabel("Average System Cost ($/MWh)")
    axes[0].set_title("System Cost vs. SCC")
    axes[0].set_xlim(-5, 210)
    axes[0].legend()

    # Right: emissions
    axes[1].plot(df_prm["scc_per_tco2"], df_prm["operational_emissions_mtco2"],
                 color=COLOR_PRM, lw=2.5, marker="o", markersize=6,
                 label="With PRM (realistic)")
    axes[1].plot(df_no_prm["scc_per_tco2"], df_no_prm["operational_emissions_mtco2"],
                 color=COLOR_NO_PRM, lw=2.5, marker="o", markersize=6,
                 ls="--", label="Without PRM (theoretical)")
    axes[1].axhline(30, color="gray", lw=1, ls="--", alpha=0.7,
                    label="CPUC 2030 target (30 MtCO2)")
    axes[1].axhline(20, color="gray", lw=1, ls=":",  alpha=0.7,
                    label="CPUC 2035 target (~20 MtCO2)")
    axes[1].set_xlabel("Social Cost of Carbon ($/tCO2)")
    axes[1].set_ylabel("Operational CO2 (MtCO2/year)")
    axes[1].set_title("Emissions vs. SCC")
    axes[1].set_xlim(-5, 210)
    axes[1].legend(fontsize=8)

    for ax in axes:
        for scc in [30, 190]:
            ax.axvline(scc, color="gray", lw=0.8, ls="--", alpha=0.4)

    fig.suptitle(
        "Cost and Emissions: PRM vs. No-PRM\n"
        "With PRM: higher cost, higher emissions at each SCC (gas floor effect)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_breakeven(config, df_prm, df_no_prm, save_path):
    """
    Figure 19: Gas vs solar LCOE breakeven chart with annotated
    PRM and no-PRM crossover points from PyPSA results.
    """
    g = config["technologies"]["natural_gas_cc"]
    s = config["technologies"]["solar"]
    r = config["project"]["discount_rate"]

    solar_lcoe, gas_no_carbon, breakeven_scc = compute_breakeven(config)

    scc_range = np.linspace(0, 210, 400)

    gas_lcoe_line = [gas_no_carbon + scc * g["co2_intensity_t_per_mwh"]
                     for scc in scc_range]

    def gas_cap_lcoe_at_cf(cf):
        cap = annualized_capital_cost_per_mw(
            g["capex_per_kw"], g["fixed_om_per_kw_year"], g["lifetime_years"], r)
        return cap / (cf * 8760)

    fuel     = g["variable_om_per_mwh"] + g["fuel_price_per_mmbtu"] * g["heat_rate_mmbtu_per_mwh"]
    gas_low  = [gas_cap_lcoe_at_cf(0.65) + fuel + scc * g["co2_intensity_t_per_mwh"]
                for scc in scc_range]
    gas_high = [gas_cap_lcoe_at_cf(0.40) + fuel + scc * g["co2_intensity_t_per_mwh"]
                for scc in scc_range]

    solar_cap_cost = annualized_capital_cost_per_mw(
        s["capex_per_kw"], s["fixed_om_per_kw_year"], s["lifetime_years"], r)
    solar_low  = solar_cap_cost / (0.32 * 8760)
    solar_high = solar_cap_cost / (0.20 * 8760)

    ymax = max(max(gas_high), solar_high) * 1.15

    fig, ax = plt.subplots(figsize=(12, 6))

    # LCOE lines
    ax.plot(scc_range, gas_lcoe_line, color=COLOR_GAS, lw=2.5,
            label="Gas LCOE (base CF 0.51) — slope = 0.36 $/MWh per $/tCO2")
    ax.fill_between(scc_range, gas_low, gas_high,
                    color=COLOR_GAS, alpha=0.12,
                    label="Gas LCOE range (CF 0.40-0.65)")
    ax.axhline(solar_lcoe, color=COLOR_SOLAR, lw=2.5,
               label=f"Solar LCOE (base CF 0.30) = ${solar_lcoe:.1f}/MWh")
    ax.axhspan(solar_low, solar_high, color=COLOR_SOLAR, alpha=0.12,
               label=f"Solar LCOE range (CF 0.20-0.32) = ${solar_low:.0f}-${solar_high:.0f}/MWh")

    # Analytical breakeven
    if 0 < breakeven_scc < 210:
        ax.axvline(breakeven_scc, color="#555555", lw=1.5, ls="-.",
                   label=f"Analytical breakeven = ${breakeven_scc:.0f}/tCO2 (no PRM, fixed CF)")
        ax.annotate(
            f"Analytical\n${breakeven_scc:.0f}/tCO2",
            xy=(breakeven_scc, solar_lcoe),
            xytext=(breakeven_scc + 10, solar_lcoe + 8),
            fontsize=9, color="#555555", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2)
        )

    # Find PyPSA crossover points from results
    # No-PRM crossover: first SCC where solar > 0 GW
    crossover_no_prm = None
    for _, row in df_no_prm.iterrows():
        if row["solar_capacity_gw"] > 0.5:
            crossover_no_prm = row["scc_per_tco2"]
            break

    # PRM crossover: first SCC where solar starts growing meaningfully
    crossover_prm = None
    for _, row in df_prm.iterrows():
        if row["solar_capacity_gw"] > 0.5:
            crossover_prm = row["scc_per_tco2"]
            break

    if crossover_no_prm is not None:
        ax.axvline(crossover_no_prm, color=COLOR_NO_PRM, lw=2, ls=":",
                   label=f"PyPSA crossover (no PRM) ≈ ${crossover_no_prm}/tCO2")

    if crossover_prm is not None:
        ax.axvline(crossover_prm, color=COLOR_PRM, lw=2, ls=":",
                   label=f"PyPSA crossover (with PRM) ≈ ${crossover_prm}/tCO2")

    # Reference SCC lines
    refs = [
        (30,  "CA C&T\n$30"),
        (51,  "EPA IWG\n$51\n(2016)"),
        (190, "EPA IWG\n$190\n(2023)"),
        (200, "Rennert\n$200"),
    ]
    for xval, label in refs:
        ax.axvline(xval, color="#aaaaaa", lw=1, ls="--", alpha=0.5)
        ax.text(xval + 1.5, ymax * 0.10, label,
                fontsize=7.5, color="#666666", va="bottom")

    # Region labels
    if 0 < breakeven_scc < 210:
        ax.text(breakeven_scc / 2, ymax * 0.55,
                "Gas\ncheaper", ha="center", fontsize=10,
                color=COLOR_GAS, alpha=0.7, style="italic")
        ax.text(min((breakeven_scc + 210) / 2, 200), ymax * 0.55,
                "Solar\ncheaper", ha="center", fontsize=10,
                color="#c9a000", alpha=0.7, style="italic")

    ax.set_xlabel("Social Cost of Carbon ($/tCO2)", fontsize=12)
    ax.set_ylabel("LCOE ($/MWh)", fontsize=12)
    ax.set_title(
        "Solar vs. Gas LCOE — Breakeven Analysis\n"
        f"Analytical breakeven at ${breakeven_scc:.0f}/tCO2 (no PRM, fixed CF)  |  "
        f"CA C&T $30 is {'below' if 30 < breakeven_scc else 'above'} breakeven  |  "
        f"EPA 2023 $190 is {'below' if 190 < breakeven_scc else 'above'} breakeven\n"
        "Sources: NREL ATB 2026; CARB (2025); EPA IWG (2016, 2023); Rennert et al. (2022)",
        fontsize=11
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 210)
    ax.set_ylim(0, ymax)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# Print results table
# ============================================================

def print_results_table(df, label):
    print(f"\n  -- {label} --")
    print(f"  {'SCC':>6}  {'Gas $/MWh':>10}  {'Solar GW':>9}  "
          f"{'Gas GW':>7}  {'Cost $/MWh':>11}  {'CO2 MtCO2':>10}  {'Shed%':>7}")
    print(f"  {'------':>6}  {'----------':>10}  {'---------':>9}  "
          f"{'-------':>7}  {'-----------':>11}  {'----------':>10}  {'-------':>7}")
    _, _, breakeven_scc = compute_breakeven(
        {"technologies": {"natural_gas_cc": {}, "solar": {}}, "project": {}})

    for _, r in df.iterrows():
        print(f"  ${r['scc_per_tco2']:>5}  "
              f"${r['gas_marginal_cost_per_mwh']:>9.1f}  "
              f"{r['solar_capacity_gw']:>9.1f}  "
              f"{r['gas_capacity_gw']:>7.1f}  "
              f"${r['avg_system_cost_per_mwh']:>10.1f}  "
              f"{r['operational_emissions_mtco2']:>10.2f}  "
              f"{r['load_shedding_pct']:>7.3f}")


# ============================================================
# Main
# ============================================================

def main():
    config     = load_config()
    results_dir = BASE_DIR / config["paths"]["results_dir"]
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Validate VOLL
    voll = config["scenario_settings"]["value_of_lost_load_per_mwh"]
    if voll < 10000:
        print(f"\n  WARNING: VOLL = ${voll:,}/MWh is below $10,000.")
        print(f"  Set value_of_lost_load_per_mwh: 30000 in data.yaml.\n")

    prm = config["scenario_settings"]["planning_reserve_margin"]

    print("\n============================================================")
    print("  Social Cost of Carbon Sensitivity Analysis")
    print("  Two runs: WITH PRM (realistic) and WITHOUT PRM (theoretical)")
    print("============================================================")
    print(f"\n  SCC values:  {SCC_VALUES}")
    print(f"  VOLL:        ${voll:,}/MWh")
    print(f"  PRM:         {prm*100:.0f}% (CPUC R.23-10-011)")
    print(f"  ELCC gas:    {config['scenario_settings']['elcc']['natural_gas_cc']}")
    print(f"  ELCC solar:  {config['scenario_settings']['elcc']['solar']}")

    ts = load_timeseries(config)

    # Analytical breakeven
    solar_lcoe, gas_no_carbon, breakeven_scc = compute_breakeven(config)
    print(f"\n  -- ANALYTICAL BREAKEVEN (no PRM, fixed CF) --")
    print(f"  Solar LCOE (CF 0.302):  ${solar_lcoe:.1f}/MWh")
    print(f"  Gas LCOE (no carbon):   ${gas_no_carbon:.1f}/MWh")
    print(f"  Breakeven SCC:          ${breakeven_scc:.0f}/tCO2")
    print(f"  CA C&T ($30):  {'BELOW' if 30 < breakeven_scc else 'ABOVE'} breakeven")
    print(f"  EPA 2023 ($190): {'BELOW' if 190 < breakeven_scc else 'ABOVE'} breakeven")
    print(f"  Note: true PyPSA breakeven is lower (gas CF drops endogenously)")

    # ── RUN A: With PRM ──────────────────────────────────────
    print(f"\n  -- RUN A: WITH ELCC-WEIGHTED PRM ({prm*100:.0f}%) --")
    print(f"  Constraint: 0.95*P_gas + 0.21*P_solar >= "
          f"{ts['demand_mw'].max() * (1+prm):,.0f} MW")
    rows_prm = []
    for scc in SCC_VALUES:
        print(f"  SCC=${scc:>4}/tCO2 ...", end=" ", flush=True)
        result = run_with_prm(config, ts, scc)
        rows_prm.append(result)
        print(f"Solar: {result['solar_capacity_gw']:.1f} GW | "
              f"Gas: {result['gas_capacity_gw']:.1f} GW | "
              f"${result['avg_system_cost_per_mwh']:.1f}/MWh | "
              f"{result['operational_emissions_mtco2']:.1f} MtCO2")

    df_prm = pd.DataFrame(rows_prm)
    df_prm.to_csv(results_dir / "carbon_sensitivity_with_prm.csv", index=False)
    print(f"  Saved: carbon_sensitivity_with_prm.csv")

    # ── RUN B: Without PRM ───────────────────────────────────
    print(f"\n  -- RUN B: WITHOUT PRM (theoretical baseline) --")
    print(f"  No reserve margin — gas optimizes freely")
    rows_no_prm = []
    for scc in SCC_VALUES:
        print(f"  SCC=${scc:>4}/tCO2 ...", end=" ", flush=True)
        result = run_without_prm(config, ts, scc)
        rows_no_prm.append(result)
        print(f"Solar: {result['solar_capacity_gw']:.1f} GW | "
              f"Gas: {result['gas_capacity_gw']:.1f} GW | "
              f"${result['avg_system_cost_per_mwh']:.1f}/MWh | "
              f"{result['operational_emissions_mtco2']:.1f} MtCO2")

    df_no_prm = pd.DataFrame(rows_no_prm)
    df_no_prm.to_csv(results_dir / "carbon_sensitivity_no_prm.csv", index=False)
    print(f"  Saved: carbon_sensitivity_no_prm.csv")

    # ── Summary tables ───────────────────────────────────────
    print(f"\n  -- RESULTS: WITH PRM --")
    print(f"  {'SCC':>6}  {'Solar GW':>9}  {'Gas GW':>7}  "
          f"{'Cost $/MWh':>11}  {'CO2 MtCO2':>10}  {'Shed%':>7}")
    print(f"  {'------':>6}  {'---------':>9}  {'-------':>7}  "
          f"{'-----------':>11}  {'----------':>10}  {'-------':>7}")
    for _, r in df_prm.iterrows():
        print(f"  ${r['scc_per_tco2']:>5}  "
              f"{r['solar_capacity_gw']:>9.1f}  "
              f"{r['gas_capacity_gw']:>7.1f}  "
              f"${r['avg_system_cost_per_mwh']:>10.1f}  "
              f"{r['operational_emissions_mtco2']:>10.2f}  "
              f"{r['load_shedding_pct']:>7.3f}")

    print(f"\n  -- RESULTS: WITHOUT PRM --")
    print(f"  {'SCC':>6}  {'Solar GW':>9}  {'Gas GW':>7}  "
          f"{'Cost $/MWh':>11}  {'CO2 MtCO2':>10}  {'Shed%':>7}")
    print(f"  {'------':>6}  {'---------':>9}  {'-------':>7}  "
          f"{'-----------':>11}  {'----------':>10}  {'-------':>7}")
    for _, r in df_no_prm.iterrows():
        print(f"  ${r['scc_per_tco2']:>5}  "
              f"{r['solar_capacity_gw']:>9.1f}  "
              f"{r['gas_capacity_gw']:>7.1f}  "
              f"${r['avg_system_cost_per_mwh']:>10.1f}  "
              f"{r['operational_emissions_mtco2']:>10.2f}  "
              f"{r['load_shedding_pct']:>7.3f}")

    # ── Figures ──────────────────────────────────────────────
    print(f"\n  -- GENERATING FIGURES --")
    plot_capacity_single(
        df_prm,
        "Optimized Capacity vs. SCC — WITH ELCC-Weighted PRM (17%)",
        figures_dir / "15_sensitivity_capacity_with_prm.png",
        with_prm=True
    )
    plot_capacity_single(
        df_no_prm,
        "Optimized Capacity vs. SCC — WITHOUT PRM (Theoretical)",
        figures_dir / "16_sensitivity_capacity_no_prm.png",
        with_prm=False
    )
    plot_comparison(df_prm, df_no_prm,
                    figures_dir / "17_sensitivity_comparison.png")
    plot_cost_emissions(df_prm, df_no_prm,
                        figures_dir / "18_sensitivity_cost_emissions.png")
    plot_breakeven(config, df_prm, df_no_prm,
                   figures_dir / "19_sensitivity_breakeven.png")

    print(f"\n  All outputs saved to: {results_dir}")
    print(f"  Figures:              {figures_dir}")
    print()


if __name__ == "__main__":
    main()
