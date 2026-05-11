import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pypsa
import yaml
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "dataset" / "data.yaml"

# ============================================================
# Report style — clean, publication-ready
# ============================================================

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.25,
    "grid.linewidth":    0.6,
    "legend.frameon":    False,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.dpi":       200,
})

# ============================================================
# Scenario display labels
# ============================================================

SCENARIO_LABELS = {
    "solar_bess":        "Solar +\nBESS",
    "solar_gas":         "Solar +\nNat. Gas",
    "solar_gas_co2cap":  "Solar + Gas\n(CO₂ Cap)",
    "solar_gas_rps":     "Solar + Gas\n(60% RPS)",
}

SCENARIO_COLORS = {
    "solar_bess":        "#2196a8",
    "solar_gas":         "#e05c5c",
    "solar_gas_co2cap":  "#e8912d",
    "solar_gas_rps":     "#6abf69",
}

TECH_COLORS = {
    "solar":           "#f5c518",
    "battery":         "#6abf69",
    "natural_gas_cc":  "#e05c5c",
    "load_shedding":   "#222222",
}

CARRIER_COLORS = {**TECH_COLORS, "load": "#000000"}


# ============================================================
# Config + paths
# ============================================================

def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def get_paths(config):
    results_dir = BASE_DIR / config["paths"]["results_dir"]
    networks_dir = results_dir / "networks"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, networks_dir, figures_dir


def load_network(config, scenario):
    _, networks_dir, _ = get_paths(config)
    path = networks_dir / f"{scenario}.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find solved network: {path}\n"
            "Run optimizer first: python src/optimizer.py"
        )
    return pypsa.Network(path)


def slabel(s):
    return SCENARIO_LABELS.get(s, s)


def scolor(s):
    return SCENARIO_COLORS.get(s, "#aaaaaa")


def build_dispatch_df(n):
    frames = []
    for gen in n.generators.index:
        carrier = n.generators.at[gen, "carrier"]
        frames.append(n.generators_t.p[gen].rename(carrier))
    for su in n.storage_units.index:
        carrier = n.storage_units.at[su, "carrier"]
        p = n.storage_units_t.p[su]
        frames.append(p.clip(lower=0).rename(f"{carrier}_discharge"))
        frames.append(p.clip(upper=0).rename(f"{carrier}_charge"))
    dispatch = pd.concat(frames, axis=1)
    return dispatch.T.groupby(dispatch.columns).sum().T


# ============================================================
# FIGURE 1: Average System Cost
# ============================================================

def plot_system_cost(comparison, save_path):
    scenarios = comparison["scenario"].tolist()
    costs     = comparison["average_system_cost_$_per_mwh"].tolist()
    labels    = [slabel(s) for s in scenarios]
    colors    = [scolor(s) for s in scenarios]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, costs, color=colors, width=0.55, alpha=0.9,
                  edgecolor="white", linewidth=0.5)

    for bar, cost in zip(bars, costs):
        offset = max(costs) * 0.015
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"${cost:,.0f}",
                ha="center", va="bottom",
                fontsize=9 if cost > 2000 else 10,
                fontweight="bold")

    baseline = comparison.loc[comparison["scenario"] == "solar_gas",
                               "average_system_cost_$_per_mwh"]
    if len(baseline):
        ax.axhline(baseline.values[0], color="#e05c5c", linewidth=1,
                   linestyle="--", alpha=0.5,
                   label=f"Solar+Gas baseline (${baseline.values[0]:,.0f}/MWh)")
        ax.legend()

    ax.set_title("Average System Cost by Scenario")
    ax.set_ylabel("$/MWh")

    fig.text(0.5, -0.04,
             "Note: solar_gas_rps ($13,991/MWh) and solar_gas_co2cap ($5,267/MWh) costs reflect\n"
             "load shedding penalties — see reliability panel for context.",
             ha="center", fontsize=8.5, color="#666666", style="italic")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# FIGURE 2: Optimized Capacity (stacked bar, excludes rps for scale)
# ============================================================

def plot_capacity_stacked(comparison, save_path):
    df = comparison[comparison["scenario"] != "solar_gas_rps"].copy()
    scenarios = df["scenario"].tolist()
    labels    = [slabel(s) for s in scenarios]
    x         = np.arange(len(scenarios))
    w         = 0.55

    solar   = df["solar_capacity_gw"].fillna(0).tolist()
    battery = df["battery_power_gw"].fillna(0).tolist()
    gas     = df["gas_capacity_gw"].fillna(0).tolist()
    bot2    = [s + b for s, b in zip(solar, battery)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, solar,   w, label="Solar PV",        color=TECH_COLORS["solar"],          alpha=0.9)
    ax.bar(x, battery, w, label="BESS (power)",    color=TECH_COLORS["battery"],         alpha=0.9, bottom=solar)
    ax.bar(x, gas,     w, label="Natural Gas CC",  color=TECH_COLORS["natural_gas_cc"],  alpha=0.9, bottom=bot2)

    ax.axhline(44, color="black", linewidth=1, linestyle=":", alpha=0.6,
               label="CA peak demand ≈ 44 GW")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Optimized Installed Capacity by Scenario\n"
                 "(solar_gas_rps excluded — requires 23,534 GW solar)")
    ax.set_ylabel("Capacity (GW)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# FIGURE 3: Annual Generation Mix (stacked bar + unserved load)
# ============================================================

def plot_generation_mix(comparison, save_path):
    scenarios = comparison["scenario"].tolist()
    labels    = [slabel(s) for s in scenarios]
    x         = np.arange(len(scenarios))
    w         = 0.55

    solar    = comparison["solar_generation_twh"].fillna(0).tolist()
    gas      = comparison["gas_generation_twh"].fillna(0).tolist()
    shed_twh = (comparison["load_shedding_mwh"].fillna(0) / 1e6).tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, solar, w, label="Solar PV",        color=TECH_COLORS["solar"],         alpha=0.9)
    ax.bar(x, gas,   w, label="Natural Gas CC",  color=TECH_COLORS["natural_gas_cc"], alpha=0.9,
           bottom=solar)
    ax.bar(x, [-s for s in shed_twh], w, label="Unserved Load (below axis)",
           color=TECH_COLORS["load_shedding"], alpha=0.55, hatch="//", edgecolor="white")

    total_load = comparison["load_mwh"].iloc[0] / 1e6
    ax.axhline(total_load, color="black", linewidth=1.2, linestyle="--", alpha=0.7,
               label=f"Annual demand ({total_load:.0f} TWh)")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Annual Generation Mix by Scenario\n"
                 "(hatched bar below axis = unserved load)")
    ax.set_ylabel("TWh/year")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# FIGURE 4: CO2 Emissions vs CPUC Targets
# ============================================================

def plot_emissions(comparison, save_path):
    scenarios = comparison["scenario"].tolist()
    labels    = [slabel(s) for s in scenarios]
    x         = np.arange(len(scenarios))
    w         = 0.35

    op_vals  = (comparison["operational_emissions_tco2"].fillna(0)  / 1e6).tolist()
    lca_vals = (comparison["total_emissions_with_lca_tco2e"].fillna(0) / 1e6).tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, op_vals,  w, label="Operational CO₂",
           color="#e05c5c", alpha=0.88)
    ax.bar(x + w/2, lca_vals, w, label="Total incl. life-cycle CO₂e",
           color="#b56ecc", alpha=0.88)

    ax.axhline(30, color="#555555", linewidth=1.2, linestyle="--", alpha=0.65,
               label="CPUC 2030 target — 30 MtCO₂ statewide (D.24-02-047)")
    ax.axhline(20, color="#555555", linewidth=1.2, linestyle=":",  alpha=0.65,
               label="CPUC 2035 target — CAISO share ~20 MtCO₂ (D.24-02-047)")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Annual CO₂ Emissions by Scenario\nvs. CPUC IRP Targets (Decision 24-02-047, Feb 2024)")
    ax.set_ylabel("MtCO₂e / year")
    ax.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# FIGURE 5: Cost vs Emissions tradeoff scatter
# ============================================================

def plot_cost_emissions_tradeoff(comparison, save_path):
    scenarios = comparison["scenario"].tolist()
    costs     = comparison["average_system_cost_$_per_mwh"].tolist()
    emissions = (comparison["operational_emissions_tco2"].fillna(0) / 1e6).tolist()
    shed_pct  = (comparison["load_shedding_mwh"].fillna(0)
                 / comparison["load_mwh"] * 100).tolist()

    fig, ax = plt.subplots(figsize=(9, 6))

    for s, cost, em, shed in zip(scenarios, costs, emissions, shed_pct):
        size   = max(shed * 300 + 80, 80)
        marker = "D" if shed > 1 else "o"
        ax.scatter(em, cost, s=size, color=scolor(s), alpha=0.85,
                   edgecolors="white", linewidths=1.5, marker=marker, zorder=3)
        ax.annotate(
            slabel(s).replace("\n", " "),
            (em, cost),
            textcoords="offset points", xytext=(10, 6),
            fontsize=9, color=scolor(s), fontweight="bold",
        )

    ymax = max(costs)
    ax.axvline(30, color="gray", linewidth=1,   linestyle="--", alpha=0.5)
    ax.axvline(20, color="gray", linewidth=1,   linestyle=":",  alpha=0.5)
    ax.text(30.8, ymax * 0.97, "2030\ntarget\n30 Mt", fontsize=8, color="gray", va="top")
    ax.text(20.8, ymax * 0.87, "2035\ntarget\n~20 Mt", fontsize=8, color="gray", va="top")

    legend_elements = [
        mpatches.Patch(color="#cccccc", label="Bubble size ∝ unserved energy (%)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
                   markersize=9, label="Reliable (shedding < 1%)"),
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor="#888",
                   markersize=9, label="Unreliable (shedding ≥ 1%)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5)

    ax.set_xlabel("Operational CO₂ Emissions (MtCO₂/year)")
    ax.set_ylabel("Average System Cost ($/MWh)")
    ax.set_title("Cost vs. Emissions Tradeoff\nBubble size = unserved energy (%)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# FIGURE 6: Reliability
# ============================================================

def plot_load_shedding(comparison, save_path):
    scenarios = comparison["scenario"].tolist()
    labels    = [slabel(s) for s in scenarios]
    shed_pct  = (comparison["load_shedding_mwh"].fillna(0)
                 / comparison["load_mwh"] * 100).tolist()
    colors    = [scolor(s) for s in scenarios]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, shed_pct, color=colors, width=0.55, alpha=0.9,
                  edgecolor="white")

    for bar, pct in zip(bars, shed_pct):
        if pct > 0.0001:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(shed_pct) * 0.01,
                    f"{pct:.3f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0.01, color="red", linewidth=1.2, linestyle="--", alpha=0.7,
               label="NERC reliability standard (~0.01% unserved energy)")
    ax.legend()
    ax.set_title("Unserved Energy (Load Shedding) by Scenario\nvs. NERC Reliability Standard")
    ax.set_ylabel("Unserved Energy (% of annual load)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# FIGURE 7: Four-panel dashboard summary
# ============================================================

def plot_dashboard(comparison, save_path):
    scenarios = comparison["scenario"].tolist()
    labels    = [slabel(s).replace("\n", " ") for s in scenarios]
    colors    = [scolor(s) for s in scenarios]
    x         = np.arange(len(scenarios))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "California Capacity Expansion — Four-Scenario Comparison",
        fontsize=14, fontweight="bold", y=1.01
    )

    # (A) System cost
    ax = axes[0, 0]
    costs = comparison["average_system_cost_$_per_mwh"].tolist()
    ax.bar(x, costs, color=colors, width=0.55, alpha=0.9, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=10)
    ax.set_title("(A) Average System Cost")
    ax.set_ylabel("$/MWh")
    for xi, c in zip(x, costs):
        ax.text(xi, c + max(costs)*0.012, f"${c:,.0f}",
                ha="center", fontsize=8, fontweight="bold")

    # (B) Operational emissions
    ax = axes[0, 1]
    em = (comparison["operational_emissions_tco2"].fillna(0) / 1e6).tolist()
    ax.bar(x, em, color=colors, width=0.55, alpha=0.9, edgecolor="white")
    ax.axhline(30, color="gray", lw=1, ls="--", alpha=0.6, label="2030 target (30 Mt)")
    ax.axhline(20, color="gray", lw=1, ls=":",  alpha=0.6, label="2035 CAISO target (20 Mt)")
    for xi, e in zip(x, em):
        ax.text(xi, e + max(em)*0.012, f"{e:.1f}",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=10)
    ax.set_title("(B) Operational CO₂ Emissions")
    ax.set_ylabel("MtCO₂/year")
    ax.legend(fontsize=7.5)

    # (C) Generation mix
    ax = axes[1, 0]
    solar_twh = comparison["solar_generation_twh"].fillna(0).tolist()
    gas_twh   = comparison["gas_generation_twh"].fillna(0).tolist()
    ax.bar(x, solar_twh, 0.55, label="Solar",     color=TECH_COLORS["solar"],         alpha=0.9)
    ax.bar(x, gas_twh,   0.55, label="Gas",        color=TECH_COLORS["natural_gas_cc"], alpha=0.9,
           bottom=solar_twh)
    total_load = comparison["load_mwh"].iloc[0] / 1e6
    ax.axhline(total_load, color="black", lw=1.2, ls="--", alpha=0.6,
               label=f"Load ({total_load:.0f} TWh)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=10)
    ax.set_title("(C) Annual Generation Mix")
    ax.set_ylabel("TWh/year")
    ax.legend(fontsize=7.5)

    # (D) Reliability
    ax = axes[1, 1]
    shed_pct = (comparison["load_shedding_mwh"].fillna(0)
                / comparison["load_mwh"] * 100).tolist()
    ax.bar(x, shed_pct, color=colors, width=0.55, alpha=0.9, edgecolor="white")
    ax.axhline(0.01, color="red", lw=1, ls="--", alpha=0.7, label="NERC target (0.01%)")
    for xi, p in zip(x, shed_pct):
        if p > 0.0001:
            ax.text(xi, p + max(shed_pct)*0.012, f"{p:.2f}%",
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=10)
    ax.set_title("(D) Unserved Energy")
    ax.set_ylabel("% of annual load")
    ax.legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# FIGURE 8: Full-year 8760-hour dispatch
# ============================================================

def plot_dispatch_8760(n, scenario, save_path):
    dispatch = build_dispatch_df(n) / 1000
    load_gw  = n.loads_t.p_set.sum(axis=1) / 1000

    pos_cols = [c for c in dispatch.columns
                if dispatch[c].max() > 0 and not c.endswith("_charge")]
    neg_cols = [c for c in dispatch.columns if c.endswith("_charge")]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.stackplot(dispatch.index,
                 [dispatch[c].clip(lower=0) for c in pos_cols],
                 labels=pos_cols,
                 colors=[CARRIER_COLORS.get(c, "#aaaaaa") for c in pos_cols],
                 alpha=0.85)
    if neg_cols:
        ax.stackplot(dispatch.index,
                     [dispatch[c].clip(upper=0) for c in neg_cols],
                     labels=neg_cols,
                     colors=[CARRIER_COLORS.get(c.replace("_charge",""), "#aaaaaa") for c in neg_cols],
                     alpha=0.5)
    ax.plot(load_gw.index, load_gw.values, color="black", lw=0.8, label="demand", zorder=5)

    ax.set_title(f"Full-Year Hourly Dispatch — {slabel(scenario).replace(chr(10),' ')} (8,760 hours)")
    ax.set_ylabel("GW")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.axhline(0, color="gray", lw=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# FIGURE 9: Representative week dispatch
# ============================================================

def plot_dispatch_week(n, scenario, week_start, label, save_path):
    dispatch = build_dispatch_df(n) / 1000
    load_gw  = n.loads_t.p_set.sum(axis=1) / 1000

    week_end = pd.Timestamp(week_start) + pd.Timedelta(days=7)
    mask     = (dispatch.index >= week_start) & (dispatch.index < week_end)
    d_week, l_week = dispatch[mask], load_gw[mask]

    pos_cols = [c for c in d_week.columns
                if d_week[c].max() > 0 and not c.endswith("_charge")]
    neg_cols = [c for c in d_week.columns if c.endswith("_charge")]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(d_week.index,
                 [d_week[c].clip(lower=0) for c in pos_cols],
                 labels=pos_cols,
                 colors=[CARRIER_COLORS.get(c, "#aaaaaa") for c in pos_cols],
                 alpha=0.85)
    if neg_cols:
        ax.stackplot(d_week.index,
                     [d_week[c].clip(upper=0) for c in neg_cols],
                     labels=neg_cols,
                     colors=[CARRIER_COLORS.get(c.replace("_charge",""), "#aaaaaa") for c in neg_cols],
                     alpha=0.5)
    ax.plot(l_week.index, l_week.values, color="black", lw=1.5, label="demand", zorder=5)

    ax.set_title(f"{slabel(scenario).replace(chr(10),' ')} — {label}")
    ax.set_ylabel("GW")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# FIGURE 10: BESS state of charge
# ============================================================

def plot_soc(n, scenario, save_path):
    if n.storage_units.empty:
        return
    soc_gwh = n.storage_units_t.state_of_charge / 1000
    fig, ax = plt.subplots(figsize=(16, 4))
    for col in soc_gwh.columns:
        ax.plot(soc_gwh.index, soc_gwh[col], lw=0.7,
                label=col, color=TECH_COLORS["battery"])
    ax.set_title(f"Battery State of Charge — {slabel(scenario).replace(chr(10),' ')}")
    ax.set_ylabel("GWh")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# FIGURE 11: Supply–demand balance
# ============================================================

def plot_supply_demand_balance(n, scenario, save_path):
    dispatch     = build_dispatch_df(n)
    total_supply = dispatch.sum(axis=1)
    total_demand = n.loads_t.p_set.sum(axis=1)
    residual     = (total_supply - total_demand) / 1000

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

    dispatch_gw = dispatch / 1000
    pos_cols = [c for c in dispatch_gw.columns
                if dispatch_gw[c].max() > 0 and not c.endswith("_charge")]
    neg_cols = [c for c in dispatch_gw.columns if c.endswith("_charge")]

    axes[0].stackplot(dispatch_gw.index,
                      [dispatch_gw[c].clip(lower=0) for c in pos_cols],
                      labels=pos_cols,
                      colors=[CARRIER_COLORS.get(c, "#aaaaaa") for c in pos_cols],
                      alpha=0.85)
    if neg_cols:
        axes[0].stackplot(dispatch_gw.index,
                          [dispatch_gw[c].clip(upper=0) for c in neg_cols],
                          labels=neg_cols,
                          colors=[CARRIER_COLORS.get(c.replace("_charge",""), "#aaaaaa") for c in neg_cols],
                          alpha=0.5)
    axes[0].plot(total_demand.index, total_demand.values / 1000,
                 color="black", lw=0.8, label="demand", zorder=5)
    axes[0].set_ylabel("GW")
    axes[0].set_title(f"Supply vs Demand — {slabel(scenario).replace(chr(10),' ')}")
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    axes[0].axhline(0, color="gray", lw=0.5)

    axes[1].plot(residual.index, residual.values, color="#3a7ebf", lw=0.6, alpha=0.8)
    axes[1].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[1].fill_between(residual.index, residual.values, 0,
                         where=(residual.values > 0.001),
                         color="#f5c518", alpha=0.4, label="surplus (curtailment)")
    axes[1].fill_between(residual.index, residual.values, 0,
                         where=(residual.values < -0.001),
                         color="#e05c5c", alpha=0.4, label="deficit (load shedding)")
    axes[1].set_ylabel("GW (supply − demand)")
    axes[1].set_xlabel("Month")
    axes[1].legend(fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# Main
# ============================================================

def main():
    config = load_config()
    results_dir, _, figures_dir = get_paths(config)

    comparison_path = results_dir / "scenario_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Cannot find {comparison_path}\n"
            "Run optimizer first: python src/optimizer.py"
        )

    comparison = pd.read_csv(comparison_path)

    print("\nGenerating report-quality summary figures...")
    plot_system_cost(         comparison, figures_dir / "01_system_cost.png")
    plot_capacity_stacked(    comparison, figures_dir / "02_optimized_capacity.png")
    plot_generation_mix(      comparison, figures_dir / "03_generation_mix.png")
    plot_emissions(           comparison, figures_dir / "04_co2_emissions.png")
    plot_cost_emissions_tradeoff(comparison, figures_dir / "05_cost_emissions_tradeoff.png")
    plot_load_shedding(       comparison, figures_dir / "06_load_shedding.png")
    plot_dashboard(           comparison, figures_dir / "07_dashboard_summary.png")

    print("\nGenerating per-scenario dispatch figures...")
    for scenario in config["scenario_settings"]["scenarios"]:
        print(f"\n  Scenario: {scenario}")
        n = load_network(config, scenario)

        plot_dispatch_8760(n, scenario,
            figures_dir / f"08_dispatch_8760_{scenario}.png")
        plot_supply_demand_balance(n, scenario,
            figures_dir / f"09_supply_demand_balance_{scenario}.png")
        plot_dispatch_week(n, scenario, "2025-07-14",
            "Summer week (Jul 14–20)",
            figures_dir / f"10_dispatch_summer_week_{scenario}.png")
        plot_dispatch_week(n, scenario, "2025-01-13",
            "Winter week (Jan 13–19)",
            figures_dir / f"11_dispatch_winter_week_{scenario}.png")
        plot_soc(n, scenario,
            figures_dir / f"12_battery_soc_{scenario}.png")

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
